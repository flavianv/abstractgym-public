"""Train the LoRA adapters used in the AbstractGym video and guide.

One script, three training sets, the same recipe for all of them:

  --task bracket_step   the bracket step adapter (Q3b): 2,305 single-move examples
                        from the development bracket cases (abstractgym.a6_sft)
  --task dfs_step       the DFS step adapter (Q4, "step-trained"): 11,002 single-move
                        examples, one per state of the canonical search on the 64
                        benchmark training grammars (abstractgym.a6_objectives.dfs_examples)
  --task membership     the answers-only adapter (Q5): yes/no examples
                        (abstractgym.a6_objectives.membership_examples)

Every example is a prompt plus a short target. For the step tasks the prompt is exactly
what the correct-state check shows the model (grammar, string, current search state)
and the target is the one correct next move as JSON, e.g. {"op":"PRN","reason":"pref"}.
The loss is on the target tokens only.

Recipe (as in the reported runs): ordinary LoRA, rank 16, alpha 32, dropout 0, on
q/k/v/o and gate/up/down projections; BF16 base; AdamW, learning rate 2e-4, weight
decay 0.01, gradient clipping 1, 5% warmup then cosine decay; micro-batch 8 with 4
accumulation steps (effective batch 32); 3 epochs, shuffled with seed + epoch.
The reported Q5 run instead matched the processed-token budget of the step adapter
with label-balanced sampling; that variant is not reproduced here.

Needs a CUDA GPU and `pip install torch transformers peft`. Example:

  python3 scripts/train_step_lora.py --task dfs_step --model Qwen/Qwen3-14B \\
      --out runs/dfs_step --gradient-checkpointing

Evaluation uses the same prompts through `abstractgym.experiment.run_suite` (complexity
suite) or the bracket helpers in `abstractgym.a6`; serve the saved adapter natively on
the original base model (a BF16 merge failed a logit-parity check in our runs).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from abstractgym.a6_objectives import dfs_examples, membership_examples  # noqa: E402
from abstractgym.a6_sft import tokenize_example, training_examples  # noqa: E402

REVISIONS = {'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
             'Qwen/Qwen3-14B': '40c069824f4251a91eefaf281ebe4c544efd3e18'}
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
TASKS = {'bracket_step': training_examples, 'dfs_step': dfs_examples, 'membership': membership_examples}


def collate(rows, pad_id, device):
    import torch
    n = max(len(r['input_ids']) for r in rows)
    pad = lambda seq, v: seq + [v] * (n - len(seq))
    return {'input_ids': torch.tensor([pad(r['input_ids'], pad_id) for r in rows], device=device),
            'labels': torch.tensor([pad(r['labels'], -100) for r in rows], device=device),
            'attention_mask': torch.tensor([pad([1] * len(r['input_ids']), 0) for r in rows], device=device)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--task', choices=sorted(TASKS), required=True)
    ap.add_argument('--model', default='Qwen/Qwen3-14B', choices=sorted(REVISIONS))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--gradient-checkpointing', action='store_true', help='needed for 14B on the dfs_step task on one B200')
    ap.add_argument('--dry-run', action='store_true', help='build and count the training examples, then stop (no GPU needed)')
    args = ap.parse_args()

    examples = TASKS[args.task]()
    print(f'{args.task}: {len(examples)} examples, {len({e["prompt"] for e in examples})} unique prompts')
    if args.dry_run:
        print('first example:\n  prompt (end): ...' + examples[0]['prompt'][-200:].replace('\n', ' ') + '\n  target: ' + examples[0]['answer'])
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    torch.manual_seed(args.seed); random.seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    rev = REVISIONS[args.model]
    tok = AutoTokenizer.from_pretrained(args.model, revision=rev)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    data = [tokenize_example(tok, e) for e in examples]

    epochs = []
    for epoch in range(args.epochs):
        order = list(range(len(data))); random.Random(args.seed + epoch).shuffle(order)
        epochs.append([order[i:i + 8] for i in range(0, len(order), 8)])

    base = AutoModelForCausalLM.from_pretrained(args.model, revision=rev, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=16, lora_alpha=32, lora_dropout=0, target_modules=TARGETS))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=.01)
    total = sum(math.ceil(len(b) / 4) for b in epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total * .05), total)

    config = dict(task=args.task, model=args.model, revision=rev, seed=args.seed, epochs=args.epochs, examples=len(data),
                  lora_rank=16, lora_alpha=32, lora_dropout=0, target_modules=TARGETS, micro_batch=8, gradient_accumulation=4,
                  lr=2e-4, weight_decay=.01, gradient_clip=1., warmup_fraction=.05, schedule='cosine', dtype='bfloat16',
                  response_only_loss=True, optimizer_updates=total,
                  supervised_tokens_per_epoch=sum(r['answer_tokens'] for r in data))
    history, start, updates = [], time.perf_counter(), 0
    for epoch, batches in enumerate(epochs):
        model.train()
        for g in range(0, len(batches), 4):
            group = batches[g:g + 4]; optimizer.zero_grad(); total_loss = 0.
            for idx in group:
                loss = model(**collate([data[i] for i in idx], tok.pad_token_id, 'cuda')).loss
                assert torch.isfinite(loss)
                (loss / len(group)).backward(); total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step(); updates += 1
            history.append(dict(update=updates, epoch=epoch + 1, loss=total_loss / len(group), lr=scheduler.get_last_lr()[0]))
            if updates % 20 == 0:
                print(json.dumps(history[-1]), flush=True)
    config['training_seconds'] = time.perf_counter() - start
    model.eval(); model.save_pretrained(args.out / 'adapter'); tok.save_pretrained(args.out / 'adapter')
    (args.out / 'training.json').write_text(json.dumps(config, indent=2) + '\n')
    (args.out / 'training_history.json').write_text(json.dumps(history) + '\n')
    print('saved', args.out / 'adapter')


if __name__ == '__main__':
    main()
