"""Zero-shot count-preserving evaluation with a mandatory pre-model gate."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from abstractgym.controller import ControllerEnv, teacher_trajectory
from abstractgym.countpreserving import predictions, validate_pairs
from abstractgym.experiment import prompt_for, run_case
from a6_prompts import member_prompt

def persist(path, destination):   # the reported runs also copied each file to durable cluster storage; the public copy keeps results local
    return None

DATA = Path('experiments/countpreserving')
MODEL = 'Qwen/Qwen3-14B'
REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'
SETTINGS = dict(model=MODEL, revision=REVISION, seed=17, temperature=0,
    top_p=1, top_k=-1, thinking=False, training='none', dtype='bfloat16',
    direct_tokens=128, fulltrace_tokens=4096, step_tokens=128,
    max_calls=512, context=16384, batch_size=32, enable_lora=False,
    fulltrace_and_live_batch_size=1, retries=0, repair=False)


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    persist(path, os.environ['RAY_CLI_OUTPUT_DIR'])


def verify_inputs(folder=DATA):
    manifest = json.loads((folder / 'manifest.json').read_text())
    required = {'A.json', 'B.json', 'shortcuts.json', 'generation.json', 'gate.json'}
    if not required <= set(manifest):
        raise ValueError('incomplete shortcut manifest')
    for name, sha in manifest.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != sha:
            raise ValueError('input hash mismatch: ' + name)
    gate = json.loads((folder / 'gate.json').read_text())
    shortcuts = json.loads((folder / 'shortcuts.json').read_text())
    if not gate['passed'] or gate['model_calls'] != 0:
        raise ValueError('shortcut gate did not pass before model calls')
    suites = {}
    for suite in ('A', 'B'):
        rows = json.loads((folder / (suite + '.json')).read_text())
        if len(rows) != 240 or len(validate_pairs(rows)) != 120:
            raise ValueError('incomplete paired suite')
        checked = shortcuts[suite]
        if not checked['passed'] or checked['violations']:
            raise ValueError('shortcut gate failed: ' + suite)
        for result in checked['scores'].values():
            cells = [result['overall'], *result['by_family'].values(), *result['by_size'].values()]
            cells += [cell for sizes in result['family_size'].values() for cell in sizes.values()]
            if any(not total or not .45 <= correct / total <= .55 for correct, total in cells):
                raise ValueError('shortcut gate contains an off-chance cell')
        raw = [predictions(row) for row in rows]
        for rule in raw[0]:
            scored = sum(p[rule] == (r['target_answer'] == 'accept') for p, r in zip(raw, rows))
            if checked['scores'][rule]['overall'] != [scored, len(rows)]:
                raise ValueError('shortcut scores do not match frozen rows')
        suites[suite] = rows
    return suites, manifest


def preflight(rows, tokenizer):
    def prompt_tokens(prompt):
        return len(tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False))
    result = []
    for row in rows:
        gold = teacher_trajectory(row)
        initial = ControllerEnv(row).observation()
        prompts = [member_prompt(row), prompt_for(initial, 'full_trace')]
        prompts += [prompt_for(step['observation'], 'oracle_step') for step in gold]
        sizes = [prompt_tokens(prompt) for prompt in prompts]
        # Record output-budget infeasibility; never remove such cases from scoring.
        target = json.dumps([step['action'] for step in gold], separators=(',', ':'))
        target_tokens = len(tokenizer.encode(target, add_special_tokens=False))
        if sizes[0] + 128 > 16384 or sizes[1] + 4096 > 16384 or max(sizes[2:]) + 128 > 16384:
            raise ValueError('context budget insufficient: ' + row['task_id'])
        if len(gold) > 512:
            raise ValueError('canonical action budget insufficient: ' + row['task_id'])
        result.append(dict(task_id=row['task_id'], teacher_actions=len(gold),
            direct_prompt_tokens=sizes[0], fulltrace_prompt_tokens=sizes[1],
            max_teacher_prompt_tokens=max(sizes[2:]), canonical_compact_tokens=target_tokens,
            compact_trace_over_output_budget=target_tokens > 4096))
    return dict(cases=result, max_teacher_prompt_tokens=max(r['max_teacher_prompt_tokens'] for r in result),
        compact_trace_over_output_budget=sum(r['compact_trace_over_output_budget'] for r in result),
        context_check_passed=True, policy='retain all cases and fixed output caps; no truncation or filtering')


def evaluate(root, rows, adapter):
    started = time.perf_counter()
    replies = adapter.generate_batch([member_prompt(row) for row in rows])
    if len(replies) != len(rows):
        raise ValueError('direct batch response count mismatch')
    records = []
    for row, reply in zip(rows, replies):
        try:
            parsed = json.loads(reply['text'])
        except ValueError:
            parsed = None
        expected = {'accept': row['target_answer'] == 'accept'}
        success = isinstance(parsed, dict) and type(parsed.get('accept')) is bool and parsed == expected
        records.append(dict(task_id=row['task_id'], family=row['family'], target_answer=row['target_answer'],
            difficulty=row['difficulty'], mode='direct', prompt=member_prompt(row), success=success, **reply))
    save(root, 'direct.json', records)
    print('direct_complete ' + json.dumps(dict(cases=len(records), seconds=time.perf_counter() - started)), flush=True)
    out = root / 'execution.jsonl'
    count = 0
    try:
        with out.open('x') as stream:
            for mode in ('full_trace', 'external_state', 'oracle_step'):
                begin = time.perf_counter()
                for index, row in enumerate(rows, 1):
                    record = run_case(row, mode, adapter, max_calls=512)
                    stream.write(json.dumps(record, sort_keys=True) + '\n')
                    stream.flush()
                    count += 1
                    if index % 12 == 0:
                        persist(out, os.environ['RAY_CLI_OUTPUT_DIR'])
                        print('evaluation_progress ' + json.dumps(dict(mode=mode, cases=index,
                            records=count, seconds=time.perf_counter() - begin)), flush=True)
                print('mode_complete ' + json.dumps(dict(mode=mode, cases=len(rows),
                    seconds=time.perf_counter() - begin)), flush=True)
    finally:
        if out.exists():
            persist(out, os.environ['RAY_CLI_OUTPUT_DIR'])
    save(root, 'completion.json', dict(completed=True, direct_cases=len(records), execution_cases=count,
        seconds=time.perf_counter() - started))
    print('baseline_complete', flush=True)


def main():
    from run_ray_cfg import VLLMAdapter   # the vLLM serving adapter used for the reported runs (GPU + vLLM); not part of the public repo
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('A', 'B'), required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    suites, manifest = verify_inputs()
    if not os.environ.get('RAY_CLI_OUTPUT_DIR'):
        raise RuntimeError('durable output directory required')
    from transformers import AutoTokenizer
    root = Path(tempfile.mkdtemp(prefix='abstractgym-countpreserving-' + args.suite + '-'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    save(root, 'preflight.json', preflight(suites[args.suite], tokenizer))
    save(root, 'runtime.json', dict(suite=args.suite, settings=SETTINGS, manifest=manifest,
        packages={p: importlib.metadata.version(p) for p in ('torch', 'vllm', 'transformers', 'huggingface_hub')},
        hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
            '--format=csv,noheader'], text=True).strip(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src/abstractgym', 'scripts') for p in Path(folder).glob('*.py')}))
    print('preflight_passed ' + args.suite, flush=True)
    if args.preflight_only:
        return
    adapter = VLLMAdapter(MODEL, REVISION, 4096, 16384, 128, batch_size=32, enable_lora=False)
    evaluate(root, suites[args.suite], adapter)


if __name__ == '__main__':
    main()
