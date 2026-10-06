# AbstractGym

A small RL-style environment that checks every step of reasoning. Each puzzle asks whether a context-free grammar generates a string; the correct procedure (a leftmost depth-first search with pruning and backtracking) is known exactly, so a model's work can be graded action by action, not just its final answer.

- **Video:** [AbstractGym: Can an LLM Execute a Program?](https://www.youtube.com/watch?v=wmclSPZ40ro) (22 min)
- **Written guide:** [flavianv.github.io/articles/abstractgym.html](https://flavianv.github.io/articles/abstractgym.html)
- **Checkpoints:** [tiny controllers](https://huggingface.co/flavianv/abstractgym-tiny-controllers) · [Qwen3-14B bracket step](https://huggingface.co/flavianv/abstractgym-qwen3-14b-bracket-step) · [Qwen3-1.7B bracket step](https://huggingface.co/flavianv/abstractgym-qwen3-1.7b-bracket-step) · [Qwen3-14B DFS step](https://huggingface.co/flavianv/abstractgym-qwen3-14b-dfs-step) · [Qwen3-14B membership](https://huggingface.co/flavianv/abstractgym-qwen3-14b-membership) · [A7 arms](https://huggingface.co/flavianv/abstractgym-a7-arms)
- **Companion series:** [Understanding Transformers](https://www.youtube.com/playlist?list=PLSQN85XNghpY), code in [ai-notes-code-public](https://github.com/flavianv/ai-notes-code-public)

This repository holds what the video and the guide rely on: the environment, the saved results behind every number, and the tests that pin those numbers.

## Install and test

The package has no runtime dependencies; the tests need `pytest`.

```bash
git clone https://github.com/flavianv/abstractgym-public
cd abstractgym-public
python3 -m pip install pytest
python3 -m pytest -q
```

## The environment

| Part | Module |
| --- | --- |
| Grammars and their six diagnostic families | `src/abstractgym/cfg.py`, `src/abstractgym/complexity.py` |
| Puzzle generator (controlled sizes, flipped yes/no pairs) | `src/abstractgym/dataset.py` |
| Independent truth: an Earley-style membership oracle | `src/abstractgym/oracle.py` |
| The canonical trace: TRY / PRN / BT / DONE / ACC | `src/abstractgym/trace.py` |
| The environment loop with a strict first-error validator | `src/abstractgym/controller.py` |
| The bracket task with an exact stack | `src/abstractgym/brackets.py` |
| Grading answers and traces separately | `src/abstractgym/evaluate.py` |
| Prompts and the experiment harness | `src/abstractgym/prompting.py`, `src/abstractgym/experiment.py`, `src/abstractgym/benchmark.py` |
| Training targets for the step experiments | `src/abstractgym/a6*.py`, `src/abstractgym/a7.py` |
| The no-cue suite (no-puzzles rebuilt without the one-letter cue) | `src/abstractgym/cuefree.py`, `src/abstractgym/complexity.py` |
| Count-preserving suites (yes and no share every count; only order decides) and their shortcut gate | `src/abstractgym/countpreserving.py` |
| Hints in the observation and a re-ask guard (off by default; prompts byte-identical when off) | `src/abstractgym/controller.py`, `src/abstractgym/experiment.py` |
| Whole-run training examples and schedules | `src/abstractgym/wholerun.py` |

The environment is used for evaluation and to generate supervised training data. RL training with the step check as the reward is the next step; it is not implemented here.

## Results

Every result file is the saved output of a run, with failures kept in every denominator. The question numbers follow the video.

| Results | What they are | Video |
| --- | --- | --- |
| `docs/results/2026-10-03/` | 240-puzzle complexity suite: direct answers, written algorithm, gym-held memory | Q1, Q3, the atlas |
| `docs/results/2026-10-03-a6/` | Bracket task: zero-shot, tiny controller, LoRA step training | Q3, Q3b |
| `docs/results/2026-10-04-x1/` … `x10/` | Follow-ups: legal-moves-only (X1), failure analysis (X2), seeds (X3), interfaces (X4), planted history (X5), raw characters (X6), DFS step training (X7), answers vs steps (X8), answer × work (X9), thinking mode (X10) | Q1–Q6 |
| `docs/cuefree_results.md`, `docs/results/cuefree/`, `experiments/cuefree/` | The no-cue suite: the same 120 yes-puzzles, no-puzzles that differ only in structure; every simple symbol check at chance | Q1 |
| `docs/countpreserving_offloading_results.md`, `docs/results/countpreserving_offloading/`, `experiments/countpreserving/`, `experiments/offloading/` | Count-preserving suites A (the six families, order-only pairs) and B (random grammars), gated by 129 shortcut rules at exactly 50%; then hints and a re-ask guard in the gym loop (H0–H5) on all four suites | Q1, Q3, by move type |
| `docs/wholerun_results.md`, `docs/wholerun_protocol.md`, `docs/results/wholerun/`, `experiments/wholerun/` | Training on whole written runs, alone and mixed with single moves; all four ways on the original and no-cue suites | Q4, Q5, the atlas |
| `docs/a7_results.md`, `docs/results/a7/`, `experiments/a7/` | A7: richer supervision and on-policy distillation, an adaptation of [Self-Play Search Distillation](https://arxiv.org/abs/2609.30936) (Molfetta et al., 2026); one-seed pilot, not a reproduction | Follow-up |

`scripts/` holds the analysis and export code the result tests use. The GPU runs used Qwen3-1.7B and Qwen3-14B with vLLM and greedy decoding; the job launchers are environment-specific and not included. `scripts/run_ray_countpreserving.py` and `scripts/run_ray_offloading.py` are kept for their settings and input checks, which the reports and tests use; the vLLM backend they call is not included.

## How the models were trained

All Qwen3 training is supervised fine-tuning of a small LoRA adapter on the frozen base model; [`scripts/train_step_lora.py`](scripts/train_step_lora.py) reproduces the recipe (`--dry-run` builds and counts the training examples without a GPU).

| Adapter (video) | Training examples | Built by |
| --- | --- | --- |
| Bracket step (Q3b) | 2,305 single moves on the development bracket cases | `abstractgym.a6_sft.training_examples` |
| DFS step, "step-trained" (Q4, atlas) | 11,002 single moves: every state of the canonical search on 64 training grammars (recursive and nested, depth ≤ 4), none from the test suite | `abstractgym.a6_objectives.dfs_examples` |
| Answers only (Q5) | yes/no examples | `abstractgym.a6_objectives.membership_examples` |
| Whole runs, and whole runs + single moves (Q4) | the 64 training searches as complete written runs, alone or mixed with the 11,002 single moves | `abstractgym.wholerun.fulltrace_examples`, `abstractgym.wholerun.training_schedule` |

Each step example is the correct-state prompt (grammar, string, current search state, built by `abstractgym.experiment.prompt_for`) and the one correct next move as JSON (from `abstractgym.controller.teacher_trajectory`); the loss covers only the target tokens (`abstractgym.a6_sft.tokenize_example`). Recipe: LoRA rank 16, alpha 32, on the attention and MLP projections; AdamW 2e-4, cosine decay with 5% warmup; effective batch 32; 3 epochs; seed 17. The step adapters never see a prompt asking for a whole written run, which is the format confound discussed in Q4.

## License

MIT
