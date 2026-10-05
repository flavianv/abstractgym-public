# A7: does richer execution evidence improve the learned step?

This is a controlled AbstractGym adaptation of [SPSD](https://arxiv.org/abs/2609.30936), frozen before the new held-out evaluations. It is not a replication of the paper's games, expert, or training configuration.

The six training arms cross three forms of supervision with two objectives:

| Supervision | SFT | On-policy distillation |
|---|---|---|
| Next action only | Reference-token cross entropy | Evidence-conditioned frozen teacher, on student tokens |
| Action plus successor state | Same, with a separate successor auxiliary task | Same teacher objective on the richer task |
| Action plus branch consequences | Same, with a separate branch auxiliary task | Same teacher objective on the richer task |

Every arm starts from Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, ordinary LoRA rank 16, seed 17. PiSSA and merged serving are excluded following the earlier failed parity checks. All evaluation conditions, including base, enable the native LoRA engine and use the same job-scoped B200 kernel workaround.

The training pool contains eight states from each of the 64 original training trajectories: one representative of each available action type, then deterministic hash-selected positions. This is 512 states, not 512 independent trajectories. Each state produces its original next-action row plus either a duplicate action row or a distinct auxiliary row. Thus the comparison holds the state pool, row count, updates, and initialization fixed. It does **not** hold action practice, target length, or FLOPs fixed.

Each full arm receives one epoch of 1,024 rows: 128 AdamW updates, effective batch 8, microbatch 1, learning rate 5e-5, cosine decay, 5% warmup, weight decay .01, gradient clipping 1. This is a small matched pilot; the much larger historical X7 run is contextual evidence, not its matched action-only control. No checkpoint selection or hyperparameter tuning uses held-out results.

For the successor auxiliary task, the target includes the exact complete observation after the correct action. For TRY branch evidence, the target includes the first at most two untried applicable productions, their child forms, and at most three subsequent canonical steps within each branch. A hypothetical alternative expansion may violate canonical production order; it is explicitly a grammar-legal counterfactual, **not** a valid next action under the test protocol. Unfinished branches remain unresolved. Non-TRY states have a single verified successor and no alternative branches.

The distillation teacher is the frozen original base, conditioned on the verified training reference. The student samples its own continuation (temperature 1, top-p .95; 128 tokens for actions, 1,024 for auxiliary outputs). Both networks score exactly the same generated response prefixes, using their different prompts. The loss is full-vocabulary forward KL from teacher to student, averaged over response tokens and then equally over rows. Teacher gradients are disabled. Malformed or truncated rollouts are retained and counted. Both teacher and student use thinking off; this is an explicit adaptation of SPSD, whose described settings differ. The teacher gets privileged information only during training.

All arms and a fresh base control receive the unchanged 240-case complexity suite and a newly frozen 192-case suite at levels 3, 6, 12, 24 with renamed terminals and nonterminals. The latter combines length/depth and representation shifts in the same six families. It does not isolate those shifts or test unseen grammar families. The historical suite has already influenced my research questions; it is not a virgin test set.

Evaluation preserves full-trace, teacher-state, and live external-state interfaces, plus direct membership. There is no correction, retry, output repair, or privileged evidence at inference. Independent calls are batched in groups of at most 32, retaining every teacher-state occurrence. Responses are replayed through the existing strict grader; live runs stop at their first error. Tests check equivalence to the serial grader.

A separate untrained thinking control raises the full-trace/step/direct output caps to 16,384/4,096/8,192 and context to 32,768. The ordinary caps remain 4,096/128/128 and context 16,384. Both are greedy, with unchanged task prompts. This addresses the earlier fixed-cap thinking truncations; it does not claim optimized reasoning decoding.

Primary outcomes are complete execution success with every failure retained, all-state correctness per case, teacher action accuracy by operation, and direct membership. Report malformed/truncated outputs, first errors, actual completion and reasoning tokens, student/teacher/rollout token costs, and wall time. Do not infer general program execution from success on a finite grammar suite. One seed and one budget cannot establish that an objective is intrinsically superior or that richer supervision cannot work.

The machine-readable protocol and data hashes are in `experiments/a7/`. Technical smoke runs use only development states, exercise both objectives, and discard their checkpoints. Full runs proceed only after parity, gradient, and checkpoint reload checks pass.
