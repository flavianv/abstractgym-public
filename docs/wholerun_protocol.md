# Whole written runs: frozen experiment protocol

The question is whether practicing complete canonical DFS traces lets
Qwen3-14B maintain search state in its own output. This follows the brief
`ABSTRACTGYM_WHOLE_RUN_EXPERIMENT.md` in the understanding-transformers tutorial.
The report will be written to `docs/wholerun_results.md` after the runs finish.

## Arms and training

All new arms start independently from `Qwen/Qwen3-14B`, revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, with ordinary LoRA and seed 17.
Use X7's 64 benchmark-build training trajectories only; no held-out examples
enter training or checkpoint selection.

| Arm | Examples | Budget |
| --- | --- | --- |
| fulltrace_sft | 64 complete traces | Three epochs |
| fulltrace_sft_matched | Same 64 complete traces | Repeat deterministically shuffled epochs until the first effective batch reaching 295,089 supervised target tokens |
| mixed_sft | 11,002 X7 next-action examples plus 64 complete traces, jointly shuffled | Three epochs |
| dfs_lora | Historical X7 adapter | Reuse; no retraining |
| zeroshot | Original pinned base | Reuse existing outputs; generate missing correct-state negatives |

The full-trace target is the raw JSON action array through global completion.
Its final ACC or root-failing action expresses the verdict under the unchanged
controller protocol. Do not append a separate answer schema or alter prompts.
Loss covers answer tokens plus EOS; prompt tokens are masked. The matched arm
matches **target tokens**, not optimizer updates or processed prompt tokens.
Report actual totals and the final effective batch's possible overshoot.

X7 settings: BF16 base, FP32 adapters, rank 16, alpha 32, dropout 0,
q/k/v/o/gate/up/down projection targets; AdamW at 2e-4, weight decay .01,
cosine schedule, 5% warmup rounded upward, gradient clip 1; microbatch 8,
four accumulation steps, effective batch 32. Incomplete final batches use the
same X7 grouping rule. Gradient checkpointing is enabled. No sequence packing
or truncation is allowed. If memory forces a batch change, record it explicitly.
Initialization and saved-checkpoint reload parity use fixed training prefixes.

## Evaluation

Each arm has 240 cases in each of the original and frozen cue-free suites.
Use the unchanged membership prompt and canonical DFS controller grader.

| Lane | Completion cap | Calls per case | Success |
| --- | --- | --- | --- |
| Direct answer | 128 | 1 | Correct boolean membership JSON |
| Written run | 4,096 | 1 | Complete valid canonical trajectory and correct verdict |
| Gym memory | 128 | At most 512 | Complete valid canonical live rollout and correct verdict |
| Correct-state check | 128 | At most 512 | Every independently supplied canonical state gets its correct action |

Context is 16,384; greedy decoding, top-p 1, top-k -1, seed 17, thinking off.
Direct and correct-state batches have at most 32 independent calls; written
and live execution are serial, as in X7. No examples, retries, repairs, prompt
changes, or result-driven checkpoint choice. Persist every failed call.
All 240 cases remain in every denominator. Correct-state checks are teacher
state diagnostics, not deployed rollouts; also report their action accuracy.

Original-suite reference outputs are reused. Cue-free zero-shot direct and
written/live outputs reuse `docs/results/cuefree`. The identical cue-free yes
instances reuse original X7/zero-shot execution and correct-state outputs;
generate missing cue-free negatives and X7 direct outputs. Identify all reuse
explicitly. Zero-shot serving keeps its historical disabled LoRA engine;
trained arms use native adapters and the same job-scoped B200 PDL workaround
as X7. This runtime difference limits causal baseline comparisons.

Skip the optional bracket suite because it uses a different action protocol
and the brief excludes introducing new bracket training examples.

## Failure and first-error diagnostics

Preserve unchanged grades. Partition failed written runs, in order, into:

1. Length-limited loop: completion hits the cap and its readable action prefix
   contains a repeated block of one to eight actions at least three times.
   Report the observed opcodes; the brief's DONE/BT wording also covers X7
   loops containing TRY/PRN.
2. One move then stop: one complete action object and a non-cap stop.
3. Malformed/invalid/trailing: broken JSON array or strict controller violation.
4. Other incomplete: all remaining unsuccessful outputs.

For malformed arrays, a JSON decoder reads only consecutive complete values
from the beginning for diagnostics; non-object values are invalid actions.
It never repairs, rescues, or regrades an
output. Record the correct prefix length, first wrong action if observable,
and its zero-based index divided by canonical trace length. A correct prefix
that stops has a **missing** move, not an observed wrong move. Keep that group
and unreadable prefixes explicit rather than imputing an error at zero.
Pair each failed written case with its gym-memory first-error position on the
same instance. Report first errors and missing moves by family and trace length.

## Artifacts and interpretation

`experiments/wholerun` freezes the protocol, examples, two suites, and hashes of
historical evidence. Runtime metadata retains software, hardware, source and
adapter hashes; raw calls, per-case grades, token counts, timings, and paired
diagnostics accompany the report under `docs/results/wholerun`.

The results report follows `docs/cuefree_results.md`: main comparison, what
the results answer, per-family and per-level tables, failure/cost accounting,
and reproducibility/limitations. The machine-readable atlas retains every
family-by-level-by-label cell. The mixed arm is compared with X7's historical
133/240 gym-memory result. One seed and correlated templates limit inference;
late wrong moves are evidence consistent with state-carrying difficulty, not
proof of an internal mechanism. Early stop/format failures remain a separate
explanation. No thinking, RL, new grammar families, or edits to old results.

## Compute orchestration

The initial launcher starts the fulltrace_sft arm on the existing single-B200
cluster. The other two training arms run independently on a dedicated cluster
with at most two one-B200 workers, using the same pinned container image.
Peak experimental allocation is three B200s; calls within each written/live
lane remain serial. After the first arm's completion artifact is saved, a
local handoff watcher stops the initial sequential launcher before the next
arm reaches evaluation, preventing duplicate evaluation calls. Any partial
training for the abandoned sequential stage is excluded from model results
and identified in execution metadata. References then run on the freed
original cluster. The dedicated cluster is removed after artifact retrieval;
durable owner-scoped HDFS outputs are retained.

## Reproduction

The frozen examples and suite files are committed; training needs the pinned
GPU runtime, Coder access, and an owned Ray cluster. To rebuild the same data
and submit a fresh serial replication on a ready cluster:

```bash
PYTHONPATH=src python3 scripts/freeze_wholerun.py
python3 scripts/submit_wholerun.py \
  --id abstractgym-wholerun-replica \
  --cluster abstractgym-cfg-pilot \
  --output build/wholerun-replica
```

Use `--arm fulltrace_sft`, `--arm fulltrace_sft_matched`, or `--arm mixed_sft`
for independent jobs, and `--references` for the two reused conditions. Use a
fresh submission ID and destination for each replication. The launch previews
and archive checksums are saved before submission. The actual study's four
accepted jobs and the intentional sequential-launcher stop are retained in
`docs/results/wholerun/execution_manifest.json`.

Fetch text artifacts with `scripts/fetch_ray_artifacts.py`; keep each arm's
runtime, training/checkpoint metadata, completion record, direct responses,
and execution JSONL under `build/wholerun-20261005/results/<arm>/`. Then:

For `dfs_lora`, also fetch the historical X7 `checkpoint_hashes.json` into that
arm's folder; the report checks it against the committed X7 training metadata.

```bash
PYTHONPATH=src python3 scripts/report_wholerun.py
python3 scripts/export_wholerun_manifest.py
PYTHONPATH=src python3 -m pytest -q
```

The full report refuses missing conditions, incomplete case coverage, changed
historical evidence, prompt/cap mismatches, or grades that fail raw-call replay.
