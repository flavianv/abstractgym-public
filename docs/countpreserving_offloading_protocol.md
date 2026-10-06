# Count-preserving puzzles and operator offloading

This study follows the combined tutorial brief. It runs in three strictly
ordered parts: construct/gate Suites A and B; evaluate their four zero-shot
interfaces; then implement and test optional hints and the legality guard before
evaluating H0-H5. No new language-model training. Private repository only.

## Shortcut gate

Each suite has 120 paired yes/no problems, 240 cases. The Earley oracle supplies
membership labels and the unchanged bounded canonical controller must complete
with those labels. Every pair preserves input bags, complete grammar terminal
bags, individual production bags, rule lengths, placeholder counts, and named
symbol counts. Only order changes. Each deterministic function of these count
features consequently makes the same prediction for both labels in a pair.

Suite A retains the six families and levels 1, 2, 4, 8, 16 with four letterings.
For length and sequence, level n means n two-symbol blocks (length 2n), because
order cannot discriminate a one-symbol example. Recursion uses n two-symbol
extensions. Alternatives use the shortest balanced two-letter bag with more
than n distinct orderings, so all options preserve each letter count. Chains
have n links and a two-letter leaf. Nesting uses n openings of each of two
types, including at level one, and swaps two different adjacent closers in the
input while keeping the grammar identical. Thus its nesting count is 2n, not n;
actual accepting depth, search steps, and string length accompany every case.
This also prevents an unbalanced type-count/closer-ratio shortcut at level one.

Suite B rejection-samples 120 independent random grammars with 3-6
nonterminals, 2-3 distinct productions each, and a declared vocabulary of 3-4
letters. Each nonterminal has a terminal leaf and terminal-prefix single-child
alternatives, optionally with a suffix. This linear-CFG sampling restriction is
explicit: it is not a sample of arbitrary CFGs. Successful canonical derivation
depth is exactly 1, 2, 4, 8, or 16, with 24 pairs at each depth. The partner swaps
two different adjacent input symbols and is retained only if the oracle rejects
and canonical DFS exhausts within 512 steps, without a cut. Duplicate grammars,
different accepting depths, non-rejecting swaps, and bounded disagreements are
discarded with reason counts. No model outcomes participate in construction.

The battery includes the prototype's four symbol checks; length, placeholder,
letter-count, parity, and ratio checks; both constant predictors; and logistic
regression on grammar/input count features including individual production bags.
Four-fold predictions hold out complete pairs and fit the vectorizer only on
training rows. Fold assignment is lettering/pair index modulo four, stratified
by family and size. All rules must score within 45-55% overall, by family, by
size, and by family/size. A failed gate prohibits model submission.

Frozen data, generation/filter statistics, classifier folds, scores, gate result,
and hashes are in `experiments/countpreserving`. The gate passes with exactly
50% for all 129 checks in both suites and all nonempty cells. This establishes
count balance, not the absence of order-sensitive structural shortcuts.

```bash
PYTHONPATH=src uv run --no-project --python 3.11 \
  --with scikit-learn==1.7.2 python scripts/freeze_countpreserving.py
PYTHONPATH=src python3 -m pytest -q tests/test_countpreserving.py
```

## Model protocol

Use Qwen/Qwen3-14B revision 40c069824f4251a91eefaf281ebe4c544efd3e18,
BF16, seed 17, greedy, thinking off, no examples. The existing membership prompt,
controller prompt, and grading remain unchanged for Part 2. Direct cap 128;
complete-trace cap 4,096; next-action cap 128 and at most 512 calls; context
16,384. Native base inference has LoRA disabled. Direct and teacher-state batches
are at most 32; complete traces and live-state execution are serial. No repair
or retries in Part 2. Historical original/cue-free comparisons explicitly reuse
committed zero-shot evidence. All failures remain in every denominator.

Part 3 follows Part 2, not concurrently. It introduces default-off current-frame
fields (`untried_rules`, `prefix_matches`, `complete_match`, `next_untried`)
derived from validator logic without consulting a reference action. Enabled
prompts add exactly one protocol sentence; disabled prompts must be byte-identical
to the pre-change evidence. A repeated-TRY guard in live execution re-asks once
with the specified bookkeeping note and grades the second answer without changing
the controller. Extra calls, activations, and second-answer correctness are logged.
H0-H5 run in the same serving job per suite with matched settings, including fresh
H0. The guard is a live-execution mechanism, not a correction to raw teacher-state
accuracy. H2 teacher checks therefore have no hint or guard; H3 teacher checks
have H1's hints. This interface distinction will be explicit in the results.

The final combined report is `docs/countpreserving_offloading_results.md`, with
the shortcut gate first, complete-run/direct/teacher metrics by label, family,
and size, direct yes-rates, per-operator accuracy and error substitutions (DONE
first), guard outcomes, and tokens/timings. One seed, correlated letterings,
linear random-grammar restriction, selection bounds, and fixed decoding budgets
limit interpretation. Optional full-trace hint controls and public mirroring are
not included in the combined work order.

## Offloading implementation and replay

The completed A/B baseline jobs and the 3,840-case replay gate are recorded in
`experiments/offloading/protocol.json` before submitting any hint experiment.
`scripts/freeze_offloading.py` refuses an incomplete baseline and refuses to
replace a different frozen protocol. `scripts/run_ray_offloading.py` verifies
the frozen suites/conditions and tokenizes all teacher states, including retry
notes, before loading the model. All canonical actions and grades are unchanged.

One serving job per suite runs H0-H5 in order, each with live execution followed
by independent correct-state checks. There is no cross-case teacher batching
change: batches stay within a case and contain at most 32 independent prompts.
The four suites may run concurrently, with one B200 per suite. Responses are not
shared between conditions, even for H2/H0 and H3/H1 teacher-interface duplicates.
The guard re-asks only once at the same state. A malformed second response or a
second repeated rule is final; the extra call must fit within the 512-call cap.
Second-answer correctness is recorded only after normal controller validation.

Focused tests cover byte-identical disabled prompts against every committed
original/cue-free zero-shot call, current-frame-only facts, unchanged canonical
actions on all 960 instances, guard decisions without a reference consultation,
single-retry/final-error behavior, and call-cap enforcement. Report replay checks
every prompt, response, batch size, output cap, grade, guard annotation and count.
The report also reproduces the baseline summary from its original raw evidence.

```bash
PYTHONPATH=src:scripts python3 scripts/freeze_offloading.py
PYTHONPATH=src:scripts python3 -m pytest -q tests/test_offloading*.py
PYTHONPATH=src:scripts python3 scripts/submit_countpreserving.py \
  --study offloading --suite A --id abstractgym-offloading-a-20261005 \
  --cluster abstractgym-countpreserving-1005 --output build/offloading-20261005/A
PYTHONPATH=src:scripts python3 scripts/watch_countpreserving.py --study offloading
PYTHONPATH=src:scripts python3 scripts/report_countpreserving_offloading.py
```

The collector reconnects to accepted jobs and never submits a replacement. It
fetches complete raw evidence before invoking the report. Incomplete or missing
conditions prohibit publication. The report keeps raw teacher accuracy separate
from guard-assisted live execution, counts every failure, and exports a hashed
audit alongside operator, label/family/size, substitution, guard and cost tables.
