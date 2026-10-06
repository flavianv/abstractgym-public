# Cue-free complexity: zero-shot Qwen3-14B

All 240 direct answers and 120 new negative cases in each execution lane are complete. Every saved call was replayed through the unchanged grader. The 120 yes cases in each execution lane reuse their original results; their full instance dictionaries are unchanged.

## Main comparison

| Lane | Original suite | Cue-free suite | Cue-free yes | Cue-free no |
| --- | --- | --- | --- | --- |
| Direct answer | 236/240 (98.3%) | 204/240 (85.0%) | 116/120 | 88/120 |
| Written algorithm (full trace) | 28/240 (11.7%) | 30/240 (12.5%) | 28/120 | 2/120 |
| Gym memory (external state) | 40/240 (16.7%) | 40/240 (16.7%) | 40/120 | 0/120 |

The separate bracket direct-answer reference is **62/92 (67.4%)**. Its cases and grammar families differ, so this is contextual rather than a paired control. All failures remain in each denominator. One decoding seed (17), no training, no retries, no examples, no thinking mode.

## What this answers

Direct accuracy changes from 236/240 to 204/240 (-13.3 percentage points): 204 cases remain correct, 4 remain wrong, 32 regress, and 0 improve. There are 0 changed response texts on the 120 identical yes cases. This isolates where the observed changes occur without claiming to identify the model's internal algorithm.

On the new negative cases, written algorithm succeeds on 2/120 and gym memory on 0/120. These are complete valid executions, not merely correct final verdicts. The historical yes successes are reused rather than newly replicated.

The outcome sits between the two scenarios in the brief. Near-perfect direct accuracy does not survive cue removal, but direct answers do not collapse to chance or to the separate bracket score. All 32 regressions are newly constructed no cases. The strongest tested count shortcut gets 172/240 (71.7%), below the model's 204/240; this does not rule out a mixture of family-specific heuristics.

For the tutorial, retain the answer-versus-execution distinction with the cue-free numbers and narrow its scope to these families. The original 236/240 alone was confounded by a perfect symbol cue. The new run establishes substantial direct-answer performance without that cue, alongside weaker execution performance; it does not establish a general membership algorithm.

The claim **“every zero-shot execution success is a yes” is false on this suite**. The new successful negative executions are listed below. Each was replayed through the canonical controller; alphabet variants are zero-indexed.

| Lane | Family | Level | Variant | Task ID |
| --- | --- | --- | --- | --- |
| full_trace | alternatives | 4 | 2 | cuefree_complexity_16eea199b2ac |
| full_trace | alternatives | 8 | 2 | cuefree_complexity_fde895919107 |

## Direct answers

### Per family

| Family | All | Yes | No |
| --- | --- | --- | --- |
| terminal_length | 32/40 (80.0%) | 20/20 | 12/20 |
| chain_depth | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | 40/40 (100.0%) | 20/20 | 20/20 |
| sequence | 32/40 (80.0%) | 20/20 | 12/20 |
| recursion | 28/40 (70.0%) | 20/20 | 8/20 |
| nesting | 32/40 (80.0%) | 16/20 | 16/20 |

### Per level

| Level | All | Yes | No |
| --- | --- | --- | --- |
| 1 | 47/48 (97.9%) | 23/24 | 24/24 |
| 2 | 45/48 (93.8%) | 21/24 | 24/24 |
| 4 | 44/48 (91.7%) | 24/24 | 20/24 |
| 8 | 34/48 (70.8%) | 24/24 | 10/24 |
| 16 | 34/48 (70.8%) | 24/24 | 10/24 |

## Written algorithm (zero-shot)

### Per family

| Family | All | Yes | No |
| --- | --- | --- | --- |
| terminal_length | 20/40 (50.0%) | 20/20 | 0/20 |
| chain_depth | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | 6/40 (15.0%) | 4/20 | 2/20 |
| sequence | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | 0/40 (0.0%) | 0/20 | 0/20 |

### Per level

| Level | All | Yes | No |
| --- | --- | --- | --- |
| 1 | 12/48 (25.0%) | 12/24 | 0/24 |
| 2 | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | 5/48 (10.4%) | 4/24 | 1/24 |
| 8 | 5/48 (10.4%) | 4/24 | 1/24 |
| 16 | 4/48 (8.3%) | 4/24 | 0/24 |

## Gym memory (zero-shot)

### Per family

| Family | All | Yes | No |
| --- | --- | --- | --- |
| terminal_length | 20/40 (50.0%) | 20/20 | 0/20 |
| chain_depth | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | 4/40 (10.0%) | 4/20 | 0/20 |
| sequence | 12/40 (30.0%) | 12/20 | 0/20 |
| recursion | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | 0/40 (0.0%) | 0/20 | 0/20 |

### Per level

| Level | All | Yes | No |
| --- | --- | --- | --- |
| 1 | 8/48 (16.7%) | 8/24 | 0/24 |
| 2 | 12/48 (25.0%) | 12/24 | 0/24 |
| 4 | 8/48 (16.7%) | 8/24 | 0/24 |
| 8 | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | 8/48 (16.7%) | 8/24 | 0/24 |

## Failure accounting and call budgets

| Lane | Failure type | Cases |
| --- | --- | --- |
| direct | incorrect_answer | 36 |
| written_algorithm | invalid_or_trailing_action | 172 |
| written_algorithm | call_or_parse_error | 38 |
| gym_memory | invalid_action | 200 |

| Lane | Calls | Completion tokens | Total tokens | Length-limited calls | Historical cases reused |
| --- | --- | --- | --- | --- | --- |
| direct | 240 | 1440 | 48900 | 0 | 0 |
| written_algorithm | 240 | 167077 | 288073 | 38 | 120 |
| gym_memory | 981 | 10474 | 688784 | 0 | 120 |

These execution totals include the reused yes calls. A successful execution means a complete, valid canonical DFS trajectory and correct verdict; a correct verdict alone does not pass. Invalid actions, malformed responses, unfinished traces, and budget exhaustion remain failures. No tool-failure recovery was tested. Prompts, outputs, action order, grades, and usage are retained in the audit artifact.

## Shortcut audit and scope

The original last-terminal rule scores 240/240; all four symbol-level rules score 120/240 on the new suite. The prototype is preserved exactly. Its count shortcuts differ from the prose brief: they score 160/240 and 172/240 overall, with above-chance alternatives scores (24/40 and 28/40) and recursion at 24/40 for the latter. This suite removes the named symbol cue, not every possible shortcut. Comparing lengths is itself the membership computation for the one-production/chain families.

| Shortcut | Original | Cue-free |
| --- | --- | --- |
| always yes | 120/240 | 120/240 |
| grammar terminals subset of string symbols | 220/240 | 120/240 |
| last rule, last terminal in string | 240/240 | 120/240 |
| longest rule no longer than the string | 120/240 | 172/240 |
| same symbol set | 220/240 | 120/240 |
| some rule has as many terminals as the string | 120/240 | 160/240 |
| string symbols subset of grammar terminals | 224/240 | 120/240 |

## Reproducibility and interpretation limits

Base `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`; BF16, temperature 0, top-p 1, top-k −1, seed 17, thinking off, no LoRA. Direct output budget 128; original member_prompt preserved exactly. Written traces use 4,096 tokens; gym actions use 128 with at most 512 calls. Context 16,384. Direct batches have at most 32 calls; both execution lanes run serial calls, as in the original run. Runtime package versions match the original.

The direct set is rerun in full, whereas execution results combine historical yes outputs with newly generated no outputs. The new direct final batch has 16 cases; the original combined-membership job included added bracket cases in that batch. Prompts are independent, but small numerical batch effects are possible. Any changed outputs on identical yes cases are listed in metadata.json. No such outcome was used to change prompts or rerun cases.

The changed negative grammars alter structural/count requirements and canonical trajectory lengths as well as removing the symbol cue. Consequently, a score difference establishes sensitivity to this contrast set; it does not by itself identify which heuristic the original model used. The families, shared templates, and alphabet permutations are correlated, and there is only one seed. These results do not prove general CFG membership or arbitrary-program execution.

[Frozen protocol](../experiments/cuefree/protocol.json), [summary by family, level, and label](results/cuefree/summary.json), [raw calls and instances](results/cuefree/audit.json.gz), [metadata and positive-output comparisons](results/cuefree/metadata.json), and [shortcut scores](results/cuefree/shortcuts.json) accompany this report. Original suite data and saved results were not modified.
