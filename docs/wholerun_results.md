# Whole written runs: Qwen3-14B LoRA

All three new adapters and both reference conditions have complete four-lane results on both 240-case suites. Every saved response was replayed through the unchanged grader. Original-suite reference outputs and identical cue-free yes executions are reused explicitly; all new-arm cases are freshly evaluated.

## Main comparison

### Original suite

| Arm | Direct /240 | Written /240 | Gym /240 | Correct-state /240 |
| --- | --- | --- | --- | --- |
| zeroshot | 236/240 (98.3%) | 28/240 (11.7%) | 40/240 (16.7%) | 40/240 (16.7%) |
| dfs_lora | 237/240 (98.8%) | 0/240 (0.0%) | 133/240 (55.4%) | 133/240 (55.4%) |
| fulltrace_sft | 239/240 (99.6%) | 66/240 (27.5%) | 29/240 (12.1%) | 29/240 (12.1%) |
| fulltrace_sft_matched | 239/240 (99.6%) | 87/240 (36.2%) | 25/240 (10.4%) | 25/240 (10.4%) |
| mixed_sft | 233/240 (97.1%) | 100/240 (41.7%) | 156/240 (65.0%) | 156/240 (65.0%) |

### Cuefree suite

| Arm | Direct /240 | Written /240 | Gym /240 | Correct-state /240 |
| --- | --- | --- | --- | --- |
| zeroshot | 204/240 (85.0%) | 30/240 (12.5%) | 40/240 (16.7%) | 40/240 (16.7%) |
| dfs_lora | 210/240 (87.5%) | 0/240 (0.0%) | 118/240 (49.2%) | 118/240 (49.2%) |
| fulltrace_sft | 202/240 (84.2%) | 77/240 (32.1%) | 31/240 (12.9%) | 32/240 (13.3%) |
| fulltrace_sft_matched | 200/240 (83.3%) | 87/240 (36.2%) | 25/240 (10.4%) | 25/240 (10.4%) |
| mixed_sft | 194/240 (80.8%) | 89/240 (37.1%) | 128/240 (53.3%) | 127/240 (52.9%) |

All failures remain in every denominator. One training/decoding seed (17), thinking off, greedy decoding, no examples, no retries, and no repairs. Written and gym success require complete valid canonical DFS execution; direct answer accuracy is a separate invocation. Correct-state checks supply each canonical state independently and require every action to be correct; they do not establish autonomous completion. There are no exact grammar/input matches between the 64 training instances and either evaluation suite.

## What this answers

The original canonical traces span 2-103 actions; 0 targets exceed the 512-action-call budget.

The cuefree canonical traces span 2-73 actions; 0 targets exceed the 512-action-call budget.

On original, written runs are zeroshot: 28/240, dfs_lora: 0/240, fulltrace_sft: 66/240, fulltrace_sft_matched: 87/240, mixed_sft: 100/240.

Mixed training gets 156/240 gym-memory completions versus the matched-suite X7 reference 133/240 (+23 cases), with 100/240 written completions. The original X7 reference is 133/240.

Written mixed-training completions improve by 72 cases over zero-shot (28/240), but remain 56 cases below the same adapter with computer-held state. Direct verdict accuracy is 233/240 versus 236/240 zero-shot; better execution does not imply better direct verdicts.

On cuefree, written runs are zeroshot: 30/240, dfs_lora: 0/240, fulltrace_sft: 77/240, fulltrace_sft_matched: 87/240, mixed_sft: 89/240.

Mixed training gets 128/240 gym-memory completions versus the matched-suite X7 reference 118/240 (+10 cases), with 89/240 written completions. The original X7 reference is 133/240.

Written mixed-training completions improve by 59 cases over zero-shot (30/240), but remain 39 cases below the same adapter with computer-held state. Direct verdict accuracy is 194/240 versus 204/240 zero-shot; better execution does not imply better direct verdicts.

This is an intermediate outcome, not a clean confirmation of either extreme in the brief. Whole-run practice materially improves complete written executions, so X7's zero cannot by itself establish an inability to execute in the model's own output. The mixed arm retains and improves the computer-held-state gain, while written completion remains substantially lower. The tested full-only budgets are just six and eight optimizer updates; the mixed arm also weights its rare full examples differently. This does not establish an optimization-independent state-carrying limit.

The tables remove the specific absence-of-whole-run-training confound at the tested budgets. Assess the written-run gains alongside failure categories and paired first-error positions below. A later observed mistake in long traces is consistent with difficulty maintaining state; early format errors or a correct prefix that stops support a different explanation. These outputs do not identify an internal mechanism, and this one-seed study does not establish a general CFG membership or arbitrary-program execution procedure.

The whole-trace-only arms saw no single-step output targets. Low next-action or gym-memory scores for those arms can therefore reflect the reverse interface-format confound. The mixed arm trains both interfaces and provides the most direct test of gaining written completions while retaining X7's gym-memory behavior. Training amounts and loss weighting still differ, as recorded below.

## Training amount

| Arm | Rows | Epochs touched | Updates | Target tokens | Processed tokens | Train seconds | Last loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fulltrace_sft | 64 | 3 | 6 | 229845 | 365499 | 151 | 0.0404701 |
| fulltrace_sft_matched | 64 | 4 | 8 | 306460 | 487332 | 204 | 0.0331728 |
| mixed_sft | 11066 | 3 | 1038 | 524934 | 28115808 | 7438 | 1.31012e-07 |

X7 processed 295,089 target tokens including EOS, 27,750,309 nonpadding tokens, and 1,032 optimizer updates. Its historical training took 6957 seconds; that adapter is reused, not retrained. The matched arm matches target tokens at the first effective batch reaching that budget; it does not match optimizer updates. Full targets contain the unchanged raw action array through completion, whose final action supplies the controller verdict. The mixed arm includes every X7 state example plus all 64 full traces, shuffled jointly in each of three epochs.

The matched arm exceeds the target-token budget by 11,371 tokens (3.85%). Matching target tokens leaves a large optimizer-update difference because whole traces aggregate many actions into each example. Loss is averaged over unmasked tokens within each microbatch, following X7. In mixed training the 64 full traces are only 0.58% of examples; their length and microbatch loss weighting mean row share and target-token share are different quantities.

## Original: Direct answer

### Per arm and label

| Arm | All | Yes | No | Action accuracy |
| --- | --- | --- | --- | --- |
| zeroshot | 236/240 (98.3%) | 116/120 | 120/120 | n/a |
| dfs_lora | 237/240 (98.8%) | 118/120 | 119/120 | n/a |
| fulltrace_sft | 239/240 (99.6%) | 119/120 | 120/120 | n/a |
| fulltrace_sft_matched | 239/240 (99.6%) | 119/120 | 120/120 | n/a |
| mixed_sft | 233/240 (97.1%) | 118/120 | 115/120 | n/a |

### Per family

| Family | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 40/40 (100.0%) | 20/20 | 20/20 |
| terminal_length | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| terminal_length | fulltrace_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| terminal_length | fulltrace_sft_matched | 40/40 (100.0%) | 20/20 | 20/20 |
| terminal_length | mixed_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| chain_depth | zeroshot | 40/40 (100.0%) | 20/20 | 20/20 |
| chain_depth | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| chain_depth | fulltrace_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| chain_depth | fulltrace_sft_matched | 40/40 (100.0%) | 20/20 | 20/20 |
| chain_depth | mixed_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | zeroshot | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | fulltrace_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | fulltrace_sft_matched | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | mixed_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| sequence | zeroshot | 40/40 (100.0%) | 20/20 | 20/20 |
| sequence | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| sequence | fulltrace_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| sequence | fulltrace_sft_matched | 40/40 (100.0%) | 20/20 | 20/20 |
| sequence | mixed_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| recursion | zeroshot | 40/40 (100.0%) | 20/20 | 20/20 |
| recursion | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| recursion | fulltrace_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| recursion | fulltrace_sft_matched | 40/40 (100.0%) | 20/20 | 20/20 |
| recursion | mixed_sft | 39/40 (97.5%) | 20/20 | 19/20 |
| nesting | zeroshot | 36/40 (90.0%) | 16/20 | 20/20 |
| nesting | dfs_lora | 37/40 (92.5%) | 18/20 | 19/20 |
| nesting | fulltrace_sft | 39/40 (97.5%) | 19/20 | 20/20 |
| nesting | fulltrace_sft_matched | 39/40 (97.5%) | 19/20 | 20/20 |
| nesting | mixed_sft | 34/40 (85.0%) | 18/20 | 16/20 |

### Per level

| Level | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| 1 | zeroshot | 47/48 (97.9%) | 23/24 | 24/24 |
| 1 | dfs_lora | 47/48 (97.9%) | 24/24 | 23/24 |
| 1 | fulltrace_sft | 48/48 (100.0%) | 24/24 | 24/24 |
| 1 | fulltrace_sft_matched | 48/48 (100.0%) | 24/24 | 24/24 |
| 1 | mixed_sft | 47/48 (97.9%) | 24/24 | 23/24 |
| 2 | zeroshot | 45/48 (93.8%) | 21/24 | 24/24 |
| 2 | dfs_lora | 46/48 (95.8%) | 22/24 | 24/24 |
| 2 | fulltrace_sft | 47/48 (97.9%) | 23/24 | 24/24 |
| 2 | fulltrace_sft_matched | 47/48 (97.9%) | 23/24 | 24/24 |
| 2 | mixed_sft | 45/48 (93.8%) | 22/24 | 23/24 |
| 4 | zeroshot | 48/48 (100.0%) | 24/24 | 24/24 |
| 4 | dfs_lora | 48/48 (100.0%) | 24/24 | 24/24 |
| 4 | fulltrace_sft | 48/48 (100.0%) | 24/24 | 24/24 |
| 4 | fulltrace_sft_matched | 48/48 (100.0%) | 24/24 | 24/24 |
| 4 | mixed_sft | 48/48 (100.0%) | 24/24 | 24/24 |
| 8 | zeroshot | 48/48 (100.0%) | 24/24 | 24/24 |
| 8 | dfs_lora | 48/48 (100.0%) | 24/24 | 24/24 |
| 8 | fulltrace_sft | 48/48 (100.0%) | 24/24 | 24/24 |
| 8 | fulltrace_sft_matched | 48/48 (100.0%) | 24/24 | 24/24 |
| 8 | mixed_sft | 47/48 (97.9%) | 24/24 | 23/24 |
| 16 | zeroshot | 48/48 (100.0%) | 24/24 | 24/24 |
| 16 | dfs_lora | 48/48 (100.0%) | 24/24 | 24/24 |
| 16 | fulltrace_sft | 48/48 (100.0%) | 24/24 | 24/24 |
| 16 | fulltrace_sft_matched | 48/48 (100.0%) | 24/24 | 24/24 |
| 16 | mixed_sft | 46/48 (95.8%) | 24/24 | 22/24 |

## Original: Written run

### Per arm and label

| Arm | All | Yes | No | Action accuracy |
| --- | --- | --- | --- | --- |
| zeroshot | 28/240 (11.7%) | 28/120 | 0/120 | 294/3860 |
| dfs_lora | 0/240 (0.0%) | 0/120 | 0/120 | 231/3860 |
| fulltrace_sft | 66/240 (27.5%) | 57/120 | 9/120 | 954/3860 |
| fulltrace_sft_matched | 87/240 (36.2%) | 64/120 | 23/120 | 1222/3860 |
| mixed_sft | 100/240 (41.7%) | 65/120 | 35/120 | 1701/3860 |

### Per family

| Family | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| terminal_length | fulltrace_sft | 24/40 (60.0%) | 20/20 | 4/20 |
| terminal_length | fulltrace_sft_matched | 28/40 (70.0%) | 20/20 | 8/20 |
| terminal_length | mixed_sft | 18/40 (45.0%) | 18/20 | 0/20 |
| chain_depth | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | fulltrace_sft | 19/40 (47.5%) | 19/20 | 0/20 |
| chain_depth | fulltrace_sft_matched | 16/40 (40.0%) | 16/20 | 0/20 |
| chain_depth | mixed_sft | 18/40 (45.0%) | 18/20 | 0/20 |
| alternatives | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| alternatives | fulltrace_sft | 19/40 (47.5%) | 14/20 | 5/20 |
| alternatives | fulltrace_sft_matched | 34/40 (85.0%) | 20/20 | 14/20 |
| alternatives | mixed_sft | 35/40 (87.5%) | 19/20 | 16/20 |
| sequence | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| sequence | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| sequence | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| sequence | fulltrace_sft_matched | 4/40 (10.0%) | 4/20 | 0/20 |
| sequence | mixed_sft | 6/40 (15.0%) | 6/20 | 0/20 |
| recursion | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | fulltrace_sft_matched | 5/40 (12.5%) | 4/20 | 1/20 |
| recursion | mixed_sft | 15/40 (37.5%) | 4/20 | 11/20 |
| nesting | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | mixed_sft | 8/40 (20.0%) | 0/20 | 8/20 |

### Per level

| Level | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| 1 | zeroshot | 12/48 (25.0%) | 12/24 | 0/24 |
| 1 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 1 | fulltrace_sft | 16/48 (33.3%) | 16/24 | 0/24 |
| 1 | fulltrace_sft_matched | 21/48 (43.8%) | 20/24 | 1/24 |
| 1 | mixed_sft | 26/48 (54.2%) | 18/24 | 8/24 |
| 2 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 2 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 2 | fulltrace_sft | 12/48 (25.0%) | 12/24 | 0/24 |
| 2 | fulltrace_sft_matched | 14/48 (29.2%) | 12/24 | 2/24 |
| 2 | mixed_sft | 22/48 (45.8%) | 10/24 | 12/24 |
| 4 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 4 | fulltrace_sft | 14/48 (29.2%) | 11/24 | 3/24 |
| 4 | fulltrace_sft_matched | 20/48 (41.7%) | 12/24 | 8/24 |
| 4 | mixed_sft | 21/48 (43.8%) | 14/24 | 7/24 |
| 8 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 8 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 8 | fulltrace_sft | 11/48 (22.9%) | 9/24 | 2/24 |
| 8 | fulltrace_sft_matched | 19/48 (39.6%) | 12/24 | 7/24 |
| 8 | mixed_sft | 15/48 (31.2%) | 11/24 | 4/24 |
| 16 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 16 | fulltrace_sft | 13/48 (27.1%) | 9/24 | 4/24 |
| 16 | fulltrace_sft_matched | 13/48 (27.1%) | 8/24 | 5/24 |
| 16 | mixed_sft | 16/48 (33.3%) | 12/24 | 4/24 |

## Original: Gym memory

### Per arm and label

| Arm | All | Yes | No | Action accuracy |
| --- | --- | --- | --- | --- |
| zeroshot | 40/240 (16.7%) | 40/120 | 0/120 | 730/3860 |
| dfs_lora | 133/240 (55.4%) | 67/120 | 66/120 | 2046/3860 |
| fulltrace_sft | 29/240 (12.1%) | 29/120 | 0/120 | 668/3860 |
| fulltrace_sft_matched | 25/240 (10.4%) | 25/120 | 0/120 | 654/3860 |
| mixed_sft | 156/240 (65.0%) | 91/120 | 65/120 | 2606/3860 |

### Per family

| Family | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| terminal_length | fulltrace_sft | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | fulltrace_sft_matched | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | mixed_sft | 36/40 (90.0%) | 20/20 | 16/20 |
| chain_depth | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | mixed_sft | 16/40 (40.0%) | 16/20 | 0/20 |
| alternatives | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | fulltrace_sft_matched | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | mixed_sft | 37/40 (92.5%) | 20/20 | 17/20 |
| sequence | zeroshot | 12/40 (30.0%) | 12/20 | 0/20 |
| sequence | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| sequence | fulltrace_sft | 1/40 (2.5%) | 1/20 | 0/20 |
| sequence | fulltrace_sft_matched | 1/40 (2.5%) | 1/20 | 0/20 |
| sequence | mixed_sft | 3/40 (7.5%) | 3/20 | 0/20 |
| recursion | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | dfs_lora | 29/40 (72.5%) | 15/20 | 14/20 |
| recursion | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | mixed_sft | 32/40 (80.0%) | 16/20 | 16/20 |
| nesting | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | dfs_lora | 24/40 (60.0%) | 12/20 | 12/20 |
| nesting | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | mixed_sft | 32/40 (80.0%) | 16/20 | 16/20 |

### Per level

| Level | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| 1 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | dfs_lora | 32/48 (66.7%) | 16/24 | 16/24 |
| 1 | fulltrace_sft | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | fulltrace_sft_matched | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | mixed_sft | 39/48 (81.2%) | 23/24 | 16/24 |
| 2 | zeroshot | 12/48 (25.0%) | 12/24 | 0/24 |
| 2 | dfs_lora | 32/48 (66.7%) | 16/24 | 16/24 |
| 2 | fulltrace_sft | 8/48 (16.7%) | 8/24 | 0/24 |
| 2 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 2 | mixed_sft | 28/48 (58.3%) | 16/24 | 12/24 |
| 4 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 4 | dfs_lora | 32/48 (66.7%) | 16/24 | 16/24 |
| 4 | fulltrace_sft | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | mixed_sft | 36/48 (75.0%) | 20/24 | 16/24 |
| 8 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 8 | dfs_lora | 21/48 (43.8%) | 11/24 | 10/24 |
| 8 | fulltrace_sft | 5/48 (10.4%) | 5/24 | 0/24 |
| 8 | fulltrace_sft_matched | 5/48 (10.4%) | 5/24 | 0/24 |
| 8 | mixed_sft | 33/48 (68.8%) | 20/24 | 13/24 |
| 16 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 16 | dfs_lora | 16/48 (33.3%) | 8/24 | 8/24 |
| 16 | fulltrace_sft | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | mixed_sft | 20/48 (41.7%) | 12/24 | 8/24 |

## Original: Correct-state check

### Per arm and label

| Arm | All | Yes | No | Action accuracy |
| --- | --- | --- | --- | --- |
| zeroshot | 40/240 (16.7%) | 40/120 | 0/120 | 1417/3860 |
| dfs_lora | 133/240 (55.4%) | 67/120 | 66/120 | 3593/3860 |
| fulltrace_sft | 29/240 (12.1%) | 29/120 | 0/120 | 1365/3860 |
| fulltrace_sft_matched | 25/240 (10.4%) | 25/120 | 0/120 | 1432/3860 |
| mixed_sft | 156/240 (65.0%) | 91/120 | 65/120 | 3657/3860 |

### Per family

| Family | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| terminal_length | fulltrace_sft | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | fulltrace_sft_matched | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | mixed_sft | 36/40 (90.0%) | 20/20 | 16/20 |
| chain_depth | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | mixed_sft | 16/40 (40.0%) | 16/20 | 0/20 |
| alternatives | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | fulltrace_sft_matched | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | mixed_sft | 37/40 (92.5%) | 20/20 | 17/20 |
| sequence | zeroshot | 12/40 (30.0%) | 12/20 | 0/20 |
| sequence | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| sequence | fulltrace_sft | 1/40 (2.5%) | 1/20 | 0/20 |
| sequence | fulltrace_sft_matched | 1/40 (2.5%) | 1/20 | 0/20 |
| sequence | mixed_sft | 3/40 (7.5%) | 3/20 | 0/20 |
| recursion | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | dfs_lora | 29/40 (72.5%) | 15/20 | 14/20 |
| recursion | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | mixed_sft | 32/40 (80.0%) | 16/20 | 16/20 |
| nesting | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | dfs_lora | 24/40 (60.0%) | 12/20 | 12/20 |
| nesting | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | mixed_sft | 32/40 (80.0%) | 16/20 | 16/20 |

### Per level

| Level | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| 1 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | dfs_lora | 32/48 (66.7%) | 16/24 | 16/24 |
| 1 | fulltrace_sft | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | fulltrace_sft_matched | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | mixed_sft | 39/48 (81.2%) | 23/24 | 16/24 |
| 2 | zeroshot | 12/48 (25.0%) | 12/24 | 0/24 |
| 2 | dfs_lora | 32/48 (66.7%) | 16/24 | 16/24 |
| 2 | fulltrace_sft | 8/48 (16.7%) | 8/24 | 0/24 |
| 2 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 2 | mixed_sft | 28/48 (58.3%) | 16/24 | 12/24 |
| 4 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 4 | dfs_lora | 32/48 (66.7%) | 16/24 | 16/24 |
| 4 | fulltrace_sft | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | mixed_sft | 36/48 (75.0%) | 20/24 | 16/24 |
| 8 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 8 | dfs_lora | 21/48 (43.8%) | 11/24 | 10/24 |
| 8 | fulltrace_sft | 5/48 (10.4%) | 5/24 | 0/24 |
| 8 | fulltrace_sft_matched | 5/48 (10.4%) | 5/24 | 0/24 |
| 8 | mixed_sft | 33/48 (68.8%) | 20/24 | 13/24 |
| 16 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 16 | dfs_lora | 16/48 (33.3%) | 8/24 | 8/24 |
| 16 | fulltrace_sft | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | mixed_sft | 20/48 (41.7%) | 12/24 | 8/24 |

## Cuefree: Direct answer

### Per arm and label

| Arm | All | Yes | No | Action accuracy |
| --- | --- | --- | --- | --- |
| zeroshot | 204/240 (85.0%) | 116/120 | 88/120 | n/a |
| dfs_lora | 210/240 (87.5%) | 118/120 | 92/120 | n/a |
| fulltrace_sft | 202/240 (84.2%) | 119/120 | 83/120 | n/a |
| fulltrace_sft_matched | 200/240 (83.3%) | 119/120 | 81/120 | n/a |
| mixed_sft | 194/240 (80.8%) | 118/120 | 76/120 | n/a |

### Per family

| Family | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 32/40 (80.0%) | 20/20 | 12/20 |
| terminal_length | dfs_lora | 39/40 (97.5%) | 20/20 | 19/20 |
| terminal_length | fulltrace_sft | 32/40 (80.0%) | 20/20 | 12/20 |
| terminal_length | fulltrace_sft_matched | 32/40 (80.0%) | 20/20 | 12/20 |
| terminal_length | mixed_sft | 32/40 (80.0%) | 20/20 | 12/20 |
| chain_depth | zeroshot | 40/40 (100.0%) | 20/20 | 20/20 |
| chain_depth | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| chain_depth | fulltrace_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| chain_depth | fulltrace_sft_matched | 40/40 (100.0%) | 20/20 | 20/20 |
| chain_depth | mixed_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | zeroshot | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | fulltrace_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | fulltrace_sft_matched | 40/40 (100.0%) | 20/20 | 20/20 |
| alternatives | mixed_sft | 40/40 (100.0%) | 20/20 | 20/20 |
| sequence | zeroshot | 32/40 (80.0%) | 20/20 | 12/20 |
| sequence | dfs_lora | 32/40 (80.0%) | 20/20 | 12/20 |
| sequence | fulltrace_sft | 32/40 (80.0%) | 20/20 | 12/20 |
| sequence | fulltrace_sft_matched | 32/40 (80.0%) | 20/20 | 12/20 |
| sequence | mixed_sft | 32/40 (80.0%) | 20/20 | 12/20 |
| recursion | zeroshot | 28/40 (70.0%) | 20/20 | 8/20 |
| recursion | dfs_lora | 28/40 (70.0%) | 20/20 | 8/20 |
| recursion | fulltrace_sft | 28/40 (70.0%) | 20/20 | 8/20 |
| recursion | fulltrace_sft_matched | 28/40 (70.0%) | 20/20 | 8/20 |
| recursion | mixed_sft | 28/40 (70.0%) | 20/20 | 8/20 |
| nesting | zeroshot | 32/40 (80.0%) | 16/20 | 16/20 |
| nesting | dfs_lora | 31/40 (77.5%) | 18/20 | 13/20 |
| nesting | fulltrace_sft | 30/40 (75.0%) | 19/20 | 11/20 |
| nesting | fulltrace_sft_matched | 28/40 (70.0%) | 19/20 | 9/20 |
| nesting | mixed_sft | 22/40 (55.0%) | 18/20 | 4/20 |

### Per level

| Level | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| 1 | zeroshot | 47/48 (97.9%) | 23/24 | 24/24 |
| 1 | dfs_lora | 48/48 (100.0%) | 24/24 | 24/24 |
| 1 | fulltrace_sft | 48/48 (100.0%) | 24/24 | 24/24 |
| 1 | fulltrace_sft_matched | 48/48 (100.0%) | 24/24 | 24/24 |
| 1 | mixed_sft | 47/48 (97.9%) | 24/24 | 23/24 |
| 2 | zeroshot | 45/48 (93.8%) | 21/24 | 24/24 |
| 2 | dfs_lora | 45/48 (93.8%) | 22/24 | 23/24 |
| 2 | fulltrace_sft | 46/48 (95.8%) | 23/24 | 23/24 |
| 2 | fulltrace_sft_matched | 45/48 (93.8%) | 23/24 | 22/24 |
| 2 | mixed_sft | 43/48 (89.6%) | 22/24 | 21/24 |
| 4 | zeroshot | 44/48 (91.7%) | 24/24 | 20/24 |
| 4 | dfs_lora | 42/48 (87.5%) | 24/24 | 18/24 |
| 4 | fulltrace_sft | 42/48 (87.5%) | 24/24 | 18/24 |
| 4 | fulltrace_sft_matched | 42/48 (87.5%) | 24/24 | 18/24 |
| 4 | mixed_sft | 40/48 (83.3%) | 24/24 | 16/24 |
| 8 | zeroshot | 34/48 (70.8%) | 24/24 | 10/24 |
| 8 | dfs_lora | 38/48 (79.2%) | 24/24 | 14/24 |
| 8 | fulltrace_sft | 33/48 (68.8%) | 24/24 | 9/24 |
| 8 | fulltrace_sft_matched | 33/48 (68.8%) | 24/24 | 9/24 |
| 8 | mixed_sft | 32/48 (66.7%) | 24/24 | 8/24 |
| 16 | zeroshot | 34/48 (70.8%) | 24/24 | 10/24 |
| 16 | dfs_lora | 37/48 (77.1%) | 24/24 | 13/24 |
| 16 | fulltrace_sft | 33/48 (68.8%) | 24/24 | 9/24 |
| 16 | fulltrace_sft_matched | 32/48 (66.7%) | 24/24 | 8/24 |
| 16 | mixed_sft | 32/48 (66.7%) | 24/24 | 8/24 |

## Cuefree: Written run

### Per arm and label

| Arm | All | Yes | No | Action accuracy |
| --- | --- | --- | --- | --- |
| zeroshot | 30/240 (12.5%) | 28/120 | 2/120 | 513/3632 |
| dfs_lora | 0/240 (0.0%) | 0/120 | 0/120 | 204/3632 |
| fulltrace_sft | 77/240 (32.1%) | 58/120 | 19/120 | 980/3632 |
| fulltrace_sft_matched | 87/240 (36.2%) | 64/120 | 23/120 | 1129/3632 |
| mixed_sft | 89/240 (37.1%) | 64/120 | 25/120 | 1459/3632 |

### Per family

| Family | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| terminal_length | fulltrace_sft | 24/40 (60.0%) | 20/20 | 4/20 |
| terminal_length | fulltrace_sft_matched | 26/40 (65.0%) | 20/20 | 6/20 |
| terminal_length | mixed_sft | 19/40 (47.5%) | 18/20 | 1/20 |
| chain_depth | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | fulltrace_sft | 19/40 (47.5%) | 19/20 | 0/20 |
| chain_depth | fulltrace_sft_matched | 16/40 (40.0%) | 16/20 | 0/20 |
| chain_depth | mixed_sft | 18/40 (45.0%) | 18/20 | 0/20 |
| alternatives | zeroshot | 6/40 (15.0%) | 4/20 | 2/20 |
| alternatives | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| alternatives | fulltrace_sft | 30/40 (75.0%) | 15/20 | 15/20 |
| alternatives | fulltrace_sft_matched | 36/40 (90.0%) | 20/20 | 16/20 |
| alternatives | mixed_sft | 35/40 (87.5%) | 19/20 | 16/20 |
| sequence | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| sequence | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| sequence | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| sequence | fulltrace_sft_matched | 4/40 (10.0%) | 4/20 | 0/20 |
| sequence | mixed_sft | 5/40 (12.5%) | 5/20 | 0/20 |
| recursion | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | fulltrace_sft_matched | 5/40 (12.5%) | 4/20 | 1/20 |
| recursion | mixed_sft | 8/40 (20.0%) | 4/20 | 4/20 |
| nesting | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | mixed_sft | 4/40 (10.0%) | 0/20 | 4/20 |

### Per level

| Level | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| 1 | zeroshot | 12/48 (25.0%) | 12/24 | 0/24 |
| 1 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 1 | fulltrace_sft | 16/48 (33.3%) | 16/24 | 0/24 |
| 1 | fulltrace_sft_matched | 21/48 (43.8%) | 20/24 | 1/24 |
| 1 | mixed_sft | 26/48 (54.2%) | 18/24 | 8/24 |
| 2 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 2 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 2 | fulltrace_sft | 19/48 (39.6%) | 12/24 | 7/24 |
| 2 | fulltrace_sft_matched | 20/48 (41.7%) | 12/24 | 8/24 |
| 2 | mixed_sft | 14/48 (29.2%) | 10/24 | 4/24 |
| 4 | zeroshot | 5/48 (10.4%) | 4/24 | 1/24 |
| 4 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 4 | fulltrace_sft | 15/48 (31.2%) | 11/24 | 4/24 |
| 4 | fulltrace_sft_matched | 18/48 (37.5%) | 12/24 | 6/24 |
| 4 | mixed_sft | 17/48 (35.4%) | 13/24 | 4/24 |
| 8 | zeroshot | 5/48 (10.4%) | 4/24 | 1/24 |
| 8 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 8 | fulltrace_sft | 14/48 (29.2%) | 10/24 | 4/24 |
| 8 | fulltrace_sft_matched | 16/48 (33.3%) | 12/24 | 4/24 |
| 8 | mixed_sft | 16/48 (33.3%) | 11/24 | 5/24 |
| 16 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | dfs_lora | 0/48 (0.0%) | 0/24 | 0/24 |
| 16 | fulltrace_sft | 13/48 (27.1%) | 9/24 | 4/24 |
| 16 | fulltrace_sft_matched | 12/48 (25.0%) | 8/24 | 4/24 |
| 16 | mixed_sft | 16/48 (33.3%) | 12/24 | 4/24 |

## Cuefree: Gym memory

### Per arm and label

| Arm | All | Yes | No | Action accuracy |
| --- | --- | --- | --- | --- |
| zeroshot | 40/240 (16.7%) | 40/120 | 0/120 | 781/3632 |
| dfs_lora | 118/240 (49.2%) | 67/120 | 51/120 | 1616/3632 |
| fulltrace_sft | 31/240 (12.9%) | 31/120 | 0/120 | 685/3632 |
| fulltrace_sft_matched | 25/240 (10.4%) | 25/120 | 0/120 | 673/3632 |
| mixed_sft | 128/240 (53.3%) | 91/120 | 37/120 | 1896/3632 |

### Per family

| Family | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| terminal_length | fulltrace_sft | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | fulltrace_sft_matched | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | mixed_sft | 33/40 (82.5%) | 20/20 | 13/20 |
| chain_depth | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | mixed_sft | 19/40 (47.5%) | 16/20 | 3/20 |
| alternatives | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | dfs_lora | 39/40 (97.5%) | 20/20 | 19/20 |
| alternatives | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | fulltrace_sft_matched | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | mixed_sft | 33/40 (82.5%) | 20/20 | 13/20 |
| sequence | zeroshot | 12/40 (30.0%) | 12/20 | 0/20 |
| sequence | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| sequence | fulltrace_sft | 3/40 (7.5%) | 3/20 | 0/20 |
| sequence | fulltrace_sft_matched | 1/40 (2.5%) | 1/20 | 0/20 |
| sequence | mixed_sft | 3/40 (7.5%) | 3/20 | 0/20 |
| recursion | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | dfs_lora | 23/40 (57.5%) | 15/20 | 8/20 |
| recursion | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | mixed_sft | 20/40 (50.0%) | 16/20 | 4/20 |
| nesting | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | dfs_lora | 16/40 (40.0%) | 12/20 | 4/20 |
| nesting | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | mixed_sft | 20/40 (50.0%) | 16/20 | 4/20 |

### Per level

| Level | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| 1 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | dfs_lora | 32/48 (66.7%) | 16/24 | 16/24 |
| 1 | fulltrace_sft | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | fulltrace_sft_matched | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | mixed_sft | 34/48 (70.8%) | 23/24 | 11/24 |
| 2 | zeroshot | 12/48 (25.0%) | 12/24 | 0/24 |
| 2 | dfs_lora | 24/48 (50.0%) | 16/24 | 8/24 |
| 2 | fulltrace_sft | 8/48 (16.7%) | 8/24 | 0/24 |
| 2 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 2 | mixed_sft | 21/48 (43.8%) | 16/24 | 5/24 |
| 4 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 4 | dfs_lora | 28/48 (58.3%) | 16/24 | 12/24 |
| 4 | fulltrace_sft | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | mixed_sft | 28/48 (58.3%) | 20/24 | 8/24 |
| 8 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 8 | dfs_lora | 19/48 (39.6%) | 11/24 | 8/24 |
| 8 | fulltrace_sft | 7/48 (14.6%) | 7/24 | 0/24 |
| 8 | fulltrace_sft_matched | 5/48 (10.4%) | 5/24 | 0/24 |
| 8 | mixed_sft | 25/48 (52.1%) | 20/24 | 5/24 |
| 16 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 16 | dfs_lora | 15/48 (31.2%) | 8/24 | 7/24 |
| 16 | fulltrace_sft | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | mixed_sft | 20/48 (41.7%) | 12/24 | 8/24 |

## Cuefree: Correct-state check

### Per arm and label

| Arm | All | Yes | No | Action accuracy |
| --- | --- | --- | --- | --- |
| zeroshot | 40/240 (16.7%) | 40/120 | 0/120 | 1382/3632 |
| dfs_lora | 118/240 (49.2%) | 67/120 | 51/120 | 3195/3632 |
| fulltrace_sft | 32/240 (13.3%) | 32/120 | 0/120 | 1278/3632 |
| fulltrace_sft_matched | 25/240 (10.4%) | 25/120 | 0/120 | 1364/3632 |
| mixed_sft | 127/240 (52.9%) | 91/120 | 36/120 | 3260/3632 |

### Per family

| Family | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | dfs_lora | 40/40 (100.0%) | 20/20 | 20/20 |
| terminal_length | fulltrace_sft | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | fulltrace_sft_matched | 20/40 (50.0%) | 20/20 | 0/20 |
| terminal_length | mixed_sft | 33/40 (82.5%) | 20/20 | 13/20 |
| chain_depth | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| chain_depth | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| chain_depth | mixed_sft | 19/40 (47.5%) | 16/20 | 3/20 |
| alternatives | zeroshot | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | dfs_lora | 39/40 (97.5%) | 20/20 | 19/20 |
| alternatives | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | fulltrace_sft_matched | 4/40 (10.0%) | 4/20 | 0/20 |
| alternatives | mixed_sft | 32/40 (80.0%) | 20/20 | 12/20 |
| sequence | zeroshot | 12/40 (30.0%) | 12/20 | 0/20 |
| sequence | dfs_lora | 0/40 (0.0%) | 0/20 | 0/20 |
| sequence | fulltrace_sft | 4/40 (10.0%) | 4/20 | 0/20 |
| sequence | fulltrace_sft_matched | 1/40 (2.5%) | 1/20 | 0/20 |
| sequence | mixed_sft | 3/40 (7.5%) | 3/20 | 0/20 |
| recursion | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | dfs_lora | 23/40 (57.5%) | 15/20 | 8/20 |
| recursion | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| recursion | mixed_sft | 20/40 (50.0%) | 16/20 | 4/20 |
| nesting | zeroshot | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | dfs_lora | 16/40 (40.0%) | 12/20 | 4/20 |
| nesting | fulltrace_sft | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | fulltrace_sft_matched | 0/40 (0.0%) | 0/20 | 0/20 |
| nesting | mixed_sft | 20/40 (50.0%) | 16/20 | 4/20 |

### Per level

| Level | Arm | All | Yes | No |
| --- | --- | --- | --- | --- |
| 1 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | dfs_lora | 32/48 (66.7%) | 16/24 | 16/24 |
| 1 | fulltrace_sft | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | fulltrace_sft_matched | 8/48 (16.7%) | 8/24 | 0/24 |
| 1 | mixed_sft | 34/48 (70.8%) | 23/24 | 11/24 |
| 2 | zeroshot | 12/48 (25.0%) | 12/24 | 0/24 |
| 2 | dfs_lora | 24/48 (50.0%) | 16/24 | 8/24 |
| 2 | fulltrace_sft | 8/48 (16.7%) | 8/24 | 0/24 |
| 2 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 2 | mixed_sft | 21/48 (43.8%) | 16/24 | 5/24 |
| 4 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 4 | dfs_lora | 28/48 (58.3%) | 16/24 | 12/24 |
| 4 | fulltrace_sft | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 4 | mixed_sft | 28/48 (58.3%) | 20/24 | 8/24 |
| 8 | zeroshot | 4/48 (8.3%) | 4/24 | 0/24 |
| 8 | dfs_lora | 19/48 (39.6%) | 11/24 | 8/24 |
| 8 | fulltrace_sft | 8/48 (16.7%) | 8/24 | 0/24 |
| 8 | fulltrace_sft_matched | 5/48 (10.4%) | 5/24 | 0/24 |
| 8 | mixed_sft | 24/48 (50.0%) | 20/24 | 4/24 |
| 16 | zeroshot | 8/48 (16.7%) | 8/24 | 0/24 |
| 16 | dfs_lora | 15/48 (31.2%) | 8/24 | 7/24 |
| 16 | fulltrace_sft | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | fulltrace_sft_matched | 4/48 (8.3%) | 4/24 | 0/24 |
| 16 | mixed_sft | 20/48 (41.7%) | 12/24 | 8/24 |

## Failure accounting

| Suite | Arm | Lane | Failure | Cases |
| --- | --- | --- | --- | --- |
| original | zeroshot | direct | incorrect_answer | 4 |
| original | zeroshot | full_trace | invalid_or_trailing_action | 184 |
| original | zeroshot | full_trace | call_or_parse_error | 26 |
| original | zeroshot | full_trace | incomplete_trace | 2 |
| original | zeroshot | external_state | invalid_action | 200 |
| original | zeroshot | oracle_step | invalid_action | 200 |
| original | dfs_lora | direct | incorrect_answer | 3 |
| original | dfs_lora | full_trace | incomplete_trace | 142 |
| original | dfs_lora | full_trace | invalid_or_trailing_action | 51 |
| original | dfs_lora | full_trace | call_or_parse_error | 47 |
| original | dfs_lora | external_state | invalid_action | 107 |
| original | dfs_lora | oracle_step | invalid_action | 107 |
| original | fulltrace_sft | direct | incorrect_answer | 1 |
| original | fulltrace_sft | full_trace | invalid_or_trailing_action | 78 |
| original | fulltrace_sft | full_trace | call_or_parse_error | 89 |
| original | fulltrace_sft | full_trace | incomplete_trace | 7 |
| original | fulltrace_sft | external_state | invalid_action | 211 |
| original | fulltrace_sft | oracle_step | invalid_action | 211 |
| original | fulltrace_sft_matched | direct | incorrect_answer | 1 |
| original | fulltrace_sft_matched | full_trace | invalid_or_trailing_action | 44 |
| original | fulltrace_sft_matched | full_trace | incomplete_trace | 12 |
| original | fulltrace_sft_matched | full_trace | call_or_parse_error | 97 |
| original | fulltrace_sft_matched | external_state | invalid_action | 215 |
| original | fulltrace_sft_matched | oracle_step | invalid_action | 215 |
| original | mixed_sft | direct | incorrect_answer | 7 |
| original | mixed_sft | full_trace | incomplete_trace | 49 |
| original | mixed_sft | full_trace | invalid_or_trailing_action | 36 |
| original | mixed_sft | full_trace | call_or_parse_error | 55 |
| original | mixed_sft | external_state | invalid_action | 84 |
| original | mixed_sft | oracle_step | invalid_action | 84 |
| cuefree | zeroshot | direct | incorrect_answer | 36 |
| cuefree | zeroshot | full_trace | invalid_or_trailing_action | 172 |
| cuefree | zeroshot | full_trace | call_or_parse_error | 38 |
| cuefree | zeroshot | external_state | invalid_action | 200 |
| cuefree | zeroshot | oracle_step | invalid_action | 200 |
| cuefree | dfs_lora | direct | incorrect_answer | 30 |
| cuefree | dfs_lora | full_trace | incomplete_trace | 142 |
| cuefree | dfs_lora | full_trace | invalid_or_trailing_action | 46 |
| cuefree | dfs_lora | full_trace | call_or_parse_error | 52 |
| cuefree | dfs_lora | external_state | invalid_action | 122 |
| cuefree | dfs_lora | oracle_step | invalid_action | 122 |
| cuefree | fulltrace_sft | direct | incorrect_answer | 38 |
| cuefree | fulltrace_sft | full_trace | invalid_or_trailing_action | 65 |
| cuefree | fulltrace_sft | full_trace | call_or_parse_error | 98 |
| cuefree | fulltrace_sft | external_state | invalid_action | 209 |
| cuefree | fulltrace_sft | oracle_step | invalid_action | 208 |
| cuefree | fulltrace_sft_matched | direct | incorrect_answer | 40 |
| cuefree | fulltrace_sft_matched | full_trace | incomplete_trace | 8 |
| cuefree | fulltrace_sft_matched | full_trace | invalid_or_trailing_action | 35 |
| cuefree | fulltrace_sft_matched | full_trace | call_or_parse_error | 110 |
| cuefree | fulltrace_sft_matched | external_state | invalid_action | 215 |
| cuefree | fulltrace_sft_matched | oracle_step | invalid_action | 215 |
| cuefree | mixed_sft | direct | incorrect_answer | 46 |
| cuefree | mixed_sft | full_trace | incomplete_trace | 42 |
| cuefree | mixed_sft | full_trace | invalid_or_trailing_action | 49 |
| cuefree | mixed_sft | full_trace | call_or_parse_error | 60 |
| cuefree | mixed_sft | external_state | invalid_action | 112 |
| cuefree | mixed_sft | oracle_step | invalid_action | 113 |

## Written-run failure accounting

| Suite | Arm | One move then stop | Cap-length loop | Malformed/invalid/trailing | Other incomplete | Failed /240 |
| --- | --- | --- | --- | --- | --- | --- |
| original | zeroshot | 0 | 26 | 184 | 2 | 212 |
| original | dfs_lora | 133 | 47 | 51 | 9 | 240 |
| original | fulltrace_sft | 0 | 89 | 78 | 7 | 174 |
| original | fulltrace_sft_matched | 0 | 97 | 44 | 12 | 153 |
| original | mixed_sft | 0 | 49 | 42 | 49 | 140 |
| cuefree | zeroshot | 0 | 38 | 172 | 0 | 210 |
| cuefree | dfs_lora | 139 | 52 | 46 | 3 | 240 |
| cuefree | fulltrace_sft | 0 | 98 | 65 | 0 | 163 |
| cuefree | fulltrace_sft_matched | 0 | 110 | 35 | 8 | 153 |
| cuefree | mixed_sft | 0 | 50 | 59 | 42 | 151 |

Categories are mutually exclusive, with cap-length repeated-action loops classified first. X7 reproduces 133 one-move stops, 47 capped loops, and 60 remaining failures (51 malformed/invalid/trailing and 9 other incomplete). Some capped loops include TRY/PRN as well as DONE/BT. The repeated-block heuristic is descriptive: legitimate repetitions before another error can also trigger it. A readable prefix of malformed output is diagnostic only and never changes the grade.

## First wrong moves versus gym memory

Indices are zero-based; fractions divide by canonical action count. A correct prefix that simply stops has a missing move, not an observed wrong action. Paired medians include only the same failed written cases with an observable wrong move in both lanes. The full per-case paired audit retains missing moves and cases with no observable action error. A fully correct canonical prefix followed by trailing actions or bad JSON closure remains a failure, but is identified separately as a completion-format failure; a trailing first error has fraction 1 and is not evidence that state was lost before completion.

| Suite | Arm | Wrong observed | Missing only | Unobservable | Completed-prefix format failures | Paired errors | Written index | Gym index | Written fraction | Gym fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | zeroshot | 210 | 2 | 0 | 7 | 194 | 1.000 | 1.000 | 0.100 | 0.143 |
| original | dfs_lora | 98 | 142 | 0 | 1 | 39 | 1.000 | 9.000 | 0.050 | 0.165 |
| original | fulltrace_sft | 167 | 7 | 0 | 24 | 166 | 3.000 | 1.000 | 0.286 | 0.158 |
| original | fulltrace_sft_matched | 141 | 12 | 0 | 15 | 140 | 6.000 | 2.000 | 0.333 | 0.163 |
| original | mixed_sft | 91 | 49 | 0 | 16 | 46 | 9.500 | 1.000 | 0.571 | 0.143 |
| cuefree | zeroshot | 210 | 0 | 0 | 7 | 194 | 1.000 | 1.000 | 0.100 | 0.167 |
| cuefree | dfs_lora | 98 | 142 | 0 | 0 | 57 | 1.000 | 6.000 | 0.065 | 0.163 |
| cuefree | fulltrace_sft | 163 | 0 | 0 | 7 | 160 | 2.000 | 1.500 | 0.225 | 0.164 |
| cuefree | fulltrace_sft_matched | 145 | 8 | 0 | 10 | 144 | 2.000 | 2.000 | 0.250 | 0.164 |
| cuefree | mixed_sft | 109 | 42 | 0 | 1 | 75 | 4.000 | 1.000 | 0.163 | 0.154 |

### Original: failed written runs by family

| Family | Arm | Failures | Wrong observed | Missing only | Median wrong index | Median fraction | Index/length rank correlation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 20 | 18 | 2 | 1.000 | 0.250 | not observed |
| terminal_length | dfs_lora | 40 | 0 | 40 | not observed | not observed | not observed |
| terminal_length | fulltrace_sft | 16 | 10 | 6 | 1.000 | 0.250 | not observed |
| terminal_length | fulltrace_sft_matched | 12 | 8 | 4 | 1.500 | 0.375 | not observed |
| terminal_length | mixed_sft | 22 | 2 | 20 | 1.000 | 0.500 | not observed |
| chain_depth | zeroshot | 36 | 36 | 0 | 1.000 | 0.100 | not observed |
| chain_depth | dfs_lora | 40 | 0 | 40 | not observed | not observed | not observed |
| chain_depth | fulltrace_sft | 21 | 20 | 1 | 5.000 | 0.312 | 0.982 |
| chain_depth | fulltrace_sft_matched | 24 | 19 | 5 | 9.000 | 0.321 | 0.923 |
| chain_depth | mixed_sft | 22 | 2 | 20 | 2.000 | 0.667 | not observed |
| alternatives | zeroshot | 36 | 36 | 0 | 1.000 | 0.058 | -0.404 |
| alternatives | dfs_lora | 40 | 8 | 32 | 2.000 | 0.343 | not observed |
| alternatives | fulltrace_sft | 21 | 21 | 0 | 1.000 | 0.250 | -0.272 |
| alternatives | fulltrace_sft_matched | 6 | 5 | 1 | 1.000 | 0.250 | 1.000 |
| alternatives | mixed_sft | 5 | 1 | 4 | 2.000 | 0.087 | not observed |
| sequence | zeroshot | 40 | 40 | 0 | 1.000 | 0.143 | 0.152 |
| sequence | dfs_lora | 40 | 12 | 28 | 2.000 | 0.267 | -0.045 |
| sequence | fulltrace_sft | 36 | 36 | 0 | 2.000 | 0.286 | 0.000 |
| sequence | fulltrace_sft_matched | 36 | 34 | 2 | 2.000 | 0.333 | 0.000 |
| sequence | mixed_sft | 34 | 30 | 4 | 4.000 | 0.571 | 0.949 |
| recursion | zeroshot | 40 | 40 | 0 | 1.000 | 0.080 | -0.378 |
| recursion | dfs_lora | 40 | 38 | 2 | 1.000 | 0.077 | -0.553 |
| recursion | fulltrace_sft | 40 | 40 | 0 | 8.000 | 0.500 | 0.642 |
| recursion | fulltrace_sft_matched | 35 | 35 | 0 | 8.000 | 0.667 | 0.768 |
| recursion | mixed_sft | 25 | 24 | 1 | 11.000 | 0.854 | 0.789 |
| nesting | zeroshot | 40 | 40 | 0 | 1.000 | 0.065 | not observed |
| nesting | dfs_lora | 40 | 40 | 0 | 1.000 | 0.065 | -0.140 |
| nesting | fulltrace_sft | 40 | 40 | 0 | 5.000 | 0.166 | 0.477 |
| nesting | fulltrace_sft_matched | 40 | 40 | 0 | 8.000 | 0.429 | 0.736 |
| nesting | mixed_sft | 32 | 32 | 0 | 9.000 | 0.762 | 0.729 |

### Original: failed written runs by canonical length

| Length | Arm | Written complete | Gym complete | Failures | Wrong observed | Missing only | Median wrong index | Median fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1-8 | zeroshot | 28/128 | 36/128 | 100 | 98 | 2 | 1.000 | 0.200 |
| 1-8 | dfs_lora | 0/128 | 80/128 | 128 | 36 | 92 | 2.000 | 0.286 |
| 1-8 | fulltrace_sft | 48/128 | 28/128 | 80 | 74 | 6 | 2.000 | 0.286 |
| 1-8 | fulltrace_sft_matched | 59/128 | 24/128 | 69 | 60 | 9 | 2.000 | 0.429 |
| 1-8 | mixed_sft | 54/128 | 87/128 | 74 | 42 | 32 | 4.000 | 0.571 |
| 9-16 | zeroshot | 0/44 | 0/44 | 44 | 44 | 0 | 1.000 | 0.083 |
| 9-16 | dfs_lora | 0/44 | 23/44 | 44 | 24 | 20 | 1.000 | 0.080 |
| 9-16 | fulltrace_sft | 7/44 | 1/44 | 37 | 36 | 1 | 5.000 | 0.312 |
| 9-16 | fulltrace_sft_matched | 12/44 | 1/44 | 32 | 31 | 1 | 4.000 | 0.300 |
| 9-16 | mixed_sft | 20/44 | 32/44 | 24 | 16 | 8 | 9.000 | 0.894 |
| 17-32 | zeroshot | 0/40 | 4/40 | 40 | 40 | 0 | 1.000 | 0.044 |
| 17-32 | dfs_lora | 0/40 | 20/40 | 40 | 22 | 18 | 1.000 | 0.048 |
| 17-32 | fulltrace_sft | 5/40 | 0/40 | 35 | 35 | 0 | 5.000 | 0.161 |
| 17-32 | fulltrace_sft_matched | 8/40 | 0/40 | 32 | 31 | 1 | 15.000 | 0.750 |
| 17-32 | mixed_sft | 18/40 | 21/40 | 22 | 18 | 4 | 16.000 | 0.762 |
| 33-64 | zeroshot | 0/20 | 0/20 | 20 | 20 | 0 | 1.000 | 0.019 |
| 33-64 | dfs_lora | 0/20 | 10/20 | 20 | 8 | 12 | 1.000 | 0.019 |
| 33-64 | fulltrace_sft | 6/20 | 0/20 | 14 | 14 | 0 | 8.000 | 0.163 |
| 33-64 | fulltrace_sft_matched | 8/20 | 0/20 | 12 | 11 | 1 | 9.000 | 0.164 |
| 33-64 | mixed_sft | 8/20 | 16/20 | 12 | 7 | 5 | 55.000 | 1.000 |
| 65+ | zeroshot | 0/8 | 0/8 | 8 | 8 | 0 | 1.000 | 0.010 |
| 65+ | dfs_lora | 0/8 | 0/8 | 8 | 8 | 0 | 1.000 | 0.010 |
| 65+ | fulltrace_sft | 0/8 | 0/8 | 8 | 8 | 0 | 16.000 | 0.165 |
| 65+ | fulltrace_sft_matched | 0/8 | 0/8 | 8 | 8 | 0 | 16.000 | 0.165 |
| 65+ | mixed_sft | 0/8 | 0/8 | 8 | 8 | 0 | 16.000 | 0.155 |

### Cuefree: failed written runs by family

| Family | Arm | Failures | Wrong observed | Missing only | Median wrong index | Median fraction | Index/length rank correlation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| terminal_length | zeroshot | 20 | 20 | 0 | 1.000 | 0.250 | not observed |
| terminal_length | dfs_lora | 40 | 0 | 40 | not observed | not observed | not observed |
| terminal_length | fulltrace_sft | 16 | 16 | 0 | 1.000 | 0.250 | not observed |
| terminal_length | fulltrace_sft_matched | 14 | 10 | 4 | 1.000 | 0.250 | not observed |
| terminal_length | mixed_sft | 21 | 3 | 18 | 1.000 | 0.500 | not observed |
| chain_depth | zeroshot | 36 | 36 | 0 | 1.000 | 0.100 | -0.275 |
| chain_depth | dfs_lora | 40 | 0 | 40 | not observed | not observed | not observed |
| chain_depth | fulltrace_sft | 21 | 21 | 0 | 5.000 | 0.312 | 0.930 |
| chain_depth | fulltrace_sft_matched | 24 | 24 | 0 | 16.000 | 0.608 | 0.928 |
| chain_depth | mixed_sft | 22 | 2 | 20 | 2.000 | 0.667 | not observed |
| alternatives | zeroshot | 34 | 34 | 0 | 1.000 | 0.117 | -0.041 |
| alternatives | dfs_lora | 40 | 4 | 36 | 2.000 | 0.400 | not observed |
| alternatives | fulltrace_sft | 10 | 10 | 0 | 1.000 | 0.170 | -0.274 |
| alternatives | fulltrace_sft_matched | 4 | 0 | 4 | not observed | not observed | not observed |
| alternatives | mixed_sft | 5 | 1 | 4 | 2.000 | 0.087 | not observed |
| sequence | zeroshot | 40 | 40 | 0 | 2.000 | 0.154 | 0.489 |
| sequence | dfs_lora | 40 | 16 | 24 | 2.000 | 0.200 | not observed |
| sequence | fulltrace_sft | 36 | 36 | 0 | 2.000 | 0.154 | not observed |
| sequence | fulltrace_sft_matched | 36 | 36 | 0 | 2.000 | 0.154 | not observed |
| sequence | mixed_sft | 35 | 35 | 0 | 2.000 | 0.290 | 0.302 |
| recursion | zeroshot | 40 | 40 | 0 | 1.000 | 0.104 | -0.336 |
| recursion | dfs_lora | 40 | 38 | 2 | 1.000 | 0.104 | -0.501 |
| recursion | fulltrace_sft | 40 | 40 | 0 | 3.000 | 0.165 | 0.733 |
| recursion | fulltrace_sft_matched | 35 | 35 | 0 | 5.000 | 0.333 | 0.526 |
| recursion | mixed_sft | 32 | 32 | 0 | 6.000 | 0.457 | 0.333 |
| nesting | zeroshot | 40 | 40 | 0 | 1.000 | 0.077 | not observed |
| nesting | dfs_lora | 40 | 40 | 0 | 1.000 | 0.077 | -0.141 |
| nesting | fulltrace_sft | 40 | 40 | 0 | 4.500 | 0.166 | 0.622 |
| nesting | fulltrace_sft_matched | 40 | 40 | 0 | 5.000 | 0.166 | 0.552 |
| nesting | mixed_sft | 36 | 36 | 0 | 6.000 | 0.389 | 0.372 |

### Cuefree: failed written runs by canonical length

| Length | Arm | Written complete | Gym complete | Failures | Wrong observed | Missing only | Median wrong index | Median fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1-8 | zeroshot | 28/112 | 36/112 | 84 | 84 | 0 | 1.000 | 0.250 |
| 1-8 | dfs_lora | 0/112 | 80/112 | 112 | 35 | 77 | 1.000 | 0.167 |
| 1-8 | fulltrace_sft | 51/112 | 28/112 | 61 | 61 | 0 | 1.000 | 0.250 |
| 1-8 | fulltrace_sft_matched | 59/112 | 24/112 | 53 | 45 | 8 | 2.000 | 0.333 |
| 1-8 | mixed_sft | 54/112 | 83/112 | 58 | 32 | 26 | 2.000 | 0.583 |
| 9-16 | zeroshot | 1/56 | 0/56 | 55 | 55 | 0 | 1.000 | 0.083 |
| 9-16 | dfs_lora | 0/56 | 23/56 | 56 | 28 | 28 | 1.000 | 0.077 |
| 9-16 | fulltrace_sft | 11/56 | 3/56 | 45 | 45 | 0 | 2.000 | 0.200 |
| 9-16 | fulltrace_sft_matched | 12/56 | 1/56 | 44 | 44 | 0 | 2.000 | 0.200 |
| 9-16 | mixed_sft | 16/56 | 28/56 | 40 | 32 | 8 | 3.000 | 0.322 |
| 17-32 | zeroshot | 1/44 | 4/44 | 43 | 43 | 0 | 1.000 | 0.048 |
| 17-32 | dfs_lora | 0/44 | 8/44 | 44 | 23 | 21 | 1.000 | 0.048 |
| 17-32 | fulltrace_sft | 9/44 | 0/44 | 35 | 35 | 0 | 4.000 | 0.160 |
| 17-32 | fulltrace_sft_matched | 8/44 | 0/44 | 36 | 36 | 0 | 4.000 | 0.160 |
| 17-32 | mixed_sft | 11/44 | 9/44 | 33 | 29 | 4 | 9.000 | 0.290 |
| 33-64 | zeroshot | 0/24 | 0/24 | 24 | 24 | 0 | 1.000 | 0.024 |
| 33-64 | dfs_lora | 0/24 | 7/24 | 24 | 8 | 16 | 1.000 | 0.024 |
| 33-64 | fulltrace_sft | 6/24 | 0/24 | 18 | 18 | 0 | 6.000 | 0.162 |
| 33-64 | fulltrace_sft_matched | 8/24 | 0/24 | 16 | 16 | 0 | 7.000 | 0.163 |
| 33-64 | mixed_sft | 8/24 | 8/24 | 16 | 12 | 4 | 7.000 | 0.163 |
| 65+ | zeroshot | 0/4 | 0/4 | 4 | 4 | 0 | 1.000 | 0.014 |
| 65+ | dfs_lora | 0/4 | 0/4 | 4 | 4 | 0 | 1.000 | 0.014 |
| 65+ | fulltrace_sft | 0/4 | 0/4 | 4 | 4 | 0 | 12.000 | 0.164 |
| 65+ | fulltrace_sft_matched | 0/4 | 0/4 | 4 | 4 | 0 | 12.000 | 0.164 |
| 65+ | mixed_sft | 0/4 | 0/4 | 4 | 4 | 0 | 12.000 | 0.164 |

## Call budgets and costs

| Suite | Arm | Lane | Calls | Completion tokens | Total tokens | Inference seconds | Capped calls | Reused cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | zeroshot | direct | 240 | 1440 | 48528 | 1 | 0 | 240 |
| original | zeroshot | full_trace | 240 | 116452 | 237076 | 866 | 26 | 240 |
| original | zeroshot | external_state | 930 | 10019 | 633418 | 75 | 0 | 240 |
| original | zeroshot | oracle_step | 3860 | 40413 | 2867477 | 34 | 0 | 240 |
| original | dfs_lora | direct | 240 | 1440 | 48528 | 1 | 0 | 240 |
| original | dfs_lora | full_trace | 240 | 196743 | 317367 | 3312 | 47 | 240 |
| original | dfs_lora | external_state | 2153 | 20561 | 1460135 | 352 | 0 | 240 |
| original | dfs_lora | oracle_step | 3860 | 34829 | 2861893 | 66 | 0 | 240 |
| original | fulltrace_sft | direct | 240 | 1440 | 48528 | 53 | 0 | 0 |
| original | fulltrace_sft | full_trace | 240 | 374294 | 494918 | 8035 | 89 | 0 |
| original | fulltrace_sft | external_state | 879 | 9514 | 541942 | 188 | 0 | 0 |
| original | fulltrace_sft | oracle_step | 3860 | 41341 | 2868405 | 75 | 0 | 0 |
| original | fulltrace_sft_matched | direct | 240 | 1440 | 48528 | 56 | 0 | 0 |
| original | fulltrace_sft_matched | full_trace | 240 | 407766 | 528390 | 8572 | 97 | 0 |
| original | fulltrace_sft_matched | external_state | 869 | 9428 | 518655 | 211 | 0 | 0 |
| original | fulltrace_sft_matched | oracle_step | 3860 | 41409 | 2868473 | 86 | 0 | 0 |
| original | mixed_sft | direct | 240 | 1440 | 48528 | 43 | 0 | 0 |
| original | mixed_sft | full_trace | 240 | 242686 | 363310 | 4414 | 55 | 0 |
| original | mixed_sft | external_state | 2690 | 25403 | 1866101 | 483 | 0 | 0 |
| original | mixed_sft | oracle_step | 3860 | 35135 | 2862199 | 69 | 0 | 0 |
| cuefree | zeroshot | direct | 240 | 1440 | 48900 | 43 | 0 | 240 |
| cuefree | zeroshot | full_trace | 240 | 167077 | 288073 | 1251 | 38 | 240 |
| cuefree | zeroshot | external_state | 981 | 10474 | 688784 | 79 | 0 | 240 |
| cuefree | zeroshot | oracle_step | 3632 | 37959 | 2701463 | 32 | 0 | 120 |
| cuefree | dfs_lora | direct | 240 | 1440 | 48900 | 2 | 0 | 0 |
| cuefree | dfs_lora | full_trace | 240 | 216871 | 337867 | 3611 | 52 | 120 |
| cuefree | dfs_lora | external_state | 1738 | 16920 | 1198904 | 286 | 0 | 120 |
| cuefree | dfs_lora | oracle_step | 3632 | 33000 | 2696504 | 61 | 0 | 120 |
| cuefree | fulltrace_sft | direct | 240 | 1440 | 48900 | 2 | 0 | 0 |
| cuefree | fulltrace_sft | full_trace | 240 | 410667 | 531663 | 8503 | 98 | 0 |
| cuefree | fulltrace_sft | external_state | 894 | 9598 | 562433 | 165 | 0 | 0 |
| cuefree | fulltrace_sft | oracle_step | 3632 | 38711 | 2702215 | 62 | 0 | 0 |
| cuefree | fulltrace_sft_matched | direct | 240 | 1440 | 48900 | 1 | 0 | 0 |
| cuefree | fulltrace_sft_matched | full_trace | 240 | 459994 | 580990 | 8716 | 110 | 0 |
| cuefree | fulltrace_sft_matched | external_state | 888 | 9578 | 548782 | 168 | 0 | 0 |
| cuefree | fulltrace_sft_matched | oracle_step | 3632 | 38765 | 2702269 | 63 | 0 | 0 |
| cuefree | mixed_sft | direct | 240 | 1440 | 48900 | 2 | 0 | 0 |
| cuefree | mixed_sft | full_trace | 240 | 262286 | 383282 | 4677 | 60 | 0 |
| cuefree | mixed_sft | external_state | 2008 | 19671 | 1440181 | 380 | 0 | 0 |
| cuefree | mixed_sft | oracle_step | 3632 | 33191 | 2696695 | 64 | 0 | 0 |

| Arm | Arm-process wall seconds | New training wall seconds | Fresh completion tokens |
| --- | --- | --- | --- |
| zeroshot | 75 | 0 | 25896 |
| dfs_lora | 1658 | 0 | 115429 |
| fulltrace_sft | 17814 | 151 | 887005 |
| fulltrace_sft_matched | 18674 | 204 | 969820 |
| mixed_sft | 18095 | 7438 | 621252 |

Inference seconds sum the backend-reported amortized per-call times. Historical calls contribute to lane token/time totals but were not rerun; arm-process wall time includes model setup, HDFS persistence, and checkpoint parity checks, but starts after dependency/environment bootstrap. Full Ray submission wall times are retained in the execution manifest. Identical reused yes calls appear in both suite totals and are not independent replications. Raw provenance enables exact deduplication. No tool-failure recovery was tested.

## Reproducibility and interpretation limits

Base `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`; ordinary LoRA rank 16, alpha 32, dropout 0, q/k/v/o/gate/up/down targets; BF16 base and FP32 adapters. AdamW, learning rate 2e-4, weight decay .01, cosine schedule, 5% warmup, gradient clip 1; microbatch 8, accumulation 4, effective batch 32 with X7 partial final-batch behavior. Gradient checkpointing, response-only loss, no target truncation or packing. Initialization and native checkpoint reload parity checks passed.

Greedy temperature 0, top-p 1, top-k -1, seed 17, thinking off, context 16,384. Direct cap 128, written cap 4,096, action cap 128, maximum 512 action calls. Prompts and grading are unchanged. Direct/correct-state batches have at most 32 calls; written/gym calls are serial. Native adapters use X7's job-scoped vLLM 0.11.2 B200 PDL workaround. The historical zero-shot reference has its LoRA engine disabled; this runtime difference remains a comparison limit.

The cue-free suite preserves the 120 yes instances exactly and changes the negative grammars. Output and success changes on identical yes instances are listed per arm/lane in metadata.json. It removes the last-terminal/symbol-presence cue, not every possible shortcut; some count/length rules remain above chance. Families and alphabet permutations share templates; one seed, few full trajectories, training-token versus update differences, and fixed completion budgets limit claims. The optional 92-case bracket full-trace test was skipped because its action protocol differs and new training data are out of scope. Existing results were not modified.

[Frozen protocol](wholerun_protocol.md), [machine-readable protocol](../experiments/wholerun/protocol.json), [family/level/label atlas](results/wholerun/summary.json), [paired diagnostics](results/wholerun/diagnostics.json), [raw calls and instances](results/wholerun/audit.json.gz), [training/runtime metadata](results/wholerun/metadata.json), [execution manifest](results/wholerun/execution_manifest.json), and [cost accounting](results/wholerun/costs.json) accompany this report.
