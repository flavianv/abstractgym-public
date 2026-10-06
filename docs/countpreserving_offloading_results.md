# Count-preserving puzzles and operator offloading: zero-shot Qwen3-14B

All ordered parts are complete: construction and the shortcut gate, four zero-shot interfaces on A/B, then fresh H0-H5 in two interfaces on all four suites. The 480 new baseline cases per interface and 11,520 hint-study cases are retained, including every failed response. The 3,840-case baseline comparison and all hint responses were replayed through the grader. No training, thinking mode, examples, repair, or model-outcome-driven case selection. Only H2/H3 live execution permits one bookkeeping re-ask.

## Main comparison

| Interface | original | cuefree | A | B |
| --- | --- | --- | --- | --- |
| Direct answer | 236/240 (98.3%) | 204/240 (85.0%) | 187/240 (77.9%) | 146/240 (60.8%) |
| Full execution in one reply | 28/240 (11.7%) | 30/240 (12.5%) | 32/240 (13.3%) | 5/240 (2.1%) |
| Gym memory | 40/240 (16.7%) | 40/240 (16.7%) | 28/240 (11.7%) | 4/240 (1.7%) |
| Correct-state check (all actions correct) | 40/240 (16.7%) | 40/240 (16.7%) | 28/240 (11.7%) | 4/240 (1.7%) |

Original/cue-free baselines reuse the committed whole-run zero-shot audit; A/B are fresh. Direct-answer correctness is not execution correctness. Gym/full-trace success requires a complete valid canonical trajectory and the correct membership verdict. Correct-state all-case success requires every independently queried teacher-state action to be correct; it is not a deployed rollout.

## What this answers

A direct answer is correct on 187/240 (77.9%) and B on 146/240 (60.8%), versus chance 120/240. Every yes/no pair has identical count features, so a deterministic count-only rule cannot produce these above-chance results. This does not establish a general membership algorithm: A contains simple order-sensitive structures and B samples bounded linear CFGs. B also emits only 66 yes answers out of 240 (27.5%), with 46/120 yes and 100/120 no correct. A emits 129/240 yes (53.8%), but its nesting family emits no on every case. An overall score must therefore be read with family/size and reply-bias tables.

The following are matched, fresh H0 comparisons. Each value is all; yes; no, with failures retained. These are scaffolded zero-shot executions, not new model capabilities established without the scaffold.

| Suite | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| original | 41/240; 41/120; 0/120 | 62/240; 62/120; 0/120 | 39/240; 39/120; 0/120 | 62/240; 62/120; 0/120 | 64/240; 64/120; 0/120 | 64/240; 64/120; 0/120 |
| cuefree | 40/240; 40/120; 0/120 | 62/240; 62/120; 0/120 | 39/240; 39/120; 0/120 | 62/240; 62/120; 0/120 | 64/240; 64/120; 0/120 | 64/240; 64/120; 0/120 |
| A | 28/240; 28/120; 0/120 | 64/240; 64/120; 0/120 | 31/240; 28/120; 3/120 | 64/240; 64/120; 0/120 | 64/240; 64/120; 0/120 | 64/240; 64/120; 0/120 |
| B | 4/240; 4/120; 0/120 | 6/240; 6/120; 0/120 | 4/240; 4/120; 0/120 | 6/240; 6/120; 0/120 | 11/240; 11/120; 0/120 | 11/240; 11/120; 0/120 |

original: the best complete-run condition(s) are H4, H5, 64/240 versus fresh H0 41/240. DONE accuracy is 19/492 in H0, 53/492 in H1 and 53/492 in H3. Adding comparisons (H4 versus H1) changes complete runs by +2; adding next_untried (H5 versus H4) changes them by +0. Use the per-operator and wrong-substitution tables below to distinguish exhaustion assistance from comparison or production-selection assistance.

cuefree: the best complete-run condition(s) are H4, H5, 64/240 versus fresh H0 40/240. DONE accuracy is 48/516 in H0, 54/516 in H1 and 54/516 in H3. Adding comparisons (H4 versus H1) changes complete runs by +2; adding next_untried (H5 versus H4) changes them by +0. Use the per-operator and wrong-substitution tables below to distinguish exhaustion assistance from comparison or production-selection assistance.

A: the best complete-run condition(s) are H1, H3, H4, H5, 64/240 versus fresh H0 28/240. DONE accuracy is 49/512 in H0, 81/512 in H1 and 81/512 in H3. Adding comparisons (H4 versus H1) changes complete runs by +0; adding next_untried (H5 versus H4) changes them by +0. Use the per-operator and wrong-substitution tables below to distinguish exhaustion assistance from comparison or production-selection assistance.

B: the best complete-run condition(s) are H4, H5, 11/240 versus fresh H0 4/240. DONE accuracy is 0/618 in H0, 72/618 in H1 and 72/618 in H3. Adding comparisons (H4 versus H1) changes complete runs by +5; adding next_untried (H5 versus H4) changes them by +0. Use the per-operator and wrong-substitution tables below to distinguish exhaustion assistance from comparison or production-selection assistance.

The guard activates 19 times across 1,920 guarded live cases. Its second answer is correct 8/19 times. In fresh H0, 840/847 failed live cases stop before the first required DONE. These deployed first errors must be distinguished from frequent DONE mistakes on independently supplied teacher states. Listing remaining rules does not by itself establish an exhaustion solution; the DONE rows and complete negative-run counts are the relevant evidence.

The highest DONE accuracy in any suite/condition is 15.8%, not near ceiling. Across the hint study there are only 3 complete negative runs; the table above locates them by suite and condition. H5 adds no complete runs over H4 on any suite. The proposed exhaustion rescue is therefore not observed under this fixed zero-shot protocol. The strongest supported conclusion is narrower: remaining-rule hints improve some positive executions, while earlier action/argument mistakes still dominate deployed failures.

The separately trained X7 step adapter completes 133/240 original and 118/240 cue-free cases. Those scores provide scale only: X7 received training; H0-H5 did not. The new count-preserving study does not evaluate X7 on A/B. Strong H5 explicitly delegates first-untried production selection to code.

## Shortcut gate

Before any model call, all 129 checks per suite scored exactly 50% overall and in every nonempty family, size, and family/size cell. The rule battery includes symbol-presence checks, lengths, letter counts, rule/placeholder counts, parity, ratios, constants, and a bag-of-symbols logistic classifier. Classifier predictions are four-fold out-of-fold, holding entire pairs together and fitting the vectorizer only on training folds. Pairwise feature identity makes the chance result stronger than a finite battery: any deterministic function of these count features predicts both labels identically.

| Suite | Cell | Cases | Minimum over checks | Maximum over checks |
| --- | --- | --- | --- | --- |
| A | overall | 240 | 50.0% | 50.0% |
| B | overall | 240 | 50.0% | 50.0% |
| A | family: alternatives | 40 | 50.0% | 50.0% |
| A | family: chain_depth | 40 | 50.0% | 50.0% |
| A | family: nesting | 40 | 50.0% | 50.0% |
| A | family: recursion | 40 | 50.0% | 50.0% |
| A | family: sequence | 40 | 50.0% | 50.0% |
| A | family: terminal_length | 40 | 50.0% | 50.0% |
| B | family: random_multi_rule | 240 | 50.0% | 50.0% |
| A | size: 1 | 48 | 50.0% | 50.0% |
| A | size: 2 | 48 | 50.0% | 50.0% |
| A | size: 4 | 48 | 50.0% | 50.0% |
| A | size: 8 | 48 | 50.0% | 50.0% |
| A | size: 16 | 48 | 50.0% | 50.0% |
| B | size: 1 | 48 | 50.0% | 50.0% |
| B | size: 2 | 48 | 50.0% | 50.0% |
| B | size: 4 | 48 | 50.0% | 50.0% |
| B | size: 8 | 48 | 50.0% | 50.0% |
| B | size: 16 | 48 | 50.0% | 50.0% |

### Every shortcut (overall)

| Check | A | B |
| --- | --- | --- |
| always no | 120/240 | 120/240 |
| always yes | 120/240 | 120/240 |
| any placeholder count equals input | 120/240 | 120/240 |
| any rule length equals input | 120/240 | 120/240 |
| bag-of-symbols logistic regression (held-out pairs) | 120/240 | 120/240 |
| count equality:a | 120/240 | 120/240 |
| count equality:b | 120/240 | 120/240 |
| count equality:c | 120/240 | 120/240 |
| count equality:d | 120/240 | 120/240 |
| count equality:e | 120/240 | 120/240 |
| count parity:a | 120/240 | 120/240 |
| count parity:b | 120/240 | 120/240 |
| count parity:c | 120/240 | 120/240 |
| count parity:d | 120/240 | 120/240 |
| count parity:e | 120/240 | 120/240 |
| grammar terminals subset of string symbols | 120/240 | 120/240 |
| grammar/input ratio a/a | 120/240 | 120/240 |
| grammar/input ratio a/b | 120/240 | 120/240 |
| grammar/input ratio a/c | 120/240 | 120/240 |
| grammar/input ratio a/d | 120/240 | 120/240 |
| grammar/input ratio a/e | 120/240 | 120/240 |
| grammar/input ratio b/a | 120/240 | 120/240 |
| grammar/input ratio b/b | 120/240 | 120/240 |
| grammar/input ratio b/c | 120/240 | 120/240 |
| grammar/input ratio b/d | 120/240 | 120/240 |
| grammar/input ratio b/e | 120/240 | 120/240 |
| grammar/input ratio c/a | 120/240 | 120/240 |
| grammar/input ratio c/b | 120/240 | 120/240 |
| grammar/input ratio c/c | 120/240 | 120/240 |
| grammar/input ratio c/d | 120/240 | 120/240 |
| grammar/input ratio c/e | 120/240 | 120/240 |
| grammar/input ratio d/a | 120/240 | 120/240 |
| grammar/input ratio d/b | 120/240 | 120/240 |
| grammar/input ratio d/c | 120/240 | 120/240 |
| grammar/input ratio d/d | 120/240 | 120/240 |
| grammar/input ratio d/e | 120/240 | 120/240 |
| grammar/input ratio e/a | 120/240 | 120/240 |
| grammar/input ratio e/b | 120/240 | 120/240 |
| grammar/input ratio e/c | 120/240 | 120/240 |
| grammar/input ratio e/d | 120/240 | 120/240 |
| grammar/input ratio e/e | 120/240 | 120/240 |
| input ratio a/a=1 | 120/240 | 120/240 |
| input ratio a/a=2 | 120/240 | 120/240 |
| input ratio a/a=3 | 120/240 | 120/240 |
| input ratio a/b=1 | 120/240 | 120/240 |
| input ratio a/b=2 | 120/240 | 120/240 |
| input ratio a/b=3 | 120/240 | 120/240 |
| input ratio a/c=1 | 120/240 | 120/240 |
| input ratio a/c=2 | 120/240 | 120/240 |
| input ratio a/c=3 | 120/240 | 120/240 |
| input ratio a/d=1 | 120/240 | 120/240 |
| input ratio a/d=2 | 120/240 | 120/240 |
| input ratio a/d=3 | 120/240 | 120/240 |
| input ratio a/e=1 | 120/240 | 120/240 |
| input ratio a/e=2 | 120/240 | 120/240 |
| input ratio a/e=3 | 120/240 | 120/240 |
| input ratio b/a=1 | 120/240 | 120/240 |
| input ratio b/a=2 | 120/240 | 120/240 |
| input ratio b/a=3 | 120/240 | 120/240 |
| input ratio b/b=1 | 120/240 | 120/240 |
| input ratio b/b=2 | 120/240 | 120/240 |
| input ratio b/b=3 | 120/240 | 120/240 |
| input ratio b/c=1 | 120/240 | 120/240 |
| input ratio b/c=2 | 120/240 | 120/240 |
| input ratio b/c=3 | 120/240 | 120/240 |
| input ratio b/d=1 | 120/240 | 120/240 |
| input ratio b/d=2 | 120/240 | 120/240 |
| input ratio b/d=3 | 120/240 | 120/240 |
| input ratio b/e=1 | 120/240 | 120/240 |
| input ratio b/e=2 | 120/240 | 120/240 |
| input ratio b/e=3 | 120/240 | 120/240 |
| input ratio c/a=1 | 120/240 | 120/240 |
| input ratio c/a=2 | 120/240 | 120/240 |
| input ratio c/a=3 | 120/240 | 120/240 |
| input ratio c/b=1 | 120/240 | 120/240 |
| input ratio c/b=2 | 120/240 | 120/240 |
| input ratio c/b=3 | 120/240 | 120/240 |
| input ratio c/c=1 | 120/240 | 120/240 |
| input ratio c/c=2 | 120/240 | 120/240 |
| input ratio c/c=3 | 120/240 | 120/240 |
| input ratio c/d=1 | 120/240 | 120/240 |
| input ratio c/d=2 | 120/240 | 120/240 |
| input ratio c/d=3 | 120/240 | 120/240 |
| input ratio c/e=1 | 120/240 | 120/240 |
| input ratio c/e=2 | 120/240 | 120/240 |
| input ratio c/e=3 | 120/240 | 120/240 |
| input ratio d/a=1 | 120/240 | 120/240 |
| input ratio d/a=2 | 120/240 | 120/240 |
| input ratio d/a=3 | 120/240 | 120/240 |
| input ratio d/b=1 | 120/240 | 120/240 |
| input ratio d/b=2 | 120/240 | 120/240 |
| input ratio d/b=3 | 120/240 | 120/240 |
| input ratio d/c=1 | 120/240 | 120/240 |
| input ratio d/c=2 | 120/240 | 120/240 |
| input ratio d/c=3 | 120/240 | 120/240 |
| input ratio d/d=1 | 120/240 | 120/240 |
| input ratio d/d=2 | 120/240 | 120/240 |
| input ratio d/d=3 | 120/240 | 120/240 |
| input ratio d/e=1 | 120/240 | 120/240 |
| input ratio d/e=2 | 120/240 | 120/240 |
| input ratio d/e=3 | 120/240 | 120/240 |
| input ratio e/a=1 | 120/240 | 120/240 |
| input ratio e/a=2 | 120/240 | 120/240 |
| input ratio e/a=3 | 120/240 | 120/240 |
| input ratio e/b=1 | 120/240 | 120/240 |
| input ratio e/b=2 | 120/240 | 120/240 |
| input ratio e/b=3 | 120/240 | 120/240 |
| input ratio e/c=1 | 120/240 | 120/240 |
| input ratio e/c=2 | 120/240 | 120/240 |
| input ratio e/c=3 | 120/240 | 120/240 |
| input ratio e/d=1 | 120/240 | 120/240 |
| input ratio e/d=2 | 120/240 | 120/240 |
| input ratio e/d=3 | 120/240 | 120/240 |
| input ratio e/e=1 | 120/240 | 120/240 |
| input ratio e/e=2 | 120/240 | 120/240 |
| input ratio e/e=3 | 120/240 | 120/240 |
| last rule length equals input | 120/240 | 120/240 |
| last rule, last terminal in string | 120/240 | 120/240 |
| longest full rule no longer than input | 120/240 | 120/240 |
| longest rule length equals input | 120/240 | 120/240 |
| longest rule no longer than the string | 120/240 | 120/240 |
| rule/input length parity equal | 120/240 | 120/240 |
| same symbol set | 120/240 | 120/240 |
| some rule has as many terminals as the string | 120/240 | 120/240 |
| string symbols subset of grammar terminals | 120/240 | 120/240 |
| terminal multisets equal | 120/240 | 120/240 |
| terminal/input length parity equal | 120/240 | 120/240 |
| total placeholders equal input | 120/240 | 120/240 |
| total terminal count equals input | 120/240 | 120/240 |

### Construction and scope

A has six families, five sizes (1, 2, 4, 8, 16), four letterings and balanced labels. Order-sensitive blocks contain two symbols, so length/sequence level n has input length 2n; nesting uses n openings of each of two types, actual nesting 2n, even at level one. B has 120 independent random linear grammars, 3-6 nonterminals, 2-3 rules per nonterminal, 3-4 declared symbols, and successful canonical depths 1, 2, 4, 8, 16 with 24 pairs per depth. A B-negative swaps two different adjacent input symbols; only Earley-rejected, bounded-DFS-completing pairs survive. Both partners retain all bags/counts. No epsilon or left recursion; maximum 512 search steps. Grammar reuse, correlated letterings and rejection sampling limit generalization.

```json
{
  "attempts": 127,
  "generation_policy": "random terminal/terminal-prefix single-child rules; rejection sampled",
  "nonterminals": [
    3,
    6
  ],
  "rejected": {
    "bounded_solver_disagreement_or_limit": 1,
    "different_canonical_accepting_depth": 1,
    "no_rejecting_swap": 5
  },
  "retained_pairs": 120,
  "rules_per_nonterminal": [
    2,
    3
  ],
  "seed": 417,
  "vocabulary": [
    3,
    4
  ]
}
```

## Part 2: baseline atlas

Each entry is all correct/total; yes correct/total; no correct/total. A/B sizes are not paired structural equivalents of original/cue-free sizes; inspect actual string lengths and accepting depths in the frozen instances. Missing families are n/a, not zero-score cases.

### Direct answer

| Suite | All | Yes | No |
| --- | --- | --- | --- |
| original | 236/240 | 116/120 | 120/120 |
| cuefree | 204/240 | 116/120 | 88/120 |
| A | 187/240 | 98/120 | 89/120 |
| B | 146/240 | 46/120 | 100/120 |

#### Per family

| Family | original | cuefree | A | B |
| --- | --- | --- | --- | --- |
| alternatives | 40/40; 20/20; 20/20 | 40/40; 20/20; 20/20 | 38/40; 18/20; 20/20 | n/a |
| chain_depth | 40/40; 20/20; 20/20 | 40/40; 20/20; 20/20 | 39/40; 20/20; 19/20 | n/a |
| nesting | 36/40; 16/20; 20/20 | 32/40; 16/20; 16/20 | 20/40; 0/20; 20/20 | n/a |
| random_multi_rule | n/a | n/a | n/a | 146/240; 46/120; 100/120 |
| recursion | 40/40; 20/20; 20/20 | 28/40; 20/20; 8/20 | 29/40; 20/20; 9/20 | n/a |
| sequence | 40/40; 20/20; 20/20 | 32/40; 20/20; 12/20 | 22/40; 20/20; 2/20 | n/a |
| terminal_length | 40/40; 20/20; 20/20 | 32/40; 20/20; 12/20 | 39/40; 20/20; 19/20 | n/a |

#### Per size

| Size | original | cuefree | A | B |
| --- | --- | --- | --- | --- |
| 1 | 47/48; 23/24; 24/24 | 47/48; 23/24; 24/24 | 40/48; 20/24; 20/24 | 41/48; 24/24; 17/24 |
| 2 | 45/48; 21/24; 24/24 | 45/48; 21/24; 24/24 | 40/48; 20/24; 20/24 | 31/48; 16/24; 15/24 |
| 4 | 48/48; 24/24; 24/24 | 44/48; 24/24; 20/24 | 38/48; 20/24; 18/24 | 25/48; 5/24; 20/24 |
| 8 | 48/48; 24/24; 24/24 | 34/48; 24/24; 10/24 | 35/48; 20/24; 15/24 | 25/48; 1/24; 24/24 |
| 16 | 48/48; 24/24; 24/24 | 34/48; 24/24; 10/24 | 34/48; 18/24; 16/24 | 24/48; 0/24; 24/24 |

### Full execution in one reply

| Suite | All | Yes | No |
| --- | --- | --- | --- |
| original | 28/240 | 28/120 | 0/120 |
| cuefree | 30/240 | 28/120 | 2/120 |
| A | 32/240 | 28/120 | 4/120 |
| B | 5/240 | 3/120 | 2/120 |

#### Per family

| Family | original | cuefree | A | B |
| --- | --- | --- | --- | --- |
| alternatives | 4/40; 4/20; 0/20 | 6/40; 4/20; 2/20 | 8/40; 5/20; 3/20 | n/a |
| chain_depth | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 3/40; 3/20; 0/20 | n/a |
| nesting | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | n/a |
| random_multi_rule | n/a | n/a | n/a | 5/240; 3/120; 2/120 |
| recursion | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | n/a |
| sequence | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | n/a |
| terminal_length | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 21/40; 20/20; 1/20 | n/a |

#### Per size

| Size | original | cuefree | A | B |
| --- | --- | --- | --- | --- |
| 1 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 11/48; 11/24; 0/24 | 4/48; 3/24; 1/24 |
| 2 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 6/48; 4/24; 2/24 | 0/48; 0/24; 0/24 |
| 4 | 4/48; 4/24; 0/24 | 5/48; 4/24; 1/24 | 6/48; 5/24; 1/24 | 0/48; 0/24; 0/24 |
| 8 | 4/48; 4/24; 0/24 | 5/48; 4/24; 1/24 | 5/48; 4/24; 1/24 | 1/48; 0/24; 1/24 |
| 16 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 0/48; 0/24; 0/24 |

### Gym memory

| Suite | All | Yes | No |
| --- | --- | --- | --- |
| original | 40/240 | 40/120 | 0/120 |
| cuefree | 40/240 | 40/120 | 0/120 |
| A | 28/240 | 28/120 | 0/120 |
| B | 4/240 | 4/120 | 0/120 |

#### Per family

| Family | original | cuefree | A | B |
| --- | --- | --- | --- | --- |
| alternatives | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | n/a |
| chain_depth | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | n/a |
| nesting | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | n/a |
| random_multi_rule | n/a | n/a | n/a | 4/240; 4/120; 0/120 |
| recursion | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | n/a |
| sequence | 12/40; 12/20; 0/20 | 12/40; 12/20; 0/20 | 0/40; 0/20; 0/20 | n/a |
| terminal_length | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | n/a |

#### Per size

| Size | original | cuefree | A | B |
| --- | --- | --- | --- | --- |
| 1 | 8/48; 8/24; 0/24 | 8/48; 8/24; 0/24 | 8/48; 8/24; 0/24 | 4/48; 4/24; 0/24 |
| 2 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 8/48; 8/24; 0/24 | 0/48; 0/24; 0/24 |
| 4 | 8/48; 8/24; 0/24 | 8/48; 8/24; 0/24 | 4/48; 4/24; 0/24 | 0/48; 0/24; 0/24 |
| 8 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 0/48; 0/24; 0/24 |
| 16 | 8/48; 8/24; 0/24 | 8/48; 8/24; 0/24 | 4/48; 4/24; 0/24 | 0/48; 0/24; 0/24 |

### Correct-state check (all actions correct)

| Suite | All | Yes | No |
| --- | --- | --- | --- |
| original | 40/240 | 40/120 | 0/120 |
| cuefree | 40/240 | 40/120 | 0/120 |
| A | 28/240 | 28/120 | 0/120 |
| B | 4/240 | 4/120 | 0/120 |

#### Per family

| Family | original | cuefree | A | B |
| --- | --- | --- | --- | --- |
| alternatives | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | n/a |
| chain_depth | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | n/a |
| nesting | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | n/a |
| random_multi_rule | n/a | n/a | n/a | 4/240; 4/120; 0/120 |
| recursion | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | n/a |
| sequence | 12/40; 12/20; 0/20 | 12/40; 12/20; 0/20 | 0/40; 0/20; 0/20 | n/a |
| terminal_length | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | n/a |

#### Per size

| Size | original | cuefree | A | B |
| --- | --- | --- | --- | --- |
| 1 | 8/48; 8/24; 0/24 | 8/48; 8/24; 0/24 | 8/48; 8/24; 0/24 | 4/48; 4/24; 0/24 |
| 2 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 8/48; 8/24; 0/24 | 0/48; 0/24; 0/24 |
| 4 | 8/48; 8/24; 0/24 | 8/48; 8/24; 0/24 | 4/48; 4/24; 0/24 | 0/48; 0/24; 0/24 |
| 8 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 0/48; 0/24; 0/24 |
| 16 | 8/48; 8/24; 0/24 | 8/48; 8/24; 0/24 | 4/48; 4/24; 0/24 | 0/48; 0/24; 0/24 |

### Direct-answer yes-rate and malformed replies

The all-case yes-rate counts malformed replies as neither yes nor no. The valid-reply rate is reported separately; no malformed response is removed from accuracy denominators.

#### Per family

| Suite | Cell | Yes / all cases | Yes-rate (all) | Yes-rate (valid) | Malformed |
| --- | --- | --- | --- | --- | --- |
| original | alternatives | 20/40 | 50.0% | 50.0% | 0 |
| original | chain_depth | 20/40 | 50.0% | 50.0% | 0 |
| original | nesting | 16/40 | 40.0% | 40.0% | 0 |
| original | recursion | 20/40 | 50.0% | 50.0% | 0 |
| original | sequence | 20/40 | 50.0% | 50.0% | 0 |
| original | terminal_length | 20/40 | 50.0% | 50.0% | 0 |
| cuefree | alternatives | 20/40 | 50.0% | 50.0% | 0 |
| cuefree | chain_depth | 20/40 | 50.0% | 50.0% | 0 |
| cuefree | nesting | 20/40 | 50.0% | 50.0% | 0 |
| cuefree | recursion | 32/40 | 80.0% | 80.0% | 0 |
| cuefree | sequence | 28/40 | 70.0% | 70.0% | 0 |
| cuefree | terminal_length | 28/40 | 70.0% | 70.0% | 0 |
| A | alternatives | 18/40 | 45.0% | 45.0% | 0 |
| A | chain_depth | 21/40 | 52.5% | 52.5% | 0 |
| A | nesting | 0/40 | 0.0% | 0.0% | 0 |
| A | recursion | 31/40 | 77.5% | 77.5% | 0 |
| A | sequence | 38/40 | 95.0% | 95.0% | 0 |
| A | terminal_length | 21/40 | 52.5% | 52.5% | 0 |
| B | random_multi_rule | 66/240 | 27.5% | 27.5% | 0 |

#### Per size

| Suite | Cell | Yes / all cases | Yes-rate (all) | Yes-rate (valid) | Malformed |
| --- | --- | --- | --- | --- | --- |
| original | 1 | 23/48 | 47.9% | 47.9% | 0 |
| original | 2 | 21/48 | 43.8% | 43.8% | 0 |
| original | 4 | 24/48 | 50.0% | 50.0% | 0 |
| original | 8 | 24/48 | 50.0% | 50.0% | 0 |
| original | 16 | 24/48 | 50.0% | 50.0% | 0 |
| cuefree | 1 | 23/48 | 47.9% | 47.9% | 0 |
| cuefree | 2 | 21/48 | 43.8% | 43.8% | 0 |
| cuefree | 4 | 28/48 | 58.3% | 58.3% | 0 |
| cuefree | 8 | 38/48 | 79.2% | 79.2% | 0 |
| cuefree | 16 | 38/48 | 79.2% | 79.2% | 0 |
| A | 1 | 24/48 | 50.0% | 50.0% | 0 |
| A | 2 | 24/48 | 50.0% | 50.0% | 0 |
| A | 4 | 26/48 | 54.2% | 54.2% | 0 |
| A | 8 | 29/48 | 60.4% | 60.4% | 0 |
| A | 16 | 26/48 | 54.2% | 54.2% | 0 |
| B | 1 | 31/48 | 64.6% | 64.6% | 0 |
| B | 2 | 25/48 | 52.1% | 52.1% | 0 |
| B | 4 | 9/48 | 18.8% | 18.8% | 0 |
| B | 8 | 1/48 | 2.1% | 2.1% | 0 |
| B | 16 | 0/48 | 0.0% | 0.0% | 0 |

### Baseline correct-state operator accuracy

DONE is shown first. TRY_already_tried is a wrong TRY naming a production already in the current teacher frame's tried list, not just an arbitrary wrong production. Every queried state remains in the denominator.

#### original

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 20/492 | 4.1% | BT: 28, PRN: 9, REJ: 42, TRY: 9, TRY_already_tried: 384 |
| TRY | 991/1604 | 61.8% | PRN: 76, REJ: 7, TRY: 325, TRY_already_tried: 205 |
| PRN | 182/636 | 28.6% | BT: 170, CUT: 13, PRN: 93, REJ: 54, TRY: 124 |
| BT | 122/1008 | 12.1% | BT: 7, CUT: 1, DONE: 39, PRN: 50, REJ: 36, TRY: 370, TRY_already_tried: 383 |
| ACC | 102/120 | 85.0% | PRN: 12, TRY: 6 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### cuefree

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 48/516 | 9.3% | PRN: 9, REJ: 16, TRY: 15, TRY_already_tried: 428 |
| TRY | 1046/1528 | 68.5% | PRN: 8, TRY: 346, TRY_already_tried: 128 |
| PRN | 90/536 | 16.8% | ACC: 26, BT: 159, CUT: 33, PRN: 74, REJ: 6, TRY: 148 |
| BT | 96/932 | 10.3% | BT: 3, DONE: 53, PRN: 12, REJ: 18, TRY: 361, TRY_already_tried: 389 |
| ACC | 102/120 | 85.0% | PRN: 12, TRY: 6 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### A

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 49/512 | 9.6% | BT: 15, CUT: 1, REJ: 1, TRY: 43, TRY_already_tried: 403 |
| TRY | 1513/2324 | 65.1% | ACC: 3, PRN: 4, TRY: 343, TRY_already_tried: 461 |
| PRN | 167/1068 | 15.6% | ACC: 2, BT: 50, CUT: 7, DONE: 1, PRN: 19, TRY: 822 |
| BT | 86/1460 | 5.9% | BT: 13, CUT: 1, DONE: 64, TRY: 489, TRY_already_tried: 807 |
| ACC | 90/120 | 75.0% | DONE: 7, PRN: 11, TRY: 12 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### B

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 0/618 | 0.0% | BT: 3, CUT: 2, REJ: 1, TRY: 55, TRY_already_tried: 557 |
| TRY | 1525/2863 | 53.3% | PRN: 17, TRY: 869, TRY_already_tried: 452 |
| PRN | 389/1606 | 24.2% | ACC: 5, BT: 43, CUT: 12, PRN: 29, REJ: 7, TRY: 1121 |
| BT | 136/2119 | 6.4% | BT: 11, DONE: 1, PRN: 7, TRY: 1061, TRY_already_tried: 903 |
| ACC | 22/120 | 18.3% | PRN: 65, TRY: 33 |
| REJ | 0/15 | 0.0% | BT: 1, PRN: 7, TRY: 7 |
| CUT | 0/0 | n/a | none |

## Part 3: offloading atlas

| Condition | Hints | Live guard |
| --- | --- | --- |
| H0 | none | off |
| H1 | untried_rules | off |
| H2 | none | one re-ask |
| H3 | untried_rules | one re-ask |
| H4 | untried_rules, prefix_matches, complete_match | off |
| H5 | untried_rules, prefix_matches, complete_match, next_untried | off |

Each suite has one serving job with fresh H0-H5 in order. The current frame alone gets enabled fields; no hint names an opcode. Enabled prompts add the same one-sentence explanation, independent of condition. The unmodified prompt and observation are preserved byte-for-byte with hints off. untried_rules and next_untried use grammar order and the frame tried list; prefix_matches uses the validator classification without its depth cutoff; complete_match requires exact terminal equality. Hints do not mutate the stored state or the grader.

The guard applies only in gym memory. A repeated TRY is not executed; the same state is queried once more with exactly "pK was already tried here". The second answer is final, even if it repeats the error or is malformed. All calls, including rejected first answers and second answers, count toward 512; no correct opcode or reference action is supplied. The activation decision uses only the tried list. H2 correct-state checks therefore equal the H0 interface; H3 equals H1. They are fresh queries, not reused responses, and deliberately do not measure guard-assisted accuracy.

### Correct-state all-case and action-level accuracy

| Suite | Condition | All actions correct (cases) | Correct actions / all states |
| --- | --- | --- | --- |
| original | H0 | 41/240 | 1393/3860 |
| original | H1 | 62/240 | 1825/3860 |
| original | H2 | 39/240 | 1419/3860 |
| original | H3 | 62/240 | 1820/3860 |
| original | H4 | 64/240 | 1973/3860 |
| original | H5 | 64/240 | 2138/3860 |
| cuefree | H0 | 40/240 | 1370/3632 |
| cuefree | H1 | 62/240 | 1838/3632 |
| cuefree | H2 | 39/240 | 1378/3632 |
| cuefree | H3 | 62/240 | 1843/3632 |
| cuefree | H4 | 64/240 | 2022/3632 |
| cuefree | H5 | 64/240 | 2128/3632 |
| A | H0 | 28/240 | 1903/5484 |
| A | H1 | 64/240 | 2526/5484 |
| A | H2 | 28/240 | 1896/5484 |
| A | H3 | 64/240 | 2523/5484 |
| A | H4 | 64/240 | 2668/5484 |
| A | H5 | 64/240 | 3118/5484 |
| B | H0 | 4/240 | 2065/7341 |
| B | H1 | 6/240 | 3785/7341 |
| B | H2 | 4/240 | 2077/7341 |
| B | H3 | 6/240 | 3779/7341 |
| B | H4 | 11/240 | 3876/7341 |
| B | H5 | 11/240 | 3974/7341 |

### original: complete gym-memory runs

Entries are all; yes; no. Each complete-run denominator is 240 overall, with 120 per label.

| Condition | All | Yes | No |
| --- | --- | --- | --- |
| H0 | 41/240 | 41/120 | 0/120 |
| H1 | 62/240 | 62/120 | 0/120 |
| H2 | 39/240 | 39/120 | 0/120 |
| H3 | 62/240 | 62/120 | 0/120 |
| H4 | 64/240 | 64/120 | 0/120 |
| H5 | 64/240 | 64/120 | 0/120 |

#### Per family

| Cell | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| alternatives | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 |
| chain_depth | 4/40; 4/20; 0/20 | 20/40; 20/20; 0/20 | 4/40; 4/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 |
| nesting | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 |
| recursion | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 |
| sequence | 13/40; 13/20; 0/20 | 18/40; 18/20; 0/20 | 11/40; 11/20; 0/20 | 18/40; 18/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 |
| terminal_length | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 |

#### Per size

| Cell | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8/48; 8/24; 0/24 | 16/48; 16/24; 0/24 | 8/48; 8/24; 0/24 | 16/48; 16/24; 0/24 | 16/48; 16/24; 0/24 | 16/48; 16/24; 0/24 |
| 2 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 11/48; 11/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |
| 4 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |
| 8 | 5/48; 5/24; 0/24 | 10/48; 10/24; 0/24 | 4/48; 4/24; 0/24 | 10/48; 10/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |
| 16 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |

#### original H0: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 19/492 | 3.9% | BT: 29, PRN: 9, REJ: 42, TRY: 10, TRY_already_tried: 383 |
| TRY | 987/1604 | 61.5% | PRN: 77, REJ: 8, TRY: 325, TRY_already_tried: 207 |
| PRN | 177/636 | 27.8% | BT: 169, CUT: 15, PRN: 94, REJ: 58, TRY: 123 |
| BT | 107/1008 | 10.6% | BT: 8, DONE: 40, PRN: 53, REJ: 34, TRY: 371, TRY_already_tried: 395 |
| ACC | 103/120 | 85.8% | PRN: 11, TRY: 6 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### original H1: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 53/492 | 10.8% | BT: 113, CUT: 7, PRN: 34, REJ: 71, TRY_already_tried: 214 |
| TRY | 1142/1604 | 71.2% | PRN: 52, TRY: 410 |
| PRN | 159/636 | 25.0% | ACC: 5, BT: 180, PRN: 135, REJ: 82, TRY: 75 |
| BT | 360/1008 | 35.7% | CUT: 1, DONE: 75, PRN: 38, REJ: 36, TRY: 479, TRY_already_tried: 19 |
| ACC | 111/120 | 92.5% | PRN: 9 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### original H2: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 20/492 | 4.1% | BT: 29, PRN: 9, REJ: 41, TRY: 8, TRY_already_tried: 385 |
| TRY | 989/1604 | 61.7% | PRN: 77, REJ: 7, TRY: 328, TRY_already_tried: 203 |
| PRN | 184/636 | 28.9% | BT: 168, CUT: 13, PRN: 90, REJ: 56, TRY: 125 |
| BT | 123/1008 | 12.2% | BT: 6, DONE: 40, PRN: 47, REJ: 34, TRY: 368, TRY_already_tried: 390 |
| ACC | 103/120 | 85.8% | PRN: 11, TRY: 6 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### original H3: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 53/492 | 10.8% | BT: 115, CUT: 5, PRN: 33, REJ: 71, TRY_already_tried: 215 |
| TRY | 1140/1604 | 71.1% | PRN: 52, TRY: 412 |
| PRN | 158/636 | 24.8% | ACC: 5, BT: 181, PRN: 132, REJ: 85, TRY: 75 |
| BT | 358/1008 | 35.5% | CUT: 1, DONE: 76, PRN: 40, REJ: 34, TRY: 479, TRY_already_tried: 20 |
| ACC | 111/120 | 92.5% | PRN: 9 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### original H4: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 49/492 | 10.0% | BT: 139, CUT: 21, PRN: 8, REJ: 92, TRY_already_tried: 183 |
| TRY | 1286/1604 | 80.2% | PRN: 34, TRY: 284 |
| PRN | 150/636 | 23.6% | BT: 152, CUT: 4, PRN: 2, REJ: 240, TRY: 88 |
| BT | 368/1008 | 36.5% | DONE: 48, PRN: 42, REJ: 45, TRY: 482, TRY_already_tried: 23 |
| ACC | 120/120 | 100.0% | none |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### original H5: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 38/492 | 7.7% | BT: 150, CUT: 47, PRN: 9, REJ: 86, TRY_already_tried: 162 |
| TRY | 1432/1604 | 89.3% | PRN: 62, TRY: 110 |
| PRN | 195/636 | 30.7% | BT: 149, CUT: 5, PRN: 4, REJ: 221, TRY: 62 |
| BT | 353/1008 | 35.0% | BT: 14, DONE: 39, PRN: 79, REJ: 23, TRY: 439, TRY_already_tried: 61 |
| ACC | 120/120 | 100.0% | none |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

### cuefree: complete gym-memory runs

Entries are all; yes; no. Each complete-run denominator is 240 overall, with 120 per label.

| Condition | All | Yes | No |
| --- | --- | --- | --- |
| H0 | 40/240 | 40/120 | 0/120 |
| H1 | 62/240 | 62/120 | 0/120 |
| H2 | 39/240 | 39/120 | 0/120 |
| H3 | 62/240 | 62/120 | 0/120 |
| H4 | 64/240 | 64/120 | 0/120 |
| H5 | 64/240 | 64/120 | 0/120 |

#### Per family

| Cell | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| alternatives | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 |
| chain_depth | 4/40; 4/20; 0/20 | 20/40; 20/20; 0/20 | 4/40; 4/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 |
| nesting | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 |
| recursion | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 |
| sequence | 12/40; 12/20; 0/20 | 18/40; 18/20; 0/20 | 11/40; 11/20; 0/20 | 18/40; 18/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 |
| terminal_length | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 |

#### Per size

| Cell | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8/48; 8/24; 0/24 | 16/48; 16/24; 0/24 | 8/48; 8/24; 0/24 | 16/48; 16/24; 0/24 | 16/48; 16/24; 0/24 | 16/48; 16/24; 0/24 |
| 2 | 11/48; 11/24; 0/24 | 12/48; 12/24; 0/24 | 11/48; 11/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |
| 4 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |
| 8 | 5/48; 5/24; 0/24 | 10/48; 10/24; 0/24 | 4/48; 4/24; 0/24 | 10/48; 10/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |
| 16 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |

#### cuefree H0: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 48/516 | 9.3% | PRN: 9, REJ: 16, TRY: 13, TRY_already_tried: 430 |
| TRY | 1043/1528 | 68.3% | PRN: 7, TRY: 349, TRY_already_tried: 129 |
| PRN | 83/536 | 15.5% | ACC: 27, BT: 163, CUT: 37, PRN: 74, REJ: 8, TRY: 144 |
| BT | 93/932 | 10.0% | BT: 3, DONE: 53, PRN: 11, REJ: 18, TRY: 363, TRY_already_tried: 391 |
| ACC | 103/120 | 85.8% | PRN: 11, TRY: 6 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### cuefree H1: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 54/516 | 10.5% | ACC: 14, BT: 75, CUT: 15, PRN: 65, REJ: 52, TRY: 4, TRY_already_tried: 237 |
| TRY | 1199/1528 | 78.5% | PRN: 4, TRY: 325 |
| PRN | 98/536 | 18.3% | ACC: 59, BT: 215, CUT: 3, PRN: 66, REJ: 22, TRY: 73 |
| BT | 376/932 | 40.3% | ACC: 5, DONE: 61, PRN: 16, REJ: 22, TRY: 416, TRY_already_tried: 36 |
| ACC | 111/120 | 92.5% | PRN: 9 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### cuefree H2: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 48/516 | 9.3% | PRN: 9, REJ: 16, TRY: 15, TRY_already_tried: 428 |
| TRY | 1042/1528 | 68.2% | PRN: 8, TRY: 349, TRY_already_tried: 129 |
| PRN | 91/536 | 17.0% | ACC: 28, BT: 156, CUT: 33, PRN: 71, REJ: 8, TRY: 149 |
| BT | 94/932 | 10.1% | BT: 3, DONE: 55, PRN: 10, REJ: 18, TRY: 362, TRY_already_tried: 390 |
| ACC | 103/120 | 85.8% | PRN: 11, TRY: 6 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### cuefree H3: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 54/516 | 10.5% | ACC: 14, BT: 74, CUT: 15, PRN: 66, REJ: 54, TRY: 4, TRY_already_tried: 235 |
| TRY | 1201/1528 | 78.6% | PRN: 4, TRY: 323 |
| PRN | 101/536 | 18.8% | ACC: 56, BT: 215, CUT: 3, PRN: 67, REJ: 21, TRY: 73 |
| BT | 376/932 | 40.3% | ACC: 5, BT: 2, DONE: 59, PRN: 16, REJ: 22, TRY: 416, TRY_already_tried: 36 |
| ACC | 111/120 | 92.5% | PRN: 9 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### cuefree H4: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 52/516 | 10.1% | BT: 118, CUT: 2, PRN: 12, REJ: 93, TRY_already_tried: 239 |
| TRY | 1322/1528 | 86.5% | TRY: 206 |
| PRN | 150/536 | 28.0% | ACC: 18, BT: 209, CUT: 5, PRN: 14, REJ: 56, TRY: 84 |
| BT | 378/932 | 40.6% | ACC: 2, DONE: 55, PRN: 17, REJ: 12, TRY: 416, TRY_already_tried: 52 |
| ACC | 120/120 | 100.0% | none |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### cuefree H5: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 45/516 | 8.7% | ACC: 3, BT: 188, CUT: 30, PRN: 8, REJ: 126, TRY_already_tried: 116 |
| TRY | 1458/1528 | 95.4% | TRY: 70 |
| PRN | 116/536 | 21.6% | ACC: 16, BT: 208, CUT: 8, PRN: 4, REJ: 110, TRY: 74 |
| BT | 389/932 | 41.7% | BT: 13, DONE: 40, PRN: 22, REJ: 9, TRY: 403, TRY_already_tried: 56 |
| ACC | 120/120 | 100.0% | none |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

### A: complete gym-memory runs

Entries are all; yes; no. Each complete-run denominator is 240 overall, with 120 per label.

| Condition | All | Yes | No |
| --- | --- | --- | --- |
| H0 | 28/240 | 28/120 | 0/120 |
| H1 | 64/240 | 64/120 | 0/120 |
| H2 | 31/240 | 28/120 | 3/120 |
| H3 | 64/240 | 64/120 | 0/120 |
| H4 | 64/240 | 64/120 | 0/120 |
| H5 | 64/240 | 64/120 | 0/120 |

#### Per family

| Cell | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| alternatives | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 | 4/40; 4/20; 0/20 |
| chain_depth | 4/40; 4/20; 0/20 | 20/40; 20/20; 0/20 | 4/40; 4/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 |
| nesting | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 |
| recursion | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 | 0/40; 0/20; 0/20 |
| sequence | 0/40; 0/20; 0/20 | 20/40; 20/20; 0/20 | 0/40; 0/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 |
| terminal_length | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 23/40; 20/20; 3/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 | 20/40; 20/20; 0/20 |

#### Per size

| Cell | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8/48; 8/24; 0/24 | 16/48; 16/24; 0/24 | 8/48; 8/24; 0/24 | 16/48; 16/24; 0/24 | 16/48; 16/24; 0/24 | 16/48; 16/24; 0/24 |
| 2 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 8/48; 8/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |
| 4 | 4/48; 4/24; 0/24 | 12/48; 12/24; 0/24 | 7/48; 4/24; 3/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |
| 8 | 4/48; 4/24; 0/24 | 12/48; 12/24; 0/24 | 4/48; 4/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |
| 16 | 4/48; 4/24; 0/24 | 12/48; 12/24; 0/24 | 4/48; 4/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 | 12/48; 12/24; 0/24 |

#### A H0: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 49/512 | 9.6% | BT: 15, CUT: 1, REJ: 1, TRY: 44, TRY_already_tried: 402 |
| TRY | 1522/2324 | 65.5% | ACC: 3, PRN: 4, TRY: 345, TRY_already_tried: 450 |
| PRN | 162/1068 | 15.2% | ACC: 3, BT: 48, CUT: 9, DONE: 1, PRN: 22, TRY: 823 |
| BT | 80/1460 | 5.5% | BT: 12, DONE: 64, TRY: 493, TRY_already_tried: 811 |
| ACC | 90/120 | 75.0% | DONE: 7, PRN: 11, TRY: 12 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### A H1: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 81/512 | 15.8% | BT: 61, CUT: 4, PRN: 28, REJ: 11, TRY: 18, TRY_already_tried: 309 |
| TRY | 1763/2324 | 75.9% | TRY: 561 |
| PRN | 256/1068 | 24.0% | ACC: 89, BT: 138, CUT: 4, DONE: 3, PRN: 39, REJ: 11, TRY: 528 |
| BT | 311/1460 | 21.3% | BT: 6, DONE: 92, PRN: 2, REJ: 1, TRY: 953, TRY_already_tried: 95 |
| ACC | 115/120 | 95.8% | PRN: 3, TRY: 2 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### A H2: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 49/512 | 9.6% | BT: 15, CUT: 1, REJ: 1, TRY: 44, TRY_already_tried: 402 |
| TRY | 1508/2324 | 64.9% | ACC: 3, PRN: 4, TRY: 344, TRY_already_tried: 465 |
| PRN | 164/1068 | 15.4% | ACC: 2, BT: 52, CUT: 7, PRN: 19, TRY: 824 |
| BT | 85/1460 | 5.8% | BT: 13, DONE: 65, TRY: 493, TRY_already_tried: 804 |
| ACC | 90/120 | 75.0% | DONE: 7, PRN: 11, TRY: 12 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### A H3: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 81/512 | 15.8% | BT: 63, CUT: 4, PRN: 25, REJ: 13, TRY: 19, TRY_already_tried: 307 |
| TRY | 1762/2324 | 75.8% | TRY: 562 |
| PRN | 252/1068 | 23.6% | ACC: 92, BT: 142, CUT: 3, DONE: 3, PRN: 39, REJ: 9, TRY: 528 |
| BT | 313/1460 | 21.4% | BT: 7, DONE: 91, PRN: 2, REJ: 1, TRY: 953, TRY_already_tried: 93 |
| ACC | 115/120 | 95.8% | PRN: 3, TRY: 2 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### A H4: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 62/512 | 12.1% | ACC: 3, BT: 101, CUT: 11, REJ: 28, TRY: 11, TRY_already_tried: 296 |
| TRY | 1945/2324 | 83.7% | TRY: 379 |
| PRN | 179/1068 | 16.8% | ACC: 3, BT: 261, CUT: 6, REJ: 106, TRY: 513 |
| BT | 365/1460 | 25.0% | ACC: 5, BT: 7, CUT: 1, DONE: 68, REJ: 1, TRY: 948, TRY_already_tried: 65 |
| ACC | 117/120 | 97.5% | DONE: 3 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

#### A H5: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 73/512 | 14.3% | BT: 202, CUT: 9, REJ: 61, TRY: 8, TRY_already_tried: 159 |
| TRY | 2275/2324 | 97.9% | TRY: 49 |
| PRN | 268/1068 | 25.1% | ACC: 2, BT: 293, CUT: 37, REJ: 194, TRY: 274 |
| BT | 386/1460 | 26.4% | BT: 3, DONE: 56, REJ: 2, TRY: 945, TRY_already_tried: 68 |
| ACC | 116/120 | 96.7% | DONE: 4 |
| REJ | 0/0 | n/a | none |
| CUT | 0/0 | n/a | none |

### B: complete gym-memory runs

Entries are all; yes; no. Each complete-run denominator is 240 overall, with 120 per label.

| Condition | All | Yes | No |
| --- | --- | --- | --- |
| H0 | 4/240 | 4/120 | 0/120 |
| H1 | 6/240 | 6/120 | 0/120 |
| H2 | 4/240 | 4/120 | 0/120 |
| H3 | 6/240 | 6/120 | 0/120 |
| H4 | 11/240 | 11/120 | 0/120 |
| H5 | 11/240 | 11/120 | 0/120 |

#### Per family

| Cell | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| random_multi_rule | 4/240; 4/120; 0/120 | 6/240; 6/120; 0/120 | 4/240; 4/120; 0/120 | 6/240; 6/120; 0/120 | 11/240; 11/120; 0/120 | 11/240; 11/120; 0/120 |

#### Per size

| Cell | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 4/48; 4/24; 0/24 | 9/48; 9/24; 0/24 | 9/48; 9/24; 0/24 |
| 2 | 0/48; 0/24; 0/24 | 2/48; 2/24; 0/24 | 0/48; 0/24; 0/24 | 2/48; 2/24; 0/24 | 2/48; 2/24; 0/24 | 2/48; 2/24; 0/24 |
| 4 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 |
| 8 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 |
| 16 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 | 0/48; 0/24; 0/24 |

#### B H0: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 0/618 | 0.0% | BT: 3, CUT: 2, REJ: 1, TRY: 54, TRY_already_tried: 558 |
| TRY | 1523/2863 | 53.2% | PRN: 17, TRY: 867, TRY_already_tried: 456 |
| PRN | 386/1606 | 24.0% | ACC: 5, BT: 43, CUT: 12, PRN: 29, REJ: 7, TRY: 1124 |
| BT | 136/2119 | 6.4% | BT: 11, DONE: 1, PRN: 7, TRY: 1060, TRY_already_tried: 904 |
| ACC | 20/120 | 16.7% | PRN: 67, TRY: 33 |
| REJ | 0/15 | 0.0% | BT: 1, PRN: 7, TRY: 7 |
| CUT | 0/0 | n/a | none |

#### B H1: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 72/618 | 11.7% | BT: 114, CUT: 61, PRN: 46, REJ: 16, TRY: 24, TRY_already_tried: 285 |
| TRY | 2707/2863 | 94.6% | PRN: 2, TRY: 154 |
| PRN | 459/1606 | 28.6% | ACC: 183, BT: 114, CUT: 87, DONE: 7, PRN: 50, REJ: 55, TRY: 651 |
| BT | 470/2119 | 22.2% | BT: 15, CUT: 3, DONE: 67, PRN: 21, REJ: 1, TRY: 1495, TRY_already_tried: 47 |
| ACC | 77/120 | 64.2% | CUT: 2, DONE: 1, PRN: 40 |
| REJ | 0/15 | 0.0% | ACC: 1, BT: 2, PRN: 11, TRY: 1 |
| CUT | 0/0 | n/a | none |

#### B H2: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 0/618 | 0.0% | BT: 3, CUT: 2, REJ: 1, TRY: 54, TRY_already_tried: 558 |
| TRY | 1530/2863 | 53.4% | PRN: 17, TRY: 866, TRY_already_tried: 450 |
| PRN | 387/1606 | 24.1% | ACC: 4, BT: 43, CUT: 12, PRN: 29, REJ: 7, TRY: 1124 |
| BT | 137/2119 | 6.5% | BT: 11, DONE: 1, PRN: 7, TRY: 1062, TRY_already_tried: 901 |
| ACC | 23/120 | 19.2% | PRN: 65, TRY: 32 |
| REJ | 0/15 | 0.0% | BT: 1, PRN: 7, TRY: 7 |
| CUT | 0/0 | n/a | none |

#### B H3: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 72/618 | 11.7% | BT: 115, CUT: 62, PRN: 45, REJ: 16, TRY: 25, TRY_already_tried: 283 |
| TRY | 2707/2863 | 94.6% | PRN: 2, TRY: 154 |
| PRN | 453/1606 | 28.2% | ACC: 182, BT: 115, CUT: 93, DONE: 8, PRN: 51, REJ: 53, TRY: 651 |
| BT | 470/2119 | 22.2% | BT: 14, CUT: 4, DONE: 68, PRN: 20, REJ: 1, TRY: 1495, TRY_already_tried: 47 |
| ACC | 77/120 | 64.2% | CUT: 2, DONE: 1, PRN: 40 |
| REJ | 0/15 | 0.0% | ACC: 1, BT: 2, PRN: 11, TRY: 1 |
| CUT | 0/0 | n/a | none |

#### B H4: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 54/618 | 8.7% | BT: 173, CUT: 67, PRN: 13, REJ: 56, TRY: 23, TRY_already_tried: 232 |
| TRY | 2762/2863 | 96.5% | TRY: 101 |
| PRN | 396/1606 | 24.7% | ACC: 1, BT: 184, CUT: 212, PRN: 1, REJ: 180, TRY: 632 |
| BT | 545/2119 | 25.7% | BT: 19, DONE: 51, PRN: 1, REJ: 1, TRY: 1495, TRY_already_tried: 7 |
| ACC | 119/120 | 99.2% | TRY: 1 |
| REJ | 0/15 | 0.0% | BT: 3, PRN: 5, TRY: 7 |
| CUT | 0/0 | n/a | none |

#### B H5: raw correct-state operators

| Operator | Correct / total | Accuracy | Wrong substitutions (counts) |
| --- | --- | --- | --- |
| DONE | 37/618 | 6.0% | BT: 348, CUT: 118, REJ: 80, TRY: 6, TRY_already_tried: 29 |
| TRY | 2831/2863 | 98.9% | TRY: 32 |
| PRN | 387/1606 | 24.1% | BT: 258, CUT: 256, DONE: 7, PRN: 2, REJ: 380, TRY: 316 |
| BT | 601/2119 | 28.4% | BT: 11, DONE: 14, REJ: 3, TRY: 1488, TRY_already_tried: 2 |
| ACC | 118/120 | 98.3% | DONE: 2 |
| REJ | 0/15 | 0.0% | ACC: 1, BT: 8, CUT: 3, TRY: 3 |
| CUT | 0/0 | n/a | none |

### Guard outcomes

| Suite | Condition | Cases activated | Activations | Re-asks | Second answers correct | Retries blocked by call cap |
| --- | --- | --- | --- | --- | --- | --- |
| original | H2 | 0/240 | 0 | 0 | 0 | 0 |
| original | H3 | 0/240 | 0 | 0 | 0 | 0 |
| cuefree | H2 | 0/240 | 0 | 0 | 0 | 0 |
| cuefree | H3 | 3/240 | 6 | 6 | 3 | 0 |
| A | H2 | 9/240 | 10 | 10 | 4 | 0 |
| A | H3 | 2/240 | 3 | 3 | 1 | 0 |
| B | H2 | 0/240 | 0 | 0 | 0 | 0 |
| B | H3 | 0/240 | 0 | 0 | 0 | 0 |

Second-answer correctness is the unchanged canonical action test at the same state, not merely the absence of another repeated rule. A corrected second action does not guarantee the whole run finishes.

### Live first errors versus teacher-state errors

Teacher-state queries deliberately reach states that a flawed live execution may never reach. A large teacher-state DONE error count is therefore not automatically the first deployed bottleneck. The following counts classify the expected operator at the first final live error; corrected guard attempts are not counted as final failures. Wrong reason strings and wrong production IDs remain strict protocol failures even when an opcode appears plausible.

| Suite | Condition | Failed live cases | DONE | TRY | PRN | BT | ACC | REJ | CUT | Budget / unresolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | H0 | 199 | 0 | 128 | 68 | 3 | 0 | 0 | 0 | 0 |
| original | H1 | 178 | 10 | 94 | 62 | 12 | 0 | 0 | 0 | 0 |
| original | H2 | 201 | 0 | 129 | 68 | 4 | 0 | 0 | 0 | 0 |
| original | H3 | 178 | 10 | 94 | 62 | 12 | 0 | 0 | 0 | 0 |
| original | H4 | 176 | 11 | 80 | 82 | 3 | 0 | 0 | 0 | 0 |
| original | H5 | 176 | 7 | 53 | 91 | 25 | 0 | 0 | 0 | 0 |
| cuefree | H0 | 200 | 0 | 99 | 98 | 3 | 0 | 0 | 0 | 0 |
| cuefree | H1 | 178 | 0 | 80 | 93 | 5 | 0 | 0 | 0 | 0 |
| cuefree | H2 | 201 | 0 | 102 | 97 | 2 | 0 | 0 | 0 | 0 |
| cuefree | H3 | 178 | 3 | 80 | 93 | 2 | 0 | 0 | 0 | 0 |
| cuefree | H4 | 176 | 0 | 65 | 100 | 11 | 0 | 0 | 0 | 0 |
| cuefree | H5 | 176 | 5 | 35 | 108 | 28 | 0 | 0 | 0 | 0 |
| A | H0 | 212 | 7 | 118 | 46 | 41 | 0 | 0 | 0 | 0 |
| A | H1 | 176 | 5 | 58 | 73 | 40 | 0 | 0 | 0 | 0 |
| A | H2 | 209 | 5 | 118 | 42 | 44 | 0 | 0 | 0 | 0 |
| A | H3 | 176 | 6 | 58 | 73 | 39 | 0 | 0 | 0 | 0 |
| A | H4 | 176 | 5 | 39 | 104 | 28 | 0 | 0 | 0 | 0 |
| A | H5 | 176 | 17 | 13 | 107 | 39 | 0 | 0 | 0 | 0 |
| B | H0 | 236 | 0 | 110 | 57 | 59 | 5 | 5 | 0 | 0 |
| B | H1 | 234 | 0 | 48 | 110 | 66 | 5 | 5 | 0 | 0 |
| B | H2 | 236 | 0 | 110 | 57 | 59 | 5 | 5 | 0 | 0 |
| B | H3 | 234 | 0 | 48 | 110 | 66 | 5 | 5 | 0 | 0 |
| B | H4 | 229 | 0 | 32 | 148 | 44 | 0 | 5 | 0 | 0 |
| B | H5 | 229 | 0 | 11 | 139 | 74 | 0 | 5 | 0 | 0 |

First-error substitutions and inspectable examples are saved in [live_errors.json](results/countpreserving_offloading/live_errors.json). If a guard condition records zero activations, it supplied no re-ask or changed prompt; any score difference from its unguarded counterpart is not evidence for guard effectiveness. This study must distinguish latent exhaustion errors from earlier action/argument errors.

### Matched changes versus fresh H0

| Suite | Condition | Improved gym cases | Regressed gym cases |
| --- | --- | --- | --- |
| original | H0 | 0 | 0 |
| original | H1 | 21 | 0 |
| original | H2 | 0 | 2 |
| original | H3 | 21 | 0 |
| original | H4 | 23 | 0 |
| original | H5 | 23 | 0 |
| cuefree | H0 | 0 | 0 |
| cuefree | H1 | 22 | 0 |
| cuefree | H2 | 0 | 1 |
| cuefree | H3 | 22 | 0 |
| cuefree | H4 | 24 | 0 |
| cuefree | H5 | 24 | 0 |
| A | H0 | 0 | 0 |
| A | H1 | 36 | 0 |
| A | H2 | 3 | 0 |
| A | H3 | 36 | 0 |
| A | H4 | 36 | 0 |
| A | H5 | 36 | 0 |
| B | H0 | 0 | 0 |
| B | H1 | 4 | 2 |
| B | H2 | 0 | 0 |
| B | H3 | 4 | 2 |
| B | H4 | 7 | 0 |
| B | H5 | 7 | 0 |

### Historical baseline versus fresh H0

| Suite | Interface | Outputs changed (cases) | Improved | Regressed |
| --- | --- | --- | --- | --- |
| original | Gym memory | 6 | 1 | 0 |
| original | Correct-state check (all actions correct) | 67 | 1 | 0 |
| cuefree | Gym memory | 14 | 1 | 1 |
| cuefree | Correct-state check (all actions correct) | 45 | 1 | 1 |
| A | Gym memory | 5 | 0 | 0 |
| A | Correct-state check (all actions correct) | 50 | 0 | 0 |
| B | Gym memory | 0 | 0 | 0 |
| B | Correct-state check (all actions correct) | 34 | 0 | 0 |

Fresh H0, not a historical score, is the comparator for hint effects. Prompts and nominal batching are matched; greedy inference can still have numerical/runtime differences. No output was used to choose a rerun, prompt change, or case exclusion.

## Failure accounting and call budgets

| Part | Suite | Condition | Interface | Failure | Cases |
| --- | --- | --- | --- | --- | --- |
| baseline | original | zero-shot | Direct answer | incorrect_answer | 4 |
| baseline | original | zero-shot | Full execution in one reply | call_or_parse_error | 26 |
| baseline | original | zero-shot | Full execution in one reply | incomplete_trace | 2 |
| baseline | original | zero-shot | Full execution in one reply | invalid_or_trailing_action | 184 |
| baseline | original | zero-shot | Gym memory | invalid_action | 200 |
| baseline | original | zero-shot | Correct-state check (all actions correct) | invalid_action | 200 |
| baseline | cuefree | zero-shot | Direct answer | incorrect_answer | 36 |
| baseline | cuefree | zero-shot | Full execution in one reply | call_or_parse_error | 38 |
| baseline | cuefree | zero-shot | Full execution in one reply | invalid_or_trailing_action | 172 |
| baseline | cuefree | zero-shot | Gym memory | invalid_action | 200 |
| baseline | cuefree | zero-shot | Correct-state check (all actions correct) | invalid_action | 200 |
| baseline | A | zero-shot | Direct answer | incorrect_answer | 53 |
| baseline | A | zero-shot | Full execution in one reply | call_or_parse_error | 60 |
| baseline | A | zero-shot | Full execution in one reply | incomplete_trace | 11 |
| baseline | A | zero-shot | Full execution in one reply | invalid_or_trailing_action | 137 |
| baseline | A | zero-shot | Gym memory | invalid_action | 212 |
| baseline | A | zero-shot | Correct-state check (all actions correct) | invalid_action | 212 |
| baseline | B | zero-shot | Direct answer | incorrect_answer | 94 |
| baseline | B | zero-shot | Full execution in one reply | call_or_parse_error | 155 |
| baseline | B | zero-shot | Full execution in one reply | invalid_or_trailing_action | 80 |
| baseline | B | zero-shot | Gym memory | invalid_action | 236 |
| baseline | B | zero-shot | Correct-state check (all actions correct) | invalid_action | 236 |
| offloading | original | H0 | Gym memory | invalid_action | 199 |
| offloading | original | H0 | Correct-state check (all actions correct) | invalid_action | 199 |
| offloading | original | H1 | Gym memory | invalid_action | 178 |
| offloading | original | H1 | Correct-state check (all actions correct) | invalid_action | 178 |
| offloading | original | H2 | Gym memory | invalid_action | 201 |
| offloading | original | H2 | Correct-state check (all actions correct) | invalid_action | 201 |
| offloading | original | H3 | Gym memory | invalid_action | 178 |
| offloading | original | H3 | Correct-state check (all actions correct) | invalid_action | 178 |
| offloading | original | H4 | Gym memory | invalid_action | 176 |
| offloading | original | H4 | Correct-state check (all actions correct) | invalid_action | 176 |
| offloading | original | H5 | Gym memory | invalid_action | 176 |
| offloading | original | H5 | Correct-state check (all actions correct) | invalid_action | 176 |
| offloading | cuefree | H0 | Gym memory | invalid_action | 200 |
| offloading | cuefree | H0 | Correct-state check (all actions correct) | invalid_action | 200 |
| offloading | cuefree | H1 | Gym memory | invalid_action | 178 |
| offloading | cuefree | H1 | Correct-state check (all actions correct) | invalid_action | 178 |
| offloading | cuefree | H2 | Gym memory | invalid_action | 201 |
| offloading | cuefree | H2 | Correct-state check (all actions correct) | invalid_action | 201 |
| offloading | cuefree | H3 | Gym memory | invalid_action | 178 |
| offloading | cuefree | H3 | Correct-state check (all actions correct) | invalid_action | 178 |
| offloading | cuefree | H4 | Gym memory | invalid_action | 176 |
| offloading | cuefree | H4 | Correct-state check (all actions correct) | invalid_action | 176 |
| offloading | cuefree | H5 | Gym memory | invalid_action | 176 |
| offloading | cuefree | H5 | Correct-state check (all actions correct) | invalid_action | 176 |
| offloading | A | H0 | Gym memory | invalid_action | 212 |
| offloading | A | H0 | Correct-state check (all actions correct) | invalid_action | 212 |
| offloading | A | H1 | Gym memory | invalid_action | 176 |
| offloading | A | H1 | Correct-state check (all actions correct) | invalid_action | 176 |
| offloading | A | H2 | Gym memory | invalid_action | 209 |
| offloading | A | H2 | Correct-state check (all actions correct) | invalid_action | 212 |
| offloading | A | H3 | Gym memory | invalid_action | 176 |
| offloading | A | H3 | Correct-state check (all actions correct) | invalid_action | 176 |
| offloading | A | H4 | Gym memory | invalid_action | 176 |
| offloading | A | H4 | Correct-state check (all actions correct) | invalid_action | 176 |
| offloading | A | H5 | Gym memory | invalid_action | 176 |
| offloading | A | H5 | Correct-state check (all actions correct) | invalid_action | 176 |
| offloading | B | H0 | Gym memory | invalid_action | 236 |
| offloading | B | H0 | Correct-state check (all actions correct) | invalid_action | 236 |
| offloading | B | H1 | Gym memory | invalid_action | 234 |
| offloading | B | H1 | Correct-state check (all actions correct) | invalid_action | 234 |
| offloading | B | H2 | Gym memory | invalid_action | 236 |
| offloading | B | H2 | Correct-state check (all actions correct) | invalid_action | 236 |
| offloading | B | H3 | Gym memory | invalid_action | 234 |
| offloading | B | H3 | Correct-state check (all actions correct) | invalid_action | 234 |
| offloading | B | H4 | Gym memory | invalid_action | 229 |
| offloading | B | H4 | Correct-state check (all actions correct) | invalid_action | 229 |
| offloading | B | H5 | Gym memory | invalid_action | 229 |
| offloading | B | H5 | Correct-state check (all actions correct) | invalid_action | 229 |

Malformed responses, invalid actions, incomplete traces and exhausted budgets count as failures. Correct-state failures count per state and per all-actions-correct case; teacher-state checks continue after an error without revealing the answer. Live execution stops on the first final invalid action. There was no injected infrastructure/tool-failure recovery test.

### Baseline costs

| Suite | Interface | Calls | Completion tokens | Total tokens | Length-limited | Fresh cases |
| --- | --- | --- | --- | --- | --- | --- |
| original | Direct answer | 240 | 1440 | 48528 | 0 | 0 |
| original | Gym memory | 930 | 10019 | 633418 | 0 | 0 |
| original | Full execution in one reply | 240 | 116452 | 237076 | 26 | 0 |
| original | Correct-state check (all actions correct) | 3860 | 40413 | 2867477 | 0 | 0 |
| cuefree | Direct answer | 240 | 1440 | 48900 | 0 | 0 |
| cuefree | Gym memory | 981 | 10474 | 688784 | 0 | 0 |
| cuefree | Full execution in one reply | 240 | 167077 | 288073 | 38 | 0 |
| cuefree | Correct-state check (all actions correct) | 3632 | 37959 | 2701463 | 0 | 0 |
| A | Direct answer | 240 | 1440 | 56984 | 0 | 240 |
| A | Gym memory | 715 | 7671 | 445553 | 0 | 240 |
| A | Full execution in one reply | 240 | 257442 | 386562 | 60 | 240 |
| A | Correct-state check (all actions correct) | 5484 | 58863 | 6818423 | 0 | 240 |
| B | Direct answer | 240 | 1440 | 92754 | 0 | 240 |
| B | Gym memory | 516 | 5658 | 365853 | 0 | 240 |
| B | Full execution in one reply | 240 | 639972 | 804858 | 155 | 240 |
| B | Correct-state check (all actions correct) | 7341 | 80632 | 7293432 | 0 | 240 |

### Hint-study costs

| Suite | Condition | Interface | Calls | Prompt tokens | Completion tokens | Total tokens | Length-limited | Inference seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | H0 | Gym memory | 932 | 624630 | 10036 | 634666 | 0 | 116.2 |
| original | H0 | Correct-state check (all actions correct) | 3860 | 2827064 | 40485 | 2867549 | 0 | 33.4 |
| original | H1 | Gym memory | 934 | 654033 | 9879 | 663912 | 0 | 76.3 |
| original | H1 | Correct-state check (all actions correct) | 3860 | 3062728 | 38224 | 3100952 | 0 | 34.6 |
| original | H2 | Gym memory | 925 | 619840 | 9969 | 629809 | 0 | 76.3 |
| original | H2 | Correct-state check (all actions correct) | 3860 | 2827064 | 40420 | 2867484 | 0 | 33.2 |
| original | H3 | Gym memory | 934 | 654033 | 9879 | 663912 | 0 | 76.8 |
| original | H3 | Correct-state check (all actions correct) | 3860 | 3062728 | 38213 | 3100941 | 0 | 34.9 |
| original | H4 | Gym memory | 1021 | 729410 | 10781 | 740191 | 0 | 82.1 |
| original | H4 | Correct-state check (all actions correct) | 3860 | 3109048 | 38342 | 3147390 | 0 | 35.6 |
| original | H5 | Gym memory | 1146 | 850369 | 12157 | 862526 | 0 | 93.8 |
| original | H5 | Correct-state check (all actions correct) | 3860 | 3144512 | 38481 | 3182993 | 0 | 36.3 |
| cuefree | H0 | Gym memory | 991 | 684260 | 10585 | 694845 | 0 | 122.3 |
| cuefree | H0 | Correct-state check (all actions correct) | 3632 | 2663504 | 37946 | 2701450 | 0 | 31.7 |
| cuefree | H1 | Gym memory | 1039 | 774949 | 11028 | 785977 | 0 | 84.7 |
| cuefree | H1 | Correct-state check (all actions correct) | 3632 | 2885424 | 35412 | 2920836 | 0 | 31.7 |
| cuefree | H2 | Gym memory | 977 | 675234 | 10435 | 685669 | 0 | 78.5 |
| cuefree | H2 | Correct-state check (all actions correct) | 3632 | 2663504 | 37961 | 2701465 | 0 | 31.0 |
| cuefree | H3 | Gym memory | 1048 | 785647 | 11100 | 796747 | 0 | 84.0 |
| cuefree | H3 | Correct-state check (all actions correct) | 3632 | 2885424 | 35442 | 2920866 | 0 | 33.0 |
| cuefree | H4 | Gym memory | 1139 | 860337 | 12059 | 872396 | 0 | 92.8 |
| cuefree | H4 | Correct-state check (all actions correct) | 3632 | 2929008 | 35479 | 2964487 | 0 | 35.3 |
| cuefree | H5 | Gym memory | 1248 | 963204 | 13178 | 976382 | 0 | 111.2 |
| cuefree | H5 | Correct-state check (all actions correct) | 3632 | 2962264 | 35197 | 2997461 | 0 | 36.6 |
| A | H0 | Gym memory | 711 | 435630 | 7622 | 443252 | 0 | 100.0 |
| A | H0 | Correct-state check (all actions correct) | 5484 | 6759560 | 58898 | 6818458 | 0 | 46.8 |
| A | H1 | Gym memory | 1074 | 1041928 | 11371 | 1053299 | 0 | 89.8 |
| A | H1 | Correct-state check (all actions correct) | 5484 | 7100424 | 56177 | 7156601 | 0 | 48.3 |
| A | H2 | Gym memory | 726 | 443614 | 7742 | 451356 | 0 | 59.5 |
| A | H2 | Correct-state check (all actions correct) | 5484 | 6759560 | 58859 | 6818419 | 0 | 45.6 |
| A | H3 | Gym memory | 1078 | 1044508 | 11400 | 1055908 | 0 | 89.5 |
| A | H3 | Correct-state check (all actions correct) | 5484 | 7100424 | 56126 | 7156550 | 0 | 47.7 |
| A | H4 | Gym memory | 1099 | 1077272 | 11616 | 1088888 | 0 | 91.0 |
| A | H4 | Correct-state check (all actions correct) | 5484 | 7166232 | 55669 | 7221901 | 0 | 50.3 |
| A | H5 | Gym memory | 1179 | 1147292 | 12430 | 1159722 | 0 | 99.4 |
| A | H5 | Correct-state check (all actions correct) | 5484 | 7217824 | 55019 | 7272843 | 0 | 52.0 |
| B | H0 | Gym memory | 516 | 360195 | 5658 | 365853 | 0 | 82.9 |
| B | H0 | Correct-state check (all actions correct) | 7341 | 7212800 | 80641 | 7293441 | 0 | 53.2 |
| B | H1 | Gym memory | 631 | 490731 | 6893 | 497624 | 0 | 53.8 |
| B | H1 | Correct-state check (all actions correct) | 7341 | 7657846 | 76376 | 7734222 | 0 | 54.8 |
| B | H2 | Gym memory | 516 | 360195 | 5658 | 365853 | 0 | 43.9 |
| B | H2 | Correct-state check (all actions correct) | 7341 | 7212800 | 80625 | 7293425 | 0 | 52.9 |
| B | H3 | Gym memory | 631 | 490731 | 6893 | 497624 | 0 | 53.7 |
| B | H3 | Correct-state check (all actions correct) | 7341 | 7657846 | 76363 | 7734209 | 0 | 55.1 |
| B | H4 | Gym memory | 661 | 527818 | 7012 | 534830 | 0 | 54.6 |
| B | H4 | Correct-state check (all actions correct) | 7341 | 7745938 | 76393 | 7822331 | 0 | 58.1 |
| B | H5 | Gym memory | 710 | 575746 | 7411 | 583157 | 0 | 57.9 |
| B | H5 | Correct-state check (all actions correct) | 7341 | 7815479 | 75200 | 7890679 | 0 | 61.3 |

### Evaluation wall time per condition

| Suite | H0 | H1 | H2 | H3 | H4 | H5 |
| --- | --- | --- | --- | --- | --- | --- |
| original | 252.5s | 233.9s | 214.9s | 217.6s | 223.6s | 224.7s |
| cuefree | 249.5s | 217.3s | 203.8s | 229.2s | 229.7s | 238.8s |
| A | 253.7s | 241.6s | 214.7s | 247.5s | 236.5s | 282.8s |
| B | 241.3s | 217.3s | 206.7s | 223.0s | 222.6s | 223.4s |

Inference seconds are batch wall time apportioned over independent prompts, then summed; evaluation wall time includes grading and durable snapshots, not cluster startup, dependencies, model loading or preflight. Four suites run concurrently on one B200 per job. Early failed runs can cost less than successful runs simply because they stop sooner. Token totals include every guard call.

## Reproducibility and interpretation limits

Base Qwen/Qwen3-14B at revision 40c069824f4251a91eefaf281ebe4c544efd3e18; BF16, temperature 0, top-p 1, top-k -1, seed 17, thinking off, LoRA disabled. Direct cap 128, full trace 4,096, actions 128, context 16,384, at most 512 calls including retries. Independent direct and correct-state batches have at most 32 prompts; teacher batches stay within a case, matching the baseline harness. Live and full-trace calls are serial. Runtime package versions, hardware, source hashes, immutable inputs, prompts, outputs, grades, token use and timing are retained.

One seed, correlated variants, selected bounded grammars, and structural changes across suites limit generalization. The shortcut result excludes deterministic count-only decisions on these matched pairs, not structural/order-sensitive shortcuts. B is a rejection-sampled linear-CFG subset, not arbitrary CFG membership. Offloading measures the coupled model-plus-harness system; it does not by itself show that the model learned exhaustion, comparison, search or backtracking. The strong next_untried hint delegates selection to code. No fine-tuning or RL was run in this study. Existing whole-run/cue-free results, prompts and data were not rewritten.

[Frozen hint protocol](../experiments/offloading/protocol.json), [count-preserving instances and gate](../experiments/countpreserving/manifest.json), [full shortcut cells](../experiments/countpreserving/shortcuts.json), [baseline summary](results/countpreserving_offloading/baseline_summary.json), [hint summary](results/countpreserving_offloading/summary.json), [operator substitutions](results/countpreserving_offloading/operators.json), [guard outcomes](results/countpreserving_offloading/guards.json), [costs](results/countpreserving_offloading/costs.json), [runtime and deployment metadata](results/countpreserving_offloading/metadata.json), [paired comparisons](results/countpreserving_offloading/comparisons.json), [raw calls and instances](results/countpreserving_offloading/audit.json.gz) accompany this report.
