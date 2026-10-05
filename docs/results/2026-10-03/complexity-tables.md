# Qwen3-14B increasing complexity

Strict completed cases / total. Each grammar cell contains four accept/reject pairs.
Levels are family-specific parameters, not a common difficulty unit.

## full_trace

| Family / condition | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| terminal_length / original | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 |
| terminal_length / clarified | 4/8 | 4/8 | 4/8 | 1/8 | 0/8 |
| terminal_length / clarified_fewshot | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 |
| terminal_length / one_rule_specialized | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| chain_depth / original | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| chain_depth / clarified | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 |
| chain_depth / clarified_fewshot | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| alternatives / original | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| alternatives / clarified | 4/8 | 1/8 | 0/8 | 0/8 | 0/8 |
| alternatives / clarified_fewshot | 4/8 | 4/8 | 3/8 | 2/8 | 0/8 |
| sequence / original | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| sequence / clarified | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| sequence / clarified_fewshot | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| recursion / original | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| recursion / clarified | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| recursion / clarified_fewshot | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| nesting / original | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| nesting / clarified | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| nesting / clarified_fewshot | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |

## external_state

| Family / condition | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| terminal_length / original | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 |
| terminal_length / clarified | 4/8 | 8/8 | 8/8 | 6/8 | 4/8 |
| terminal_length / clarified_fewshot | 6/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| terminal_length / one_rule_specialized | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 |
| chain_depth / original | 0/8 | 4/8 | 0/8 | 0/8 | 0/8 |
| chain_depth / clarified | 0/8 | 4/8 | 4/8 | 0/8 | 0/8 |
| chain_depth / clarified_fewshot | 0/8 | 4/8 | 4/8 | 0/8 | 0/8 |
| alternatives / original | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| alternatives / clarified | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| alternatives / clarified_fewshot | 6/8 | 1/8 | 0/8 | 0/8 | 0/8 |
| sequence / original | 0/8 | 4/8 | 4/8 | 0/8 | 4/8 |
| sequence / clarified | 4/8 | 4/8 | 4/8 | 4/8 | 2/8 |
| sequence / clarified_fewshot | 0/8 | 0/8 | 4/8 | 4/8 | 4/8 |
| recursion / original | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| recursion / clarified | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| recursion / clarified_fewshot | 3/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| nesting / original | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| nesting / clarified | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| nesting / clarified_fewshot | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |

## oracle_step

| Family / condition | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| terminal_length / original | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 |
| terminal_length / clarified | 4/8 | 8/8 | 8/8 | 7/8 | 4/8 |
| terminal_length / clarified_fewshot | 6/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| terminal_length / one_rule_specialized | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 |
| chain_depth / original | 0/8 | 4/8 | 0/8 | 0/8 | 0/8 |
| chain_depth / clarified | 0/8 | 4/8 | 4/8 | 0/8 | 0/8 |
| chain_depth / clarified_fewshot | 0/8 | 4/8 | 4/8 | 0/8 | 0/8 |
| alternatives / original | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| alternatives / clarified | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| alternatives / clarified_fewshot | 6/8 | 1/8 | 0/8 | 0/8 | 0/8 |
| sequence / original | 0/8 | 4/8 | 4/8 | 0/8 | 4/8 |
| sequence / clarified | 4/8 | 4/8 | 4/8 | 4/8 | 2/8 |
| sequence / clarified_fewshot | 0/8 | 0/8 | 4/8 | 4/8 | 4/8 |
| recursion / original | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| recursion / clarified | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| recursion / clarified_fewshot | 3/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| nesting / original | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| nesting / clarified | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| nesting / clarified_fewshot | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |

## Direct membership

| Family | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| alternatives | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| chain_depth | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| nesting | 7/8 | 5/8 | 8/8 | 8/8 | 8/8 |
| paired_brackets | — | 4/8 | 4/8 | 4/8 | 4/8 |
| recursion | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| sequence | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| terminal_length | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
