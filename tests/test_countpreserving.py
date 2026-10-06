from collections import Counter
import json
from pathlib import Path

from abstractgym.cfg import Grammar
from abstractgym.complexity import FAMILIES, LEVELS
from abstractgym.countpreserving import build_suite_a, build_suite_b, predictions, validate_pairs
from abstractgym.controller import teacher_trajectory
from abstractgym.oracle import accepts

ROOT = Path(__file__).resolve().parents[1]


def test_a_and_b_are_oracle_checked_count_invariant_bounded_pairs():
    a = build_suite_a()
    b, metadata = build_suite_b()
    assert len(a) == len(b) == 240 and metadata['retained_pairs'] == 120
    assert Counter(r['family'] for r in a) == {f: 40 for f in FAMILIES}
    for suite, rows in [('A', a), ('B', b)]:
        assert rows == json.loads((ROOT / f'experiments/countpreserving/{suite}.json').read_text())
        assert Counter(r['difficulty']['level'] for r in rows) == {n: 48 for n in LEVELS}
        for pair in validate_pairs(rows).values():
            assert predictions(pair[0]) == predictions(pair[1])
        for row in rows:
            grammar = Grammar.from_dict(row['grammar'])
            grammar.validate_v0_constraints()
            assert accepts(grammar, tuple(row['input'])) == (row['target_answer'] == 'accept')
            assert len(teacher_trajectory(row)) <= 512
            if suite == 'B':
                assert 3 <= len(grammar.nonterminals) <= 6
                assert 3 <= len(grammar.terminals) <= 4
                assert all(2 <= len(grammar.productions_for(nt)) <= 3 for nt in grammar.nonterminals)
                if row['target_answer'] == 'accept':
                    assert row['difficulty']['accepting_depth'] == row['difficulty']['level']
        if suite == 'B':
            for pair in validate_pairs(rows).values():
                yes, no = pair
                assert yes['grammar'] == no['grammar']
                different = [i for i, (x, y) in enumerate(zip(yes['input'], no['input'])) if x != y]
                assert len(different) == 2 and different[1] == different[0] + 1
                i, j = different
                assert yes['input'][i] == no['input'][j] and yes['input'][j] == no['input'][i]


def test_shortcut_gate_and_pair_grouped_classifier_have_no_leakage():
    folder = ROOT / 'experiments/countpreserving'
    assert json.loads((folder / 'gate.json').read_text())['model_calls'] == 0
    audited = json.loads((folder / 'shortcuts.json').read_text())
    for suite in audited.values():
        assert suite['passed'] and not suite['violations']
        assert len(suite['scores']) == 129
        for scores in suite['scores'].values():
            cells = [scores['overall']] + list(scores['by_family'].values()) + list(scores['by_size'].values())
            cells += [v for sizes in scores['family_size'].values() for v in sizes.values()]
            assert all(c * 2 == n for c, n in cells)
        held = []
        for fold in suite['classifier']['folds']:
            assert not set(fold['train_pairs']) & set(fold['heldout_pairs'])
            held.extend(fold['heldout_pairs'])
        assert len(held) == len(set(held)) == 120


def test_nesting_smallest_pair_balances_both_types_without_changing_grammar():
    rows = [r for r in build_suite_a() if r['family'] == 'nesting' and r['difficulty']['level'] == 1]
    for yes, no in validate_pairs(rows).values():
        assert yes['grammar'] == no['grammar']
        assert len(yes['input']) == 5 and len(set(yes['input'])) == 5
