from copy import deepcopy
import json

import pytest

from abstractgym.controller import teacher_trajectory
from abstractgym.experiment import prompt_for, run_case
from report_countpreserving_offloading import changes, guard_stats, live_errors, operator_table, replay
from run_ray_offloading import CONDITIONS, verify_study_inputs


def fixture():
    suites, _ = verify_study_inputs()
    return next(r for r in suites['A'] if r['family'] == 'chain_depth'
                and r['difficulty']['level'] == 1 and r['target_answer'] == 'reject')


def fake_native(row, mode, config):
    """Explicit reference-only response fixture, never experimental evidence."""
    gold = teacher_trajectory(row, hints=config['hints'])
    replies = []
    for step in gold:
        prompt = prompt_for(step['observation'], mode)
        if config['legality_guard'] and mode == 'external_state' and step['action']['op'] == 'DONE':
            p = step['observation']['stack'][-1]['tried'][0]
            replies.append((prompt, {'op': 'TRY', 'production': p}))
            prompt += '\n' + p + ' was already tried here'
        replies.append((prompt, step['action']))
    iterator = iter(enumerate(replies))
    def adapter(prompt):
        index, (expected, action) = next(iterator)
        assert prompt == expected
        return dict(text=json.dumps(action), model_prompt=prompt, max_tokens=128,
            batch_size=min(32, len(replies) - index // 32 * 32) if mode == 'oracle_step' else 1,
            finish_reason='stop', usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2))
    return adapter


@pytest.mark.parametrize('condition', list(CONDITIONS))
@pytest.mark.parametrize('mode', ['external_state', 'oracle_step'])
def test_replay_verifies_hints_and_guarded_calls(condition, mode):
    row, config = fixture(), CONDITIONS[condition]
    record = run_case(row, mode, fake_native(row, mode, config), max_calls=512, **config)
    record.update(condition=condition, guard_effective=config['legality_guard'] and mode == 'external_state')
    assert record['success']
    replay(row, record, config)
    corrupted = deepcopy(record)
    corrupted['calls'][0]['prompt'] += 'changed'
    with pytest.raises(AssertionError):
        replay(row, corrupted, config)
    if record.get('guard'):
        corrupted = deepcopy(record)
        corrupted['guard']['second_correct'] += 1
        with pytest.raises(AssertionError):
            replay(row, corrupted, config)
        counts = guard_stats([record])
        assert counts['second_correct'] == counts['activations'] > 0


def test_changes_match_ids_and_count_regressions():
    before = [dict(task_id='a', success=False, calls=[dict(text='bad')]),
              dict(task_id='b', success=True, calls=[dict(text='ok')])]
    after = [dict(task_id='b', success=False, calls=[dict(text='bad')]),
             dict(task_id='a', success=True, calls=[dict(text='ok')])]
    assert changes(before, after) == dict(cases=2, improved=1, regressed=1, output_changed_cases=2)
    with pytest.raises(ValueError, match='unpaired'):
        changes(before, after[:1])


def test_operator_table_keeps_zero_totals_and_done_first():
    from analyze_countpreserving import OPERATORS
    metrics = {op: dict(correct=0, total=0, mistakes={}) for op in OPERATORS}
    metrics['DONE'] = dict(correct=1, total=4, mistakes={'TRY_already_tried': 3})
    result = operator_table(metrics)
    assert result.index('| DONE |') < result.index('| TRY |')
    assert '1/4 | 25.0%' in result and 'TRY_already_tried: 3' in result
    assert '0/0 | n/a' in result


def test_live_errors_keep_wrong_reason_and_unreached_done_separate():
    row = fixture()
    gold = teacher_trajectory(row)
    replies = []
    for step in gold:
        action = step['action']
        if action['op'] == 'PRN':
            action = {'op': 'PRN', 'reason': 'term'}
        replies.append(action)
    iterator = iter(replies)
    record = run_case(row, 'external_state', lambda p: {'text': json.dumps(next(iterator)), 'usage': {}})
    errors = live_errors([row], [record])
    assert errors['failed_cases'] == 1
    assert errors['expected_operator'] == {'PRN': 1}
    assert errors['substitutions'] == {'PRN -> PRN(term)': 1}
    assert errors['final_wrong_repeated_try'] == 0
    assert errors['before_first_done'] == 1
