import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_countpreserving import direct_bias, operator_stats, summarize
from abstractgym.controller import teacher_trajectory
from run_ray_countpreserving import verify_inputs


def test_bias_keeps_malformed_responses_in_all_case_denominator():
    texts = ['{"accept":true}', '{"accept":false}', '{"accept":true}',
        'not JSON', '{"accept":1}', '{"accept":true,"extra":0}']
    result = direct_bias([dict(text=text) for text in texts])
    assert result['cases'] == 6 and result['yes'] == 2 and result['no'] == 1
    assert result['malformed'] == 3
    assert result['yes_rate_all_cases'] == 2 / 6
    assert result['yes_rate_valid_replies'] == 2 / 3


def test_operator_audit_counts_repeated_try_separately_and_never_drops_parse_errors():
    suites, _ = verify_inputs()
    row = suites['A'][1]
    gold = teacher_trajectory(row)
    calls = []
    for step in gold:
        action = step['action']
        if action['op'] == 'DONE':
            action = {'op': 'TRY', 'production': step['observation']['stack'][-1]['tried'][0]}
        calls.append(dict(text=json.dumps(action)))
    record = dict(task_id=row['task_id'], mode='oracle_step', calls=calls)
    result = operator_stats([row], [record])
    assert list(result)[0] == 'DONE'
    assert result['DONE'] == dict(correct=0, total=1, mistakes={'TRY_already_tried': 1})
    calls[0]['text'] = 'bad JSON'
    result = operator_stats([row], [record])
    assert result['TRY']['mistakes'] == {'malformed_or_unknown': 1}
    assert sum(op['total'] for op in result.values()) == len(gold)
    with pytest.raises(ValueError, match='incomplete teacher'):
        operator_stats([row], [dict(record, calls=calls[:-1])])


def test_random_suite_summary_uses_its_actual_family():
    suites, _ = verify_inputs()
    records = [dict(row, mode='direct', success=row['target_answer'] == 'accept', text='{"accept":true}')
        for row in suites['B']]
    result = summarize(records)
    assert set(result['by_family']) == {'random_multi_rule'}
    assert result['overall']['correct'] == 120 and result['overall']['total'] == 240
    assert result['overall']['reply_bias']['yes_rate_all_cases'] == 1
    assert all(cell['total'] == 48 for cell in result['by_level'].values())
