import json
import gzip
import hashlib
from pathlib import Path

from abstractgym.a6_objectives import dfs_examples
from abstractgym.benchmark import build_suite
from abstractgym.controller import ControllerEnv, teacher_trajectory
from abstractgym.experiment import prompt_for, run_case
from abstractgym.wholerun import (fulltrace_examples, training_schedule,
    diagnostic_actions, written_diagnostic)


def test_full_examples_preserve_x7_training_and_fulltrace_prompt():
    full = fulltrace_examples()
    steps = dfs_examples()
    rows = {r['task_id']: r for r in build_suite()[0] if r['split'] == 'train'}
    assert len(full) == len(rows) == 64
    assert len(steps) == 11002
    assert {e['trajectory'] for e in steps} == set(rows)
    for e in full:
        row = rows[e['trajectory']]
        assert e['prompt'] == prompt_for(ControllerEnv(row).observation(), 'full_trace')
        actions = json.loads(e['answer'])
        assert actions == [s['action'] for s in teacher_trajectory(row)]
        env = ControllerEnv(row)
        for action in actions:
            env.step(action)
        assert env.outcome == row['target_answer']


def test_token_matched_schedule_crosses_budget_once_without_truncation():
    data = [{'answer_tokens': n} for n in (11, 17, 23, 29, 31)]
    schedule = training_schedule(data, 'fulltrace_sft_matched', target_budget=300, batch_size=2)
    counts = [sum(data[i]['answer_tokens'] for i in g['indices']) for g in schedule]
    assert sum(counts[:-1]) < 300 <= sum(counts)
    assert schedule == training_schedule(data, 'fulltrace_sft_matched', target_budget=300, batch_size=2)
    plain = training_schedule(data, 'fulltrace_sft', batch_size=2)
    assert len(plain) == 9
    assert [g['epoch'] for g in plain] == [1] * 3 + [2] * 3 + [3] * 3


def test_diagnostics_never_upgrade_malformed_grades():
    row = next(r for r in build_suite()[0] if r['split'] == 'train')
    actions = [s['action'] for s in teacher_trajectory(row)]
    text = json.dumps(actions[:-1])[:-1] + ','
    record = run_case(row, 'full_trace', lambda p: {'text': text, 'finish_reason': 'length'})
    assert not record['success'] and record['failure'] == 'call_or_parse_error'
    d = written_diagnostic(row, record)
    assert d['correct_prefix_actions'] == len(actions) - 1
    assert d['first_wrong_move'] is None and d['first_missing_move'] == len(actions) - 1
    assert diagnostic_actions('[{"op":"ACC"},]') == ([{'op': 'ACC'}], False)
    assert diagnostic_actions('[{"op":"ACC"}]junk') == ([{'op': 'ACC'}], False)
    assert diagnostic_actions('```json []```') == ([], False)


def test_written_wrong_and_one_move_are_distinguished():
    row = next(r for r in build_suite()[0] if r['split'] == 'train')
    first = teacher_trajectory(row)[0]['action']
    r = run_case(row, 'full_trace', lambda p: {'text': json.dumps([first]), 'finish_reason': 'stop'})
    d = written_diagnostic(row, r)
    assert d['category'] == 'one_move_then_stop'
    assert d['first_wrong_move'] is None and d['first_missing_move'] == 1
    r = run_case(row, 'full_trace', lambda p: {'text': '[{"op":"ACC"}]', 'finish_reason': 'stop'})
    assert written_diagnostic(row, r)['first_wrong_move'] == 0


def test_unreadable_prefix_is_not_an_observed_wrong_or_missing_move():
    row = next(r for r in build_suite()[0] if r['split'] == 'train')
    r = run_case(row, 'full_trace', lambda p: {'text': 'not JSON', 'finish_reason': 'stop'})
    d = written_diagnostic(row, r)
    assert d['first_wrong_move'] is None and d['first_missing_move'] is None
    assert d['category'] == 'malformed_invalid_or_trailing'


def test_frozen_inputs_and_no_exact_training_evaluation_overlap():
    root = Path(__file__).resolve().parents[1] / 'experiments/wholerun'
    for name, sha in json.loads((root / 'manifest.json').read_text()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    assert json.loads(gzip.decompress((root / 'single_step.json.gz').read_bytes())) == dfs_examples()
    assert json.loads((root / 'fulltrace.json').read_text()) == fulltrace_examples()
    key = lambda row: json.dumps({k: row[k] for k in ('grammar', 'input')}, sort_keys=True)
    train = {key(r) for r in build_suite()[0] if r['split'] == 'train'}
    for suite in ('original', 'cuefree'):
        rows = json.loads((root / (suite + '.json')).read_text())
        assert len(rows) == 240 and sum(r['target_answer'] == 'accept' for r in rows) == 120
        assert not train.intersection(key(r) for r in rows)


def test_trailing_action_is_identified_as_a_completion_format_failure():
    row = next(r for r in build_suite()[0] if r['split'] == 'train')
    actions = [s['action'] for s in teacher_trajectory(row)]
    record = run_case(row, 'full_trace', lambda p: {
        'text': json.dumps(actions + [{'op': 'BT'}]), 'finish_reason': 'stop'})
    d = written_diagnostic(row, record)
    assert not record['success']
    assert d['first_wrong_is_trailing'] and d['canonical_prefix_completed']
    assert d['first_wrong_move'] == len(actions) and d['first_wrong_fraction'] == 1


def test_nonobject_action_has_an_observed_wrong_move():
    row = next(r for r in build_suite()[0] if r['split'] == 'train')
    first = teacher_trajectory(row)[0]['action']
    record = run_case(row, 'full_trace', lambda p: {
        'text': json.dumps([first, 'not-an-action']), 'finish_reason': 'stop'})
    d = written_diagnostic(row, record)
    assert d['valid_array'] and d['first_wrong_move'] == 1 and d['first_missing_move'] is None
    record = run_case(row, 'full_trace', lambda p: {'text': '[{"op": []}]', 'finish_reason': 'stop'})
    assert written_diagnostic(row, record)['first_wrong_move'] == 0
