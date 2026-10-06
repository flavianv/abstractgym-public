import gzip
import json
from pathlib import Path

import pytest

from abstractgym.controller import ControllerEnv, HINT_FIELDS, teacher_trajectory
from abstractgym.experiment import HINT_SENTENCE, prompt_for, run_case


def row():
    rows = json.loads(Path('experiments/countpreserving/A.json').read_text())
    return next(r for r in rows if r['family'] == 'chain_depth'
                and r['difficulty']['level'] == 1 and r['target_answer'] == 'reject')


def test_disabled_hints_preserve_every_published_zero_shot_prompt():
    audit = json.loads(gzip.decompress(Path('docs/results/wholerun/audit.json.gz').read_bytes()))
    checked = 0
    for suite in ('original', 'cuefree'):
        for record in audit['runs'][suite]['zeroshot']:
            if record['mode'] == 'direct':
                continue
            for call in record['calls']:
                observation = json.loads(call['prompt'].split('\nObservation:\n')[1])
                assert prompt_for(observation, record['mode']) == call['prompt']
                checked += 1
    assert checked > 5000
    env = ControllerEnv(row())
    while env.outcome is None:
        plain = env.observation()
        for off in (None, False, (), []):
            assert env.observation(hints=off) == plain
            assert prompt_for(env.observation(hints=off), 'external_state') == prompt_for(plain, 'external_state')
        env.step(env.reference_action())


def test_hints_are_current_frame_facts_and_do_not_mutate_policy():
    env = ControllerEnv(row())
    while env.outcome is None:
        action = env.reference_action()
        plain = env.observation()
        hinted = env.observation(hints=True)
        current = hinted['stack'][-1]
        assert all(not set(HINT_FIELDS) & set(f) for f in hinted['stack'][:-1])
        form = current['form']
        nt = next((s for s in form if s in env.grammar.nonterminals), None)
        expected = [env.grammar.production_id(p) for p in env.grammar.productions_for(nt)
                    if env.grammar.production_id(p) not in current['tried']] if nt else []
        prefix = form[:next((i for i, s in enumerate(form) if s in env.grammar.nonterminals), len(form))]
        assert current['untried_rules'] == expected
        assert current['next_untried'] == (expected[0] if expected else None)
        assert current['prefix_matches'] == (prefix == list(env.tokens[:len(prefix)]))
        assert current['complete_match'] == (nt is None and form == list(env.tokens))
        assert prompt_for(hinted, 'oracle_step').count(HINT_SENTENCE) == 1
        current['tried'].append('corruption')
        assert env.observation() == plain
        assert env.reference_action() == action
        env.step(action)
    assert env.observation(hints=True) == env.observation()
    assert [s['action'] for s in teacher_trajectory(row(), hints=True)] == [s['action'] for s in teacher_trajectory(row())]


def test_subset_unknown_and_comparison_before_cutoff():
    env = ControllerEnv(row())
    hinted = env.observation(hints='untried_rules')['stack'][-1]
    assert set(hinted) & set(HINT_FIELDS) == {'untried_rules'}
    with pytest.raises(ValueError, match='unknown hint'):
        env.observation(hints=['correct_action'])
    env.stack = [{'form': [env.tokens[0], env.grammar.start, *env.tokens], 'phase': 'entered', 'tried': []}]
    assert env.observation(hints=True)['stack'][-1]['prefix_matches'] is True
    env.stack[-1]['form'] = list(env.tokens)
    assert env.observation(hints=True)['stack'][-1]['complete_match'] is True
    env.stack[-1]['form'] = list(reversed(env.tokens))
    env.max_depth = -1
    assert env.observation(hints=True)['stack'][-1]['prefix_matches'] is False


def scripted(answers):
    remaining = iter(answers)
    def adapter(prompt):
        return {'text': json.dumps(next(remaining)), 'usage': {}}
    return adapter


def repeated_answers(second_correct=True):
    gold = teacher_trajectory(row())
    index = next(i for i, s in enumerate(gold) if s['action']['op'] == 'DONE')
    repeated = {'op': 'TRY', 'production': gold[index]['observation']['stack'][-1]['tried'][0]}
    actions = [s['action'] for s in gold]
    return index, actions[:index] + [repeated, actions[index] if second_correct else repeated] + actions[index+1:]


def test_guard_reasks_once_and_keeps_grading_and_budget():
    index, answers = repeated_answers()
    result = run_case(row(), 'external_state', scripted(answers), legality_guard=True, max_calls=512)
    assert result['success']
    assert result['guard'] == dict(activations=1, reasks=1, second_correct=1, budget_blocked=0)
    assert result['model_calls'] == result['target_actions'] + 1
    assert result['calls'][index]['guard_rejected'] is True
    assert result['calls'][index+1]['prompt'] == result['calls'][index]['prompt'] + '\n' + answers[index]['production'] + ' was already tried here'
    result = run_case(row(), 'external_state', scripted(answers), legality_guard=True, max_calls=index+1)
    assert result['failure'] == 'call_budget_exhausted'
    assert result['model_calls'] == index+1
    assert result['guard'] == dict(activations=1, reasks=0, second_correct=0, budget_blocked=1)


def test_guard_second_answer_is_final_and_other_errors_are_not_repaired():
    index, answers = repeated_answers(False)
    result = run_case(row(), 'external_state', scripted(answers), legality_guard=True)
    assert not result['success'] and result['failure'] == 'invalid_action'
    assert result['model_calls'] == index+2
    assert result['guard']['activations'] == result['guard']['reasks'] == 1
    assert result['guard']['second_correct'] == 0
    for bad in ({'op': 'DONE'}, {'op': 'TRY', 'production': 'p999'}, {'op': 'TRY', 'production': []}):
        result = run_case(row(), 'external_state', scripted([bad]), legality_guard=True)
        assert result['failure'] == 'invalid_action' and result['model_calls'] == 1
        assert result['guard']['activations'] == 0


def test_guard_is_live_only_and_hint_reference_control_completes():
    gold = teacher_trajectory(row())
    actions = [s['action'] for s in gold]
    result = run_case(row(), 'oracle_step', scripted(actions), legality_guard=True)
    assert result['success'] and 'guard' not in result
    for mode in ('oracle_step', 'external_state', 'full_trace'):
        assert run_case(row(), mode, reference=True, hints=True, legality_guard=True)['success']


def test_guard_decision_does_not_consult_reference_action(monkeypatch):
    index, answers = repeated_answers()
    original = ControllerEnv.reference_action
    references = []
    def reference(env):
        references.append(env.steps)
        return original(env)
    monkeypatch.setattr(ControllerEnv, 'reference_action', reference)
    # Teacher construction occurs before calls; only compare the rejected call
    # with its re-ask, before the second answer reaches the normal validator.
    counts = []
    remaining = iter(answers)
    def adapter(prompt):
        counts.append(len(references))
        return {'text': json.dumps(next(remaining)), 'usage': {}}
    result = run_case(row(), 'external_state', adapter, legality_guard=True)
    assert result['success']
    assert counts[index] == counts[index+1]
