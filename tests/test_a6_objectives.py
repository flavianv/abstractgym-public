import json
from collections import Counter
from abstractgym.a6 import dev_data
from abstractgym.a6_objectives import membership_examples,dfs_examples
from abstractgym.benchmark import build_suite


def test_objectives_use_only_development_trajectories():
    members=membership_examples();dev=dev_data()[0]
    assert len(members)==367
    assert {r['trajectory'] for r in members}=={r['id'] for r in dev}
    assert sum(r['action']['accept'] for r in members)==31
    assert all(json.loads(r['answer'])==r['action'] for r in members)
    dfs=dfs_examples();rows=build_suite()[0]
    assert len(dfs)==11002
    assert {r['trajectory'] for r in dfs}=={r['task_id'] for r in rows if r['split']=='train'}
    assert Counter(r['action']['op'] for r in dfs)=={'TRY':3710,'BT':3614,'PRN':2618,'DONE':1028,'ACC':32}
    assert all(json.loads(r['answer'])==r['action'] for r in dfs)


def test_dfs_training_has_no_exact_complexity_cases_or_teacher_prompts():
    from abstractgym.complexity import build_complexity_suite
    from abstractgym.controller import teacher_trajectory
    from abstractgym.experiment import prompt_for
    train=[r for r in build_suite()[0] if r['split']=='train'];held=build_complexity_suite()
    key=lambda r:json.dumps({k:r[k] for k in ('grammar','input')},sort_keys=True)
    assert not {key(r) for r in train}&{key(r) for r in held}
    prompts={r['prompt'] for r in dfs_examples()}
    oracle=[prompt_for(s['observation'],'oracle_step') for r in held for s in teacher_trajectory(r)]
    assert len(oracle)==3860 and not prompts&set(oracle)


def test_dfs_training_inventory():
    from abstractgym.controller import teacher_trajectory
    rows=[r for r in build_suite()[0] if r['split']=='train']
    assert Counter(r['family'] for r in rows)=={'regular':32,'nested':32}
    assert (min(r['difficulty']['derivation_depth'] for r in rows),max(r['difficulty']['derivation_depth'] for r in rows))==(1,4)
    assert max(len(s['observation']['stack']) for r in rows for s in teacher_trajectory(r))==6
