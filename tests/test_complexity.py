from collections import defaultdict
from abstractgym.complexity import build_complexity_suite, build_primitive_ladder, clarify_prompt
from abstractgym.controller import ControllerEnv, teacher_trajectory
from abstractgym.experiment import prompt_for


def test_complexity_labels_pairs_and_reference_budget():
    rows=build_complexity_suite()
    assert len(rows)==240
    pairs=defaultdict(list)
    for row in rows:
        pairs[row['pair_id']].append(row)
        env=ControllerEnv(row)
        for step in teacher_trajectory(row):env.step(step['action'])
        assert env.outcome==row['target_answer']
        assert row['difficulty']['search_steps']<512
        p=prompt_for(ControllerEnv(row).observation(),'full_trace')
        assert clarify_prompt(p).split('Observation:\n')[1]==p.split('Observation:\n')[1]
    for pair in pairs.values():
        assert pair[0]['input']==pair[1]['input']
        assert {r['target_answer'] for r in pair}=={'accept','reject'}


def test_primitive_ladder_answers():
    tasks=build_primitive_ladder()
    assert len({t['id'] for t in tasks})==len(tasks)
    assert {t['level'] for t in tasks}=={1,2,4,8,16,32}
    assert all(isinstance(t['expected'],dict) for t in tasks)


def test_balanced_membership_pairs_have_same_terminal_counts():
    from collections import Counter
    from abstractgym.complexity import build_balanced_membership_suite
    rows=build_balanced_membership_suite()
    assert len(rows)==32
    for pos,neg in zip(rows[::2],rows[1::2]):
        assert pos['input']==neg['input']
        assert pos['target_answer']=='accept' and neg['target_answer']=='reject'
        inventories=[]
        for row in (pos,neg):
            terminals=set(row['grammar']['terminals'])
            counts=Counter(x for rule in row['grammar']['productions'] for x in rule['rhs'] if x in terminals)
            inventories.append(counts)
            assert set(row['input'])==terminals
        assert inventories[0]==inventories[1]


def test_posthoc_witness_check_distinguishes_derivation_from_canonical_trace():
    import importlib.util
    import json
    from pathlib import Path
    from abstractgym.experiment import run_case
    spec=importlib.util.spec_from_file_location('complexity_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_complexity.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    row=next(r for r in build_complexity_suite(levels=(2,),variants=1)
             if r['family']=='recursion' and r['target_answer']=='accept')
    actions=[{'op':'TRY','production':'p0'},{'op':'TRY','production':'p1'},{'op':'ACC'}]
    text=json.dumps(actions)
    assert module.simple_accepting_derivation(row,text)
    assert not run_case(row,'full_trace',lambda prompt:{'text':text})['success']
    assert not module.simple_accepting_derivation(row,json.dumps(actions[1:]))
    assert not module.simple_accepting_derivation(row,json.dumps(actions+[{'op':'ACC'}]))
    assert not module.simple_accepting_derivation(row,'not JSON')
