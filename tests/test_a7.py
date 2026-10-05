from collections import Counter
from copy import deepcopy
import json
import pytest
from abstractgym.a7 import training_pool, examples, VARIANTS, shifted_suite, restore, consequences
from abstractgym.benchmark import build_suite
from abstractgym.complexity import build_complexity_suite
from abstractgym.controller import ControllerEnv, teacher_trajectory
from abstractgym.experiment import run_case
from abstractgym.batched_experiment import evaluate_batched

@pytest.fixture(scope='module')
def pool(): return training_pool()

def test_training_split_and_transition_audit(pool):
    rows,_,_=build_suite();train={r['task_id']:r for r in rows if r['split']=='train'}
    assert len(pool)==512 and set(Counter(s['trajectory'] for s in pool).values())=={8}
    for s in pool:
        row=train[s['trajectory']];env=ControllerEnv(row)
        for t in teacher_trajectory(row)[:s['position']]:env.step(t['action'])
        assert env.observation()==s['observation']
        assert env.step(s['action'])==s['successor']
    for variant in VARIANTS:
        data=examples(pool,variant);assert len(data)==1024
        for s,e in zip(pool,data[::2]):assert json.loads(e['answer'])==s['action']

def test_counterfactuals_have_distinct_legal_children(pool):
    compared=0
    for s in pool:
        if s['action']['op']!='TRY':continue
        env=restore(s['observation']);branches=s['evidence']['branches']
        if len(branches)==2:compared+=1
        assert len({b['production'] for b in branches})==len(branches)
        for b in branches:
            form=env.stack[-1]['form'];i=next(i for i,x in enumerate(form) if x in env.grammar.nonterminals)
            rule=env.grammar.production_by_id(b['production'])
            assert rule.lhs==form[i] and b['production'] not in env.stack[-1]['tried']
            assert b['child_form']==form[:i]+list(rule.rhs)+form[i+1:]
            branch=deepcopy(env);branch.stack[-1]['tried'].append(b['production']);branch.stack.append(dict(form=b['child_form'],tried=[],phase='entered'));branch.steps+=1
            for action in b['continuation']:branch.step(action)
            assert branch.observation()==b['state']
            assert len(b['continuation'])<=3
    assert compared>0

def test_shift_is_new_and_paired():
    old=build_complexity_suite();new=shifted_suite()
    assert len(new)==192 and not {r['task_id'] for r in old}&{r['task_id'] for r in new}
    assert set(Counter(r['pair_id'] for r in new).values())=={2}
    assert {r['difficulty']['level'] for r in new}=={3,6,12,24}
    assert all(r['grammar']['start']=='Root' for r in new)
    for r in new:teacher_trajectory(r)

def test_batched_grader_matches_legacy_success_and_errors():
    rows=build_complexity_suite(levels=(1,),variants=1)
    class Adapter:
        def __call__(self,prompt):
            obs=json.loads(prompt.split('Observation:\n')[1]);env=restore(obs)
            if obs['input'][0]=='d': return {'text':'bad json'}
            if 'Return one JSON array' in prompt:
                actions=[]
                while env.outcome is None:
                    action=env.reference_action();actions.append(action);env.step(action)
                return {'text':json.dumps(actions)}
            return {'text':json.dumps(env.reference_action())}
        def generate_batch(self,prompts):return [self(p) for p in prompts]
    adapter=Adapter();records=evaluate_batched(rows,adapter)
    for r in records:
        row=next(x for x in rows if x['task_id']==r['task_id']);legacy=run_case(row,r['mode'],adapter)
        for key in ('success','failure','first_error','outcome','correct_actions','target_actions','model_calls'):
            assert legacy[key]==r[key]
