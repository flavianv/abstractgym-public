import json
from abstractgym.a6_followup import choices,legal_actions,prompt,parse_action,oracle_states
from abstractgym.brackets import suite,stack_prompt


def test_constraint_domain_does_not_give_away_correct_transition():
    for row in suite():
        actions=legal_actions(row)
        assert len(actions)==6
        for state in oracle_states(row):
            assert state['expected'] in actions
            assert len([a for a in actions if a!=state['expected']])==5
            args=(row,state['pos'],state['mode'],state['stack'])
            assert prompt(*args,'constrained_json')==stack_prompt(*args)
            assert json.loads(prompt(*args,'role_id').rsplit('\n',1)[1])==json.loads(stack_prompt(*args).rsplit('\n',1)[1])
        for interface in ('constrained_json','role_id'):
            assert [parse_action(c,row,interface) for c in choices(row,interface)]==actions
    assert parse_action('PUSH_7',suite()[0],'role_id') is None


def test_teacher_states_cover_the_complete_frozen_rollouts():
    rows=[r for r in suite() if r['split']=='test']
    states=[s for r in rows for s in oracle_states(r)]
    assert len(states)==988
    for row in rows:
        trace=oracle_states(row)
        assert trace[-1]['expected']=={'op':'ACCEPT' if row['accept'] else 'REJECT'}
        assert all(s['expected']['op'] not in ('ACCEPT','REJECT') for s in trace[:-1])
