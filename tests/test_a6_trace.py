import json
from abstractgym.brackets import suite
from abstractgym.a6_trace import reference,injected_prefix,score,trace_prompt


def test_full_trace_prompt_specifies_every_action_schema():
    text=trace_prompt(suite()[0])
    for action in ('{"op":"PUSH","value":"token"}', '{"op":"CENTER"}', '{"op":"POP"}', '{"op":"ACCEPT"}', '{"op":"REJECT"}'):
        assert action in text
    assert 'Return one JSON array containing all actions' in text


def test_trace_scoring_preserves_forced_errors_without_crediting_them():
    for row in suite():
        gold=reference(row)
        assert score(json.dumps(gold),row)['success']
        assert not score(json.dumps(gold[:-1]),row)['success']
        assert not score(json.dumps(gold+[{'op':'ACCEPT'}]),row)['success']
        if row['depth']>=4:
            for k in (0,1,2,4):
                prefix=injected_prefix(row,k)
                forced=json.loads(prefix[:-1]+']')
                assert sum(a!=b for a,b in zip(forced,gold))==k
                result=score(json.dumps(gold[4:])[1:],row,prefix,True)
                assert result['success'] and result['target_actions']==len(gold)-4
                assert result['actions'][:4]==forced
