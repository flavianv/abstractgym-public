from abstractgym.brackets import suite,expected,advance,score_answer
import json


def test_stack_matches_independent_earley_labels():
    rows=suite()
    assert len({json.dumps((r['grammar'],r['input']),sort_keys=True) for r in rows})==len(rows)
    for row in rows:
        pos,mode,stack=0,'opening',[]
        for _ in range(len(row['input'])+1):
            action=expected(row,pos,mode,stack)
            if action['op'] in ('ACCEPT','REJECT'):
                assert (action['op']=='ACCEPT')==row['accept'];break
            pos,mode,stack=advance(action,pos,mode,stack)
        else:raise AssertionError('did not terminate')


def test_witness_checks_rule_application_and_full_yield():
    for row in suite():
        if row['accept']:
            opening=row['input'][:row['depth']]
            rules=[list(row['pairs']).index(t) for t in opening]+[2]
            assert score_answer(row,json.dumps(dict(accept=True,rules=rules)),True)['verified']
            assert not score_answer(row,json.dumps(dict(accept=True,rules=rules+[2])),True)['verified']
        assert not score_answer(row,'{"accept":1}')['schema']


def test_order_negatives_preserve_token_counts():
    from collections import Counter
    rows=suite();byid={r['id']:r for r in rows}
    for row in rows:
        if row['negative']=='order_only':
            positive=byid[row['id'].replace('-False','-True')]
            assert Counter(row['input'])==Counter(positive['input'])
