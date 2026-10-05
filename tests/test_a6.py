import itertools
from abstractgym.a6 import (dev_data,exhaustive_rows,evaluate_execution,summarize,extra_rows,
                            abstract_state,role_map,encode_step,decode_action,action_id)
from abstractgym.brackets import suite


def test_training_coverage_is_from_dev_traces_only():
    rows,steps,unique=dev_data()
    assert len(rows)==367 and len(steps)==2305 and len(unique)==36
    assert all(set(r['input'])<=set('uvwxy') for r in rows)
    assert unique==dict(exhaustive_rows())
    # All state labels arose in an actual reference trajectory; no test-row labels.
    assert set(tuple(s['state']) for s in steps)==set(unique)


def test_abstract_harness_matches_frozen_reference_on_all_cases():
    rows=[r for r in suite() if r['split']=='test']
    policy=dict(exhaustive_rows())
    records=evaluate_execution(rows,lambda keys:[policy[k] for k in keys])
    score=summarize(rows,records)
    assert score['held_out']['total']==[92,92]
    assert score['label_blind']['always_accept']==[46,92]
    assert score['label_blind']['always_reject']==[46,92]
    for r in rows:
        for i in range(6):assert action_id(decode_action(i,r),r)==i


def test_first_wrong_action_stops_without_repair():
    row=next(r for r in suite() if r['split']=='test' and r['depth']==2 and r['accept'])
    records=evaluate_execution([row],lambda keys:[5]*len(keys))
    assert records[0]['first_error']==1
    assert len(records[0]['calls'])==1 and not records[0]['success']


def test_symbol_abstraction_is_renaming_equivariant():
    row=next(r for r in suite() if r['alphabet']=='abcde' and r['depth']==2 and r['accept'])
    mapping=dict(zip('abcde',['alpha','beta','gamma','delta','center']))
    renamed=dict(row,input=[mapping[t] for t in row['input']],center=mapping[row['center']],pairs={mapping[k]:mapping[v] for k,v in row['pairs'].items()})
    for pos in range(len(row['input'])+1):
        for phase in ('opening','closing'):
            for stack in ([],list(row['pairs'].values())):
                assert abstract_state(row,pos,phase,stack)==abstract_state(renamed,pos,phase,[mapping[t] for t in stack])


def test_extra_cases_do_not_replace_frozen_cases():
    extra=extra_rows();frozen=suite()
    assert len(extra)==32 and len(frozen)==138
    assert not {r['id'] for r in extra}&{r['id'] for r in frozen}
    assert {r['depth'] for r in extra}=={32,64}
