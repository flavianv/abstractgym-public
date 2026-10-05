import gzip
import importlib.util
import json
from pathlib import Path
import pytest
from abstractgym.a6_followup import prompt,choices,parse_action,oracle_states
from abstractgym.a6_raw import observation,parse
from abstractgym.brackets import expected,advance,suite

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('export_followups',ROOT/'scripts/export_a6_followups.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def artifacts(name):
    root=ROOT/'docs/results'/('2026-10-04-'+name)
    return json.loads((root/'summary.json').read_text()),json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()))


@pytest.mark.parametrize('name',['x1','x6'])
def test_live_followup_records_replay_without_repair(name):
    summary,audit=artifacts(name);rows=audit['instances'];lookup={r['id']:r for r in rows}
    assert rows==[r for r in suite() if r['split']=='test']
    runs=audit['conditions'].items() if name=='x1' else [(r['seed'],r['records']) for r in audit['results']]
    for (condition,records),reported in zip(runs,summary):
        for key,value in module.execution_summary(records,rows).items():assert reported[key]==value
        for record in records:
            row=lookup[record['id']];pos,mode,stack=0,'opening',[]
            for i,call in enumerate(record['calls']):
                gold=expected(row,pos,mode,stack);assert call['expected']==gold
                if name=='x1':
                    assert call['model_prompt']==prompt(row,pos,mode,stack,condition)
                    assert call['text'] in choices(row,condition)
                    assert call['action']==parse_action(call['text'],row,condition)
                else:
                    assert call['input_ids']==observation(row,pos,mode,stack)
                    assert call['action']==parse(call['output_ids'])
                assert call['correct']==(call['action']==gold)
                if not call['correct'] or gold['op'] in ('ACCEPT','REJECT'):
                    assert i==len(record['calls'])-1
                    assert record['success']==call['correct'];break
                pos,mode,stack=advance(call['action'],pos,mode,stack)
            else:raise AssertionError('unfinished execution')


def test_x1_reported_counts():
    summary,_=artifacts('x1')
    for r,total,actions,letter,bracket in zip(summary,[82,79],[[962,972],[965,978]],[[2,4,8,8,8,8,8],[2,4,8,8,8,7,7]],[[2,4,7,7,6,5,5],[2,4,6,7,6,5,5]]):
        assert r['held_out']['total']==[total,92] and r['action_accuracy']==actions
        assert r['first_error_median']==10 and r['push_value_errors']==0
        assert [v[0] for v in r['held_out']['letters']['by_depth'].values()]==letter
        assert [v[0] for v in r['held_out']['brackets']['by_depth'].values()]==bracket
        assert r['label_blind']['always_accept']==r['label_blind']['always_reject']==[46,92]


def test_x2_reported_counts_and_independent_teacher_states():
    summary,audit=artifacts('x2')
    rows={r['id']:r for r in suite() if r['split']=='test'}
    states={(s['id'],s['call']):s for row in rows.values() for s in oracle_states(row)}
    for r,overall,letters,brackets,later in zip(summary,[[881,988],[587,988]],[[494,494],[474,494]],[[387,494],[113,494]],[[150,223],[110,446]]):
        records=audit['conditions'][r['condition']]
        for record in records:
            state=states[record['id'],record['call']];row=rows[record['id']]
            assert all(record[k]==state[k] for k in ('pos','mode','stack','expected'))
            assert record['model_prompt']==prompt(row,state['pos'],state['mode'],state['stack'],'free_json')
            assert record['action']==parse_action(record['text'],row)
            assert record['correct']==(record['action']==state['expected'])
        assert module.teacher_summary(records,audit['instances'])=={k:v for k,v in r.items() if k!='condition'}
        assert r['action_accuracy']==overall
        assert r['letters']['action_accuracy']==letters and r['brackets']['action_accuracy']==brackets
        assert r['brackets']['after_call1_failure']==later
    assert [v[0] for v in summary[0]['brackets']['by_action'].values()]==[174,46,126,23,18]
    assert [v[0] for v in summary[1]['brackets']['by_action'].values()]==[0,46,22,23,22]


def test_x6_training_fit_and_raw_domain_counts():
    summary,audit=artifacts('x6')
    for r,a,letter,bracket in zip(summary,audit['results'],[15,23,20],[12,23,23]):
        assert r['held_out']['total']==[0,92] and r['first_error_median']==1
        assert r['training']['train_accuracy']==[36,36] and r['training']['parameters']==81004
        assert r['exhaustive']=={'uvwxy':[36,36],'abcde':[letter,36],'([)]#':[bracket,36]}
        for alphabet,counts in r['exhaustive'].items():
            cases=[x for x in a['exhaustive'] if x['alphabet']==alphabet]
            assert [sum(parse(x['output_ids'])==x['expected'] for x in cases),len(cases)]==counts


def test_x1_residual_errors_are_decisions_not_wrong_symbols():
    from collections import Counter
    _,audit=artifacts('x1')
    for name,want in [('constrained_json',{('POP','REJECT'):5,('REJECT','POP'):5}),('role_id',{('POP','REJECT'):5,('REJECT','POP'):8})]:
        errors=Counter((c['expected']['op'],c['action']['op']) for r in audit['conditions'][name] for c in r['calls'] if not c['correct'])
        assert errors==want
