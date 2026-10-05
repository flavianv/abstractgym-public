import gzip
import json
from pathlib import Path
import sys
from abstractgym.a6_trace import score,trace_prompt
from abstractgym.brackets import suite,score_answer,direct_prompt,stack_prompt,expected,advance
from abstractgym.a6_followup import oracle_states
from abstractgym.a6 import digest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from export_a6_objectives import complexity_summary
from export_a6_followups import execution_summary,teacher_summary
from export_a6_x45 import case_summary
from test_a6_report_replay import replay_complexity


def test_x10_thinking_budget_failure_accounting():
    root=ROOT/'docs/results/2026-10-04-x10';s=json.loads((root/'summary.json').read_text())
    a=json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()));meta=json.loads((root/'metadata.json').read_text())
    assert meta['plan']['preserve_prompt'] and meta['plan']['temperature']==0
    rows=[r for r in suite() if r['split']=='test'];lookup={r['id']:r for r in rows}
    assert meta['plan']['bracket_hash']==digest(rows)
    assert meta['plan']['complexity_hash']==digest(a['complexity_instances'])
    assert s['brackets']['full']==case_summary(a['brackets']['full'],rows)
    for r in a['brackets']['full']:
        assert r['model_prompt']==trace_prompt(lookup[r['id']])
        assert all(r[k]==v for k,v in score(r['text'],lookup[r['id']]).items())
    for r in a['brackets']['direct']:
        assert r['model_prompt']==direct_prompt(lookup[r['id']])
        assert r['verified']==score_answer(lookup[r['id']],r['text'])['verified']
    states={(s['id'],s['call']):s for row in rows for s in oracle_states(row)}
    def parsed(text):
        try:return json.loads(text)
        except ValueError:return None
    for c in a['brackets']['teacher']:
        state=states[c['id'],c['call']];row=lookup[c['id']]
        assert all(c[k]==state[k] for k in ('pos','mode','stack','expected'))
        assert c['model_prompt']==stack_prompt(row,state['pos'],state['mode'],state['stack'])
        assert c['action']==parsed(c['text'])
        assert c['correct']==(c['action']==state['expected'])
    for r in a['brackets']['live']:
        row=lookup[r['id']];pos,mode,stack=0,'opening',[]
        for i,c in enumerate(r['calls']):
            gold=expected(row,pos,mode,stack)
            assert c['model_prompt']==stack_prompt(row,pos,mode,stack)
            assert c['expected']==gold and c['action']==parsed(c['text'])
            assert c['correct']==(c['action']==gold)
            if not c['correct'] or gold['op'] in ('ACCEPT','REJECT'):
                assert i==len(r['calls'])-1 and r['success']==c['correct']
                assert r['first_error']==(None if c['correct'] else i+1)
                break
            pos,mode,stack=advance(c['action'],pos,mode,stack)
        else:raise AssertionError('unfinished execution')
    assert {k:v for k,v in s['brackets']['oracle'].items() if k!='usage'}==teacher_summary(a['brackets']['teacher'],rows)
    live_summary=execution_summary(a['brackets']['live'],rows)
    assert {k:v for k,v in s['brackets']['live'].items() if k!='usage'}=={k:v for k,v in live_summary.items() if k!='usage'}
    assert {k:v for k,v in s['complexity'].items() if k!='direct'}==complexity_summary(a['complexity'],a['complexity_instances'])
    replay_complexity(a['complexity_instances'],a['complexity'])
    calls=a['brackets']['full']+a['brackets']['teacher']+a['brackets']['direct']+[c for r in a['brackets']['live'] for c in r['calls']]+[c for r in a['complexity'] for c in r['calls']]+a['membership']
    for c in calls:
        assert c['reasoning_tokens']+c['answer_tokens']==c['usage']['completion_tokens']
        assert c['usage']['completion_tokens']<=c['max_tokens']
        if not c['reasoning_closed']:assert c['text']=='' and c['answer_tokens']==0


def test_x10_reported_bracket_counts():
    s=json.loads((ROOT/'docs/results/2026-10-04-x10/summary.json').read_text())['brackets']
    assert s['full']['held_out']['total']==[43,92]
    assert s['oracle']['oracle_trace']['held_out']['total']==[0,92]
    assert s['oracle']['action_accuracy']==[0,988]
    assert s['live']['held_out']['total']==[0,92]
    assert s['direct']['exact']==[16,92]
    for name,calls,truncated,completion,reasoning,answer in [
        ('full',92,31,127635,123614,4021),
        ('oracle',988,988,126464,126464,0),
        ('live',92,92,11776,11776,0),
        ('direct',92,76,44755,44643,112),
    ]:
        u=s[name]['usage']
        assert [u[k] for k in ('calls','truncated','completion_tokens','reasoning_tokens','answer_tokens')]==[calls,truncated,completion,reasoning,answer]
    x4=json.loads((ROOT/'docs/results/2026-10-04-x4/summary.json').read_text())
    comparison={r['model']:r['full']['usage']['completion_tokens'] for r in x4 if r['model'].startswith('14b')}
    assert comparison=={'14b_zeroshot':9352,'14b_lora':7009}


def test_x10_reported_complexity_counts():
    root=ROOT/'docs/results/2026-10-04-x10'
    s=json.loads((root/'summary.json').read_text())['complexity']
    assert [s[m]['overall']['exact'][0] for m in ('full_trace','oracle_step','external_state')]==[44,0,0]
    assert s['direct']['exact']==[9,240]
    assert [v['exact'][0] for v in s['full_trace']['by_family'].values()]==[5,18,1,1,6,13]
    for mode,calls,truncated,completion,reasoning,answer in [
        ('full_trace',240,136,798391,789711,8680),
        ('oracle_step',3860,3860,494080,494080,0),
        ('external_state',240,240,30720,30720,0),
        ('direct',240,231,29990,29904,86),
    ]:
        u=s[mode]['usage'] if mode=='direct' else s[mode]['overall']['usage']
        assert [u[k] for k in ('calls','truncated','completion_tokens','reasoning_tokens','answer_tokens')]==[calls,truncated,completion,reasoning,answer]
    meta=json.loads((root/'metadata.json').read_text())
    assert round(meta['completion']['seconds'])==6898
