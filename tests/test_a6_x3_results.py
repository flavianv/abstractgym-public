import gzip
import importlib.util
import json
from pathlib import Path
import sys
from abstractgym.a6 import digest
from abstractgym.a6_replication import fresh_suite
from abstractgym.a6_followup import prompt,parse_action
from abstractgym.brackets import expected,advance

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from export_a6_x3 import summarize_records


def test_x3_frozen_records_replay_and_reported_counts():
    root=ROOT/'docs/results/2026-10-04-x3'
    audit=json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()))
    summary=json.loads((root/'summary.json').read_text())
    assert audit['instances']['fresh']==fresh_suite()
    assert digest(fresh_suite())=='22111497547715c6ba59d7544e0bbe2e51339d4189111f42e74c6d42d8e63e3e'
    counts=[60,166,36,175,60,231,43,156,54,152,46,170]
    for run,s,total in zip(audit['runs'],summary,counts):
        rows=fresh_suite() if run['test_set']=='fresh' else audit['instances']['original'];lookup={r['id']:r for r in rows}
        assert {k:v for k,v in s.items() if k not in ('model','seed','test_set')}==summarize_records(run['records'],rows)
        assert s['held_out']['total']==[total,len(rows)]
        assert {r['id'] for r in run['records']}==set(lookup)
        for r in run['records']:
            row=lookup[r['id']];pos,mode,stack=0,'opening',[]
            for i,c in enumerate(r['calls']):
                gold=expected(row,pos,mode,stack)
                assert c['model_prompt']==prompt(row,pos,mode,stack,'free_json')
                assert c['expected']==gold and c['action']==parse_action(c['text'],row)
                assert c['correct']==(c['action']==gold)
                if not c['correct'] or gold['op'] in ('ACCEPT','REJECT'):
                    assert i==len(r['calls'])-1 and r['success']==c['correct'];break
                pos,mode,stack=advance(c['action'],pos,mode,stack)
            else:raise AssertionError('unfinished rollout')
    letters=[s['by_alphabet']['abcde']['total'][0] for s in summary if s['test_set']=='original']
    assert letters==[46,32,46,42,46,42]
    maps=[[v['total'][0] for v in s['by_alphabet'].values()] for s in summary if s['test_set']=='fresh']
    assert maps==[[17,68,71,10],[57,61,47,10],[76,75,76,4],[51,53,51,1],[54,49,47,2],[51,52,59,8]]


def test_x3_seed_aggregates_match_all_three_runs():
    from statistics import mean
    root=ROOT/'docs/results/2026-10-04-x3';s=json.loads((root/'summary.json').read_text());m=json.loads((root/'metadata.json').read_text())
    for r in m['aggregates']:
        values=[x['held_out']['total'][0]/x['held_out']['total'][1] for x in s if x['model']==r['model'] and x['test_set']==r['test_set']]
        assert len(values)==3
        assert (r['mean'],r['minimum'],r['maximum'])==(mean(values),min(values),max(values))
