import gzip
import json
from pathlib import Path
import sys
from abstractgym.a6_trace import score,trace_prompt,injected_prefix,reference
from abstractgym.a6_followup import oracle_states,parse_action
from abstractgym.brackets import suite
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from export_a6_x45 import case_summary,null_summary,injection_summary,summarize_tiny,tiny_x5
from export_a6_followups import teacher_summary,execution_summary


def test_x4_x5_scores_recompute_from_predictions():
    root=ROOT/'docs/results/2026-10-04-x4'
    summary=json.loads((root/'summary.json').read_text());audit=json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()))
    five=json.loads((ROOT/'docs/results/2026-10-04-x5/summary.json').read_text())
    rows=[r for r in suite() if r['split']=='test'];lookup={r['id']:r for r in rows};assert audit['instances']==rows
    for s,f in zip(summary,five):
        run=audit['runs'][s['model']]
        assert s['full']==case_summary(run['full'],rows)
        assert s['oracle']==teacher_summary(run['teacher'],rows)
        assert s['live']==execution_summary(run['live'],rows)
        assert f['independence']==null_summary(rows,run['teacher'],run['full'],run['live'])
        assert len(run['teacher'])==988
        states={(s['id'],s['call']):s for r in rows for s in oracle_states(r)}
        for r in run['teacher']:
            assert r['expected']==states[r['id'],r['call']]['expected']
            assert r['correct']==(parse_action(r['text'],lookup[r['id']])==r['expected'])
        for r in run['full']:
            row=lookup[r['id']];assert r['model_prompt']==trace_prompt(row)
            assert all(r[k]==v for k,v in score(r['text'],row).items())
        for k,records in run['injection'].items():
            assert len(records)==48
            assert f['injection'][k]==injection_summary(records,[r for r in rows if r['depth']>=4])
            for r in records:
                row=lookup[r['id']];prefix=injected_prefix(row,int(k))
                assert r['assistant_prefix']==prefix and r['model_prompt']==trace_prompt(row)
                assert all(r[key]==v for key,v in score(r['text'],row,prefix,True).items())
    meta=json.loads((root/'metadata.json').read_text());assert meta['tiny']['summary']==summarize_tiny(audit['tiny'])
    m5=json.loads((ROOT/'docs/results/2026-10-04-x5/metadata.json').read_text());assert m5['tiny']==tiny_x5(audit['tiny'],audit['tiny_injection'])
    assert [s['full'] for s in meta['tiny']['summary']]==[[60,92]]*3
    assert [s['oracle_sequence'] for s in meta['tiny']['summary']]==[[60,92]]*3
    assert [s['teacher_actions'] for s in meta['tiny']['summary']]==[[796,988],[820,988],[794,988]]
    assert [[v['action_accuracy'][0] for v in s['injection'].values()] for s in m5['tiny']]==[[266,246,246,176],[314,296,274,212],[244,220,210,172]]


def test_x4_x5_reported_totals_and_injection_counts():
    four=json.loads((ROOT/'docs/results/2026-10-04-x4/summary.json').read_text())
    five=json.loads((ROOT/'docs/results/2026-10-04-x5/summary.json').read_text())
    assert [r['full']['held_out']['total'][0] for r in four]==[30,28,0,0]
    assert [r['oracle']['oracle_trace']['held_out']['total'][0] for r in four]==[43,60,0,43]
    assert [r['live']['held_out']['total'][0] for r in four]==[43,60,0,43]
    assert [r['full']['valid_arrays'][0] for r in four[:2]]==[83,84]
    assert [[s['action_accuracy'][0] for s in r['injection'].values()] for r in five]==[[502,415,444,214],[424,412,408,236],[60,67,47,53],[21,6,8,11]]
    assert all(v['action_accuracy'][1]==576 for r in five for v in r['injection'].values())
    assert [[s['usage']['truncated'] for s in r['injection'].values()] for r in five]==[[0,3,3,4],[3,2,3,1],[22,21,26,20],[31,33,35,33]]
    meta=json.loads((ROOT/'docs/results/2026-10-04-x4/metadata.json').read_text())
    assert len(meta['strict_rank_inversions'])==1
    assert meta['strict_rank_inversions'][0]['left']=='14b_zeroshot'
    assert [v['different_calls'] for v in meta['batch_consistency'].values()]==[0,1,0,0]


def test_posthoc_corrupted_state_diagnostic_keeps_primary_scores():
    from analyze_a6_history_state import summarize,corrupted_reference
    root=ROOT/'docs/results/2026-10-04-x5';audit=json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()))
    saved=json.loads((root/'posthoc_corrupted_state.json').read_text())['results']
    assert saved==summarize(audit)
    for row in audit['instances']:
        if row['depth']>=4:assert corrupted_reference(row,0)==reference(row)[4:]
    for model,counts in [('14b_zeroshot',[18,2,4,2]),('14b_lora',[20,5,3,2])]:
        selected=[r for r in saved if r['model']==model]
        assert [r['targets_differ'] for r in selected]==[0,24,24,32]
        assert [r['corrupted_correct'] for r in selected]==counts
        assert selected[-1]['corrupted_only'] and len(selected[-1]['corrupted_only'])==2
