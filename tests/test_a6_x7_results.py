import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from export_a6_objectives import complexity_summary
from test_a6_report_replay import replay_complexity
from abstractgym.a6 import digest
from abstractgym.a6_objectives import dfs_examples


def test_x7_scores_replay_and_no_heldout_selection():
    root=ROOT/'docs/results/2026-10-04-x7'
    summary=json.loads((root/'summary.json').read_text());audit=json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()))
    meta=json.loads((root/'metadata.json').read_text());rows=audit['instances']
    assert len(rows)==240
    assert meta['plan']['held_out_hash']==digest(rows)
    assert meta['plan']['training_hash']==digest(dfs_examples())
    for name,records in audit['runs'].items():
        assert len(records)==720
        assert summary[name]==complexity_summary(records,rows)
        replay_complexity(rows,records)
    assert [summary['zeroshot'][m]['overall']['exact'][0] for m in ('full_trace','oracle_step','external_state')]==[28,40,40]
    assert summary['zeroshot']['teacher_by_action']['DONE']==[20,492]
    t=meta['training']['training']
    assert t['gradient_checkpointing']
    assert t['scheduled_processed_tokens']==27750309 and t['scheduled_supervised_tokens']==295089
    assert t['optimizer_updates']==1032 and t['max_length']==1125
    assert round(t['training_seconds'])==6957
    for name in ('initialization_parity','reload_parity'):
        parity=meta['training'][name]
        assert parity['allclose'] and parity['mean_abs']==parity['max_abs']==0
    assert meta['plan']['training_examples']==11002 and meta['plan']['training_trajectories']==64


def test_x7_reported_generalization_and_failure_counts():
    root=ROOT/'docs/results/2026-10-04-x7'
    s=json.loads((root/'summary.json').read_text());trained=s['dfs_lora']
    assert [trained[m]['overall']['exact'][0] for m in ('full_trace','oracle_step','external_state')]==[0,133,133]
    assert trained['oracle_step']['overall']['action_accuracy']==[3593,3860]
    assert trained['teacher_by_action']=={'TRY':[1433,1604],'ACC':[120,120],'PRN':[609,636],'BT':[1008,1008],'DONE':[423,492]}
    assert s['zeroshot']['teacher_by_action']=={'TRY':[991,1604],'ACC':[102,120],'PRN':[182,636],'BT':[122,1008],'DONE':[20,492]}
    want={'alternatives':[8,8,8,8,8],'chain_depth':[0,0,0,0,0],
          'nesting':[8,8,8,0,0],'recursion':[8,8,8,5,0],
          'sequence':[0,0,0,0,0],'terminal_length':[8,8,8,8,8]}
    for family,counts in want.items():
        cells=[r for r in trained['external_state']['cells'] if r['family']==family]
        assert [r['level'] for r in cells]==[1,2,4,8,16]
        assert [r['exact'] for r in cells]==[[n,8] for n in counts]
    assert [v['exact'][0] for v in s['zeroshot']['external_state']['by_family'].values()]==[4,4,0,0,12,20]
    assert trained['full_trace']['overall']['failures']=={'incomplete_trace':142,'invalid_or_trailing_action':51,'call_or_parse_error':47}
    assert trained['full_trace']['overall']['usage']['truncated']==47
    assert trained['full_trace']['overall']['usage']['completion_tokens']==196743
    audit=json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()))
    full=[r for r in audit['runs']['dfs_lora'] if r['mode']=='full_trace']
    assert {r['task_id'] for r in full if r['failure']=='call_or_parse_error'}=={r['task_id'] for r in full if r['calls'][0]['finish_reason']=='length'}
    example=next(r for r in full if r['task_id']=='complexity_20a7be0db426')
    assert example['calls'][0]['finish_reason']=='length'
    assert example['calls'][0]['text'].count('{"op":"DONE"},{"op":"BT"}')>10
    assert round(json.loads((root/'metadata.json').read_text())['completion']['seconds'])==10942
