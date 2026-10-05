import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_a6_x9 import summarize,correlation
from a6_prompts import member_prompt


def test_answer_trace_confusions_and_rank_correlations():
    root=ROOT/'docs/results/2026-10-04-x9'
    summary=json.loads((root/'summary.json').read_text());audit=json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()))
    meta=json.loads((root/'metadata.json').read_text())
    lookup={r['task_id']:r for r in audit['instances']}
    for model,answers in audit['answers'].items():
        outcomes={}
        for answer in answers:
            row=lookup[answer['task_id']]
            assert answer['model_prompt']==member_prompt(row) and answer['max_tokens']==128
            try:parsed=json.loads(answer['text'])
            except ValueError:parsed=None
            correct=parsed=={'accept':row['target_answer']=='accept'}
            assert answer['success']==correct
            outcomes[answer['task_id']]=correct
        assert all(r['answer_correct']==outcomes[r['task_id']] for r in audit['records'] if r['model']==model)
    assert [{k:v for k,v in s.items() if k!='label_blind'} for s in summary]==summarize(audit['records'])
    for s in summary:
        assert sum(s['confusion'].values())==s['total']==240
        assert sum(map(len,s['case_ids'].values()))==240
        if s['model']=='14b_zeroshot':
            assert s['answer_correct']==236
            assert s['confusion']['answer_True_trace_False']==(208 if s['mode']=='full_trace' else 196)
            assert s['confusion']['answer_False_trace_True']==0
    for mode,value in meta['rank_correlations'].items():
        rs=[s for s in summary if s['mode']==mode]
        assert value==correlation([s['answer_correct']/240 for s in rs],[s['trace_correct']/240 for s in rs])
    assert correlation([1,1],[1,2]) is None
    assert correlation([1,2,2],[3,1,1])==-1


def test_x9_trace_outcomes_match_replayable_prediction_audits():
    def audit(name):
        path=ROOT/'docs/results'/('2026-10-04-'+name)/'audit.json.gz'
        return json.loads(gzip.decompress(path.read_bytes()))
    x7,x9,x10=audit('x7'),audit('x9'),audit('x10')
    sources={'14b_zeroshot':x7['runs']['zeroshot'],
             '14b_dfs_lora':x7['runs']['dfs_lora'],
             '14b_thinking_fixed_caps':x10['complexity']}
    expected={(model,r['task_id'],r['mode']):r['success'] for model,rs in sources.items() for r in rs}
    actual={(r['model'],r['task_id'],r['mode']):r['trace_correct'] for r in x9['records']}
    assert actual==expected


def test_x9_reported_tables_and_ranking_reversal():
    root=ROOT/'docs/results/2026-10-04-x9'
    rows=json.loads((root/'summary.json').read_text())
    want={
        '14b_zeroshot':([28,208,0,4],[40,196,0,4],236),
        '14b_dfs_lora':([0,237,0,3],[130,107,3,0],237),
        '14b_thinking_fixed_caps':([2,7,42,189],[0,9,0,231],9),
    }
    keys=['answer_True_trace_True','answer_True_trace_False','answer_False_trace_True','answer_False_trace_False']
    for r in rows:
        full,step,answer=want[r['model']]
        assert [r['confusion'][k] for k in keys]==(full if r['mode']=='full_trace' else step)
        assert r['answer_correct']==answer
    assert json.loads((root/'metadata.json').read_text())['rank_correlations']=={'full_trace':-1.,'oracle_step':1.,'external_state':1.}
