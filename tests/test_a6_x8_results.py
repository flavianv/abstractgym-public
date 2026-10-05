import gzip
import json
from pathlib import Path
import sys
from abstractgym.brackets import suite,direct_prompt,score_answer,expected,advance
from abstractgym.a6_followup import parse_action
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from export_a6_objectives import direct_summary
from export_a6_followups import execution_summary


def test_x8_objective_results_and_no_repair_replay():
    root=ROOT/'docs/results/2026-10-04-x8';summary=json.loads((root/'summary.json').read_text())
    audit=json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()));rows=[r for r in suite() if r['split']=='test'];lookup={r['id']:r for r in rows}
    assert rows==[r for r in suite() if r['split']=='test']
    for name,direct,live in [('zeroshot',62,43),('step_lora',62,60),('membership_lora',82,11)]:
        run=audit['runs'][name];s=summary[name]
        assert s['direct']==direct_summary(run['direct'],rows)
        assert s['live']==execution_summary(run['live'],rows)
        assert s['direct']['exact']==[direct,92] and s['live']['held_out']['total']==[live,92]
        for r in run['direct']:
            row=lookup[r['id']];assert r['model_prompt']==direct_prompt(row)
            assert r['verified']==score_answer(row,r['text'])['verified']
        for r in run['live']:
            row=lookup[r['id']];pos,mode,stack=0,'opening',[]
            for i,c in enumerate(r['calls']):
                gold=expected(row,pos,mode,stack)
                assert c['expected']==gold and c['action']==parse_action(c['text'],row)
                assert c['correct']==(c['action']==gold)
                if not c['correct'] or gold['op'] in ('ACCEPT','REJECT'):
                    assert i==len(r['calls'])-1 and r['success']==c['correct'];break
                pos,mode,stack=advance(c['action'],pos,mode,stack)
            else:raise AssertionError('unfinished rollout')
    assert summary['membership_lora']['direct']['by_label']=={'True':[42,46],'False':[40,46]}
    t=json.loads((root/'metadata.json').read_text())['training']['training']
    assert t['processed_token_budget']==1351623 and t['scheduled_processed_tokens']==1355687
    assert t['scheduled_supervised_tokens']==49920 and t['optimizer_updates']==260
    assert t['balanced_membership_sampling'] and t['training_examples']==367
