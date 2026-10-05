"""Pin every A6 count displayed in tables, and recompute them from saved predictions."""
import gzip
import json
from pathlib import Path
from statistics import mean
from abstractgym.a6 import digest, evaluate_execution, summarize
from abstractgym.brackets import suite

ROOT=Path(__file__).resolve().parents[1]/'docs/results/2026-10-03-a6'
SUMMARY=[e for e in json.loads((ROOT/'a6_summary.json').read_text()) if e['condition'] in ('qwen14b_zeroshot','tiny_trained')]
AUDIT=json.loads(gzip.decompress((ROOT/'a6_audit.json.gz').read_bytes()))
METADATA=json.loads((ROOT/'a6_metadata.json').read_text())
DEPTHS=('0','1','2','3','4','8','16')
COUNTS=(2,4,8,8,8,8,8)


def test_summary_table_exact_measured_counts():
    assert [(e['condition'],e['seed']) for e in SUMMARY]==[('qwen14b_zeroshot',17)]+[('tiny_trained',s) for s in range(3)]
    baseline=SUMMARY[0]
    assert baseline['held_out']['total']==[43,92]
    assert baseline['action_accuracy']==[488,537]
    assert baseline['first_error_median']==2
    assert baseline['push_value_errors']==32
    assert baseline['direct_answer']==[62,92]
    assert baseline['exhaustive_check'] is None and baseline['train_examples']==0
    for e,direct in zip(SUMMARY[1:],(76,80,76)):
        assert e['held_out']['total']==[92,92]
        assert e['action_accuracy']==[988,988]
        assert e['exhaustive_check']==[36,36]
        assert e['extra_depths']=={'32':[16,16],'64':[16,16]}
        assert e['direct_answer']==[direct,92]
        assert e['first_error_median'] is None and e['push_value_errors']==0
        assert e['train_examples']==2305 and e['unique_train_examples']==36
        assert e['token_usage']['execution_input_tokens']==3952
        assert e['training']['step']['parameters']==68230
        assert e['training']['direct']['parameters']==67970
        assert e['training']['step']['train_accuracy']==[36,36]
        assert e['training']['direct']['train_accuracy']==[366 if e['seed']==0 else 367,367]
    assert METADATA['means']['direct_answer']==mean([76/92,80/92,76/92])
    assert METADATA['mean_execution']==1 and METADATA['means']['exhaustive_check']==1
    assert METADATA['plan']['training']['trajectories']==367
    assert METADATA['plan']['training']['direct_positive']==31


def test_every_per_depth_count_and_label_blind_annotation():
    for e in SUMMARY:
        assert e['label_blind']['always_accept']==e['label_blind']['always_reject']==[46,92]
        for name,baseline in [('letters',(2,4,4,5,5,5,4)),('brackets',(2,2,2,2,2,2,2))]:
            values=baseline if e['condition']=='qwen14b_zeroshot' else COUNTS
            assert e['held_out'][name]['by_depth']=={d:[ok,n] for d,ok,n in zip(DEPTHS,values,COUNTS)}
            for d,n in zip(DEPTHS,COUNTS):
                assert e['label_blind']['by_cell'][name][d]=={'always_accept':[n//2,n],'always_reject':[n//2,n]}
                for measure in ('held_out','direct_by_depth'):
                    ok,total=e[measure][name]['by_depth'][d]
                    status=e[measure][name]['interpretation'][d]
                    if ok*2==total:assert status=='chance / label-blind'
                    elif ok*2>total:assert status=='beats both label-blind baselines'
                    else:assert status=='below label-blind'
            if e['condition']=='tiny_trained':
                expected=(2,4,8,8,8,5 if e['seed']==1 else 4,5 if e['seed']==1 else 4)
            else:expected=(2,2,4,4,4,4,4) if name=='letters' else (2,4,8,6,6,6,6)
            assert e['direct_by_depth'][name]['by_depth']=={d:[ok,n] for d,ok,n in zip(DEPTHS,expected,COUNTS)}


def test_saved_predictions_reproduce_summaries_without_reference_repair():
    rows=AUDIT['instances']
    assert rows==[r for r in suite() if r['split']=='test']
    assert digest(rows)==METADATA['held_out_hash']=='d2c7202ea08da83ad5398723a2cc4ce9d3175b7e5f7b84ce40aa529103e0d6e2'
    measured=summarize(rows,AUDIT['baseline']['records'])
    for key in ('held_out','action_accuracy','first_error_median','label_blind'):assert measured[key]==SUMMARY[0][key]
    assert sum(r['success'] for r in AUDIT['baseline']['direct'])==62
    bad_push=sum(c['expected'] in (0,1) and json.loads(c['raw_action']).get('op')=='PUSH' and not c['correct'] for r in AUDIT['baseline']['records'] for c in r['calls'])
    assert bad_push==32
    for e,audit in zip(SUMMARY[1:],AUDIT['tiny']):
        policy={tuple(r['state']):r['prediction'] for r in audit['exhaustive']}
        assert len(policy)==36
        replay=evaluate_execution(rows,lambda keys:[policy[k] for k in keys])
        assert replay==audit['held_out']
        measured=summarize(rows,replay)
        for key in measured:assert measured[key]==e[key]
        assert evaluate_execution(AUDIT['extra'],lambda keys:[policy[k] for k in keys])==audit['extra']
        assert sum(r['success'] for r in audit['direct'])==e['direct_answer'][0]
        assert sum(r['prediction']==r['expected'] for r in audit['exhaustive'])==e['exhaustive_check'][0]
