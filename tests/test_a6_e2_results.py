"""Recompute E2 counts from saved predictions; no model execution or retraining."""
import gzip
import json
import pytest
from pathlib import Path
from abstractgym.a6 import summarize,digest
from abstractgym.brackets import suite

ROOT=Path(__file__).resolve().parents[1]/'docs/results/2026-10-03-a6'
ENTRIES={e['condition']:e for e in json.loads((ROOT/'a6_summary.json').read_text())}
AUDIT=json.loads(gzip.decompress((ROOT/'a6_e2_audit.json.gz').read_bytes()))
META=json.loads((ROOT/'a6_e2_metadata.json').read_text())


@pytest.mark.parametrize('stem',['a6_e2','a6_e2_14b'])
def test_e2_saved_predictions_and_frozen_split(stem):
    audit=json.loads(gzip.decompress((ROOT/(stem+'_audit.json.gz')).read_bytes()))
    meta=json.loads((ROOT/(stem+'_metadata.json')).read_text())
    rows=audit['instances']
    assert rows==[r for r in suite() if r['split']=='test']
    assert digest(rows)==meta['plan']['frozen_held_out_hash']
    for name,a in audit['conditions'].items():
        entry=ENTRIES[name];measured=summarize(rows,a['records'])
        for key in ('held_out','action_accuracy','first_error_median','label_blind'):
            assert measured[key]==entry[key]
        assert sum(r['success'] for r in a['direct'])==entry['direct_answer'][0]
        direct=summarize(rows,[dict(r,calls=[],first_error=None) for r in a['direct']])
        assert direct['held_out']==entry['direct_by_depth']
        assert meta['conditions'][name]['prompt_audit']['exact_frozen_prompts']
        assert meta['conditions'][name]['prompt_audit']['live_step_prompts']==entry['action_accuracy'][1]
        for record in a['records']:
            assert record['success']==all(c['correct'] for c in record['calls'])
            assert all(c['correct'] for c in record['calls'][:-1])


def test_e2_measured_table_and_errors():
    zero=ENTRIES['qwen17b_zeroshot'];trained=ENTRIES['qwen17b_lora']
    assert zero['held_out']['total']==[0,92]
    assert zero['action_accuracy']==[0,92]
    assert zero['development_action_accuracy']==[0,148]
    assert zero['invalid_json_calls']==zero['truncated_calls']==92
    assert zero['direct_answer']==[48,92]
    assert trained['held_out']['total']==[43,92]
    assert trained['action_accuracy']==[445,494]
    assert trained['development_action_accuracy']==[148,148]
    assert trained['invalid_json_calls']==44 and trained['truncated_calls']==0
    assert trained['direct_answer']==[46,92]
    assert trained['push_value_errors']==0 and trained['first_error_median']==1
    for symbols,counts in [('letters',[2,4,8,8,8,6,6]),('brackets',[1,0,0,0,0,0,0])]:
        assert [v[0] for v in trained['held_out'][symbols]['by_depth'].values()]==counts
    bad=[r['calls'][-1] for r in AUDIT['conditions']['qwen17b_lora']['records'] if not r['success']]
    invalid=[];valid=[]
    for call in bad:
        try:parsed=json.loads(call['raw_action'])
        except ValueError:invalid.append(call);continue
        valid.append((call['expected'],parsed['op']))
    assert len(invalid)==44
    assert all(c['expected'] in (0,1) for c in invalid)
    assert sorted(valid)==[(3,'REJECT')]*4+[(5,'ACCEPT')]
    training=META['conditions']['qwen17b_lora']['training']
    assert training['training_examples']==2305 and training['unique_prompts']==148
    assert training['optimizer_updates']==219 and training['trainable_parameters']==17432576
    assert META['conditions']['qwen17b_lora']['reload_parity']['max_abs']==0
    assert not META['excluded_pissa_initialization']['allclose']


def test_14b_paired_counts_and_remaining_value_error():
    zero=ENTRIES['qwen14b_e2_zeroshot'];trained=ENTRIES['qwen14b_e2_lora']
    meta=json.loads((ROOT/'a6_e2_14b_metadata.json').read_text())
    audit=json.loads(gzip.decompress((ROOT/'a6_e2_14b_audit.json.gz').read_bytes()))
    assert zero['held_out']['total']==[43,92] and trained['held_out']['total']==[60,92]
    assert zero['action_accuracy']==[488,537] and trained['action_accuracy']==[634,666]
    assert zero['development_action_accuracy']==[113,148]
    assert trained['development_action_accuracy']==[148,148]
    assert zero['direct_answer']==trained['direct_answer']==[62,92]
    assert zero['push_value_errors']==trained['push_value_errors']==32
    assert trained['invalid_json_calls']==trained['truncated_calls']==0
    assert trained['first_error_median']==1
    for symbols,counts in [('letters',[2,4,8,8,8,8,8]),('brackets',[2,2,2,2,2,2,2])]:
        assert [v[0] for v in trained['held_out'][symbols]['by_depth'].values()]==counts
    a=audit['conditions']['qwen14b_e2_lora'];z=audit['conditions']['qwen14b_e2_zeroshot']
    bad=[r['calls'][-1] for r in a['records'] if not r['success']]
    assert len(bad)==32
    assert all(json.loads(c['raw_action'])=={'op':'PUSH','value':'/]'} for c in bad)
    assert all(c['expected']==1 for c in bad)
    assert a['direct']==z['direct']
    successes=lambda records:{r['id'] for r in records if r['success']}
    assert successes(z['records'])<=successes(a['records'])
    assert len(successes(a['records'])-successes(z['records']))==17
    for key in ('held_out','action_accuracy','direct_by_depth','push_value_errors'):
        assert zero[key]==ENTRIES['qwen14b_zeroshot'][key]
    t=meta['conditions']['qwen14b_e2_lora']['training']
    assert t['trainable_parameters']==64225280 and t['optimizer_updates']==219
    assert t['training_hash']==META['conditions']['qwen17b_lora']['training']['training_hash']
    assert t['tokenized_hash']==META['conditions']['qwen17b_lora']['training']['tokenized_hash']
    assert meta['conditions']['qwen14b_e2_lora']['reload_parity']['max_abs']==0
    assert not meta['excluded_pissa_initialization']['allclose']
