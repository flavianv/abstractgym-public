"""Freeze the cue-free suite and historical reuse provenance before inference."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
from abstractgym.complexity import build_complexity_suite,build_cuefree_complexity_suite
from abstractgym.cuefree import shortcut_scores
from abstractgym.cfg import Grammar,Production
p=argparse.ArgumentParser();p.add_argument('--prototype',type=Path,required=True);a=p.parse_args()
sys.argv=[str(a.prototype),str(Path.cwd())];proto=runpy.run_path(str(a.prototype))
old=build_complexity_suite();new=build_cuefree_complexity_suite();prototype=proto['build']()
for row,r in zip(new,prototype):
 assert row['input']==r['tokens']
 assert [(p['lhs'],tuple(p['rhs'])) for p in row['grammar']['productions']]==r['rules']
 for name,predict in proto['H'].items():
  from abstractgym.cuefree import shortcut_predictions
  assert shortcut_predictions(row)[name]==predict(r)
root=Path('experiments/cuefree');root.mkdir(exist_ok=True)
def write(name,x):
 path=root/name
 data=json.dumps(x,indent=2,sort_keys=True)+'\n'
 if path.exists():assert path.read_text()==data
 else:path.write_text(data)
source=Path('build/cfg_controller_v1/ray/complexity')
paths=[source/'instances.jsonl',source/'original/records.jsonl',source/'membership/records.jsonl',source/'runtime.json',source/'membership/runtime.json']
assert [json.loads(l) for l in paths[0].read_text().splitlines()]==old
write('instances.json',new)
write('shortcuts.json',{'original':shortcut_scores(old),'cuefree':shortcut_scores(new)})
write('protocol.json',dict(model='Qwen/Qwen3-14B',revision='40c069824f4251a91eefaf281ebe4c544efd3e18',seed=17,thinking=False,temperature=0,top_p=1.0,top_k=-1,dtype='bfloat16',enable_lora=False,max_model_len=16384,direct_tokens=128,trace_tokens=4096,step_tokens=128,direct_batch_size=32,execution_batch_size=1,max_calls=512,examples=0,retries=0,training='none',new_direct_cases=240,new_execution_cases_per_lane=120,reused_yes_cases_per_execution_lane=120,lanes={'written_algorithm':'full_trace','gym_memory':'external_state'},prototype_sha256=hashlib.sha256(a.prototype.read_bytes()).hexdigest(),prototype_comparison='All 240 grammars/strings and seven shortcut predictions match.',historical_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
write('manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.name!='manifest.json'})
print('Frozen 240 cases; prototype parity and historical yes reuse verified')
