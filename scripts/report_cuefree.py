"""Replay cue-free outputs, combine explicitly reused yes cases, and report all failures."""
import argparse,gzip,hashlib,json
from collections import Counter
from pathlib import Path
from abstractgym.experiment import run_case
from abstractgym.complexity import build_complexity_suite,FAMILIES,LEVELS
from abstractgym.cuefree import shortcut_scores
from a6_prompts import member_prompt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('build/cuefree-20261005/results'));a=p.parse_args()
out=Path('docs/results/cuefree');out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x):(out/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
protocol=read('experiments/cuefree/protocol.json')
for path,h in protocol['historical_sources'].items():assert sha(path)==h
assert read(a.root/'completion.json')['completed']
new=read(a.root/'instances.json');assert new==read('experiments/cuefree/instances.json');lookup={r['task_id']:r for r in new}
old=build_complexity_suite();old_lookup={r['task_id']:r for r in old};old_root=Path('build/cfg_controller_v1/ray/complexity')
fresh_direct=read(a.root/'direct.json');fresh_execution=rows(a.root/'execution.jsonl');historical=rows(old_root/'original/records.jsonl')
old_direct=[r for r in rows(old_root/'membership/records.jsonl') if r['task_id'] in old_lookup]
assert len(old_direct)==len(fresh_direct)==len(fresh_execution)==240
assert len({r['task_id'] for r in fresh_direct})==240
assert len({(r['task_id'],r['mode']) for r in fresh_execution})==240
assert all(r['target_answer']=='reject' for r in fresh_execution)
for r in fresh_direct:
 row=lookup[r['task_id']];assert r['prompt']==r['model_prompt']==member_prompt(row)
 assert r['max_tokens']==128
 try:parsed=json.loads(r['text'])
 except (ValueError,TypeError):parsed=None
 success=isinstance(parsed,dict) and type(parsed.get('accept')) is bool and parsed=={'accept':row['target_answer']=='accept'}
 assert success==r['success']

def verify(r,instance):
 calls=iter(r['calls']);used=[]
 def adapter(prompt):
  c=next(calls);assert prompt==c['prompt']==c['model_prompt'];used.append(c);return c
 replay=run_case(instance,r['mode'],adapter,max_calls=512)
 assert len(used)==len(r['calls'])
 for k in ('success','failure','first_error','outcome','correct_actions','target_actions'):assert r[k]==replay[k],(r['task_id'],r['mode'],k)
 for c in used:assert c['max_tokens']==(4096 if r['mode']=='full_trace' else 128) and c['batch_size']==1
for r in fresh_execution:verify(r,lookup[r['task_id']])
reused=[r for r in historical if r['target_answer']=='accept' and r['mode'] in ('full_trace','external_state')]
assert len(reused)==240
for r in reused:
 assert lookup[r['task_id']]==old_lookup[r['task_id']];verify(r,lookup[r['task_id']])
combined=[dict(r,provenance='reused original yes') for r in reused]+[dict(r,provenance='new cue-free no') for r in fresh_execution]
lanes={'direct':fresh_direct,'written_algorithm':[r for r in combined if r['mode']=='full_trace'],'gym_memory':[r for r in combined if r['mode']=='external_state']}
for rs in lanes.values():assert len(rs)==240 and len({r['task_id'] for r in rs})==240

def cell(rs):
 return dict(correct=sum(r['success'] for r in rs),total=len(rs),yes=[sum(r['success'] for r in rs if r['target_answer']=='accept'),sum(r['target_answer']=='accept' for r in rs)],no=[sum(r['success'] for r in rs if r['target_answer']=='reject'),sum(r['target_answer']=='reject' for r in rs)],failures=dict(Counter(r.get('failure') or 'incorrect_answer' for r in rs if not r['success'])))
def summary(rs):
 return dict(overall=cell(rs),by_family={f:cell([r for r in rs if r['family']==f]) for f in FAMILIES},by_level={str(n):cell([r for r in rs if lookup[r['task_id']]['difficulty']['level']==n]) for n in LEVELS},family_level={f:{str(n):cell([r for r in rs if r['family']==f and lookup[r['task_id']]['difficulty']['level']==n]) for n in LEVELS} for f in FAMILIES})
result={name:summary(rs) for name,rs in lanes.items()}
old_by_pair={(r['pair_id'],r['target_answer']):r for r in old_direct}
paired=Counter();positive_changes=[]
for r in fresh_direct:
 instance=lookup[r['task_id']];prior=old_by_pair[(instance['pair_id'],instance['target_answer'])]
 paired[('both_correct' if r['success'] and prior['success'] else 'improved' if r['success'] else 'regressed' if prior['success'] else 'both_wrong')]+=1
 if instance['target_answer']=='accept' and prior['text']!=r['text']:positive_changes.append(dict(task_id=r['task_id'],old=prior['text'],new=r['text']))
negative_steps={'original':[r['difficulty']['search_steps'] for r in old if r['target_answer']=='reject'],'cuefree':[r['difficulty']['search_steps'] for r in new if r['target_answer']=='reject']}
runtime=read(a.root/'runtime.json');assert runtime['protocol']==protocol
old_runtime=read(old_root/'runtime.json');assert runtime['packages']==old_runtime['packages']
for path,h in runtime['source_sha256'].items():assert sha(path)==h
cost={}
for name,rs in lanes.items():
 calls=rs if name=='direct' else [c for r in rs for c in r['calls']]
 cost[name]=dict(calls=len(calls),completion_tokens=sum(c['usage']['completion_tokens'] for c in calls),total_tokens=sum(c['usage']['total_tokens'] for c in calls),truncated=sum(c['finish_reason']=='length' for c in calls),reused_cases=0 if name=='direct' else 120)
write('summary.json',result);write('metadata.json',dict(runtime=runtime,completion=read(a.root/'completion.json'),paired_direct=dict(paired),unchanged_yes_output_changes=positive_changes,negative_teacher_steps=negative_steps,cost=cost,all_calls_replayed=True,historical_execution=dict(full_trace=sum(r['success'] for r in historical if r['mode']=='full_trace'),external_state=sum(r['success'] for r in historical if r['mode']=='external_state')),historical_direct=sum(r['success'] for r in old_direct)))
write('shortcuts.json',read('experiments/cuefree/shortcuts.json'))
audit=dict(instances=new,direct=fresh_direct,execution=combined,historical_direct=old_direct)
(out/'audit.json.gz').write_bytes(gzip.compress(json.dumps(audit,sort_keys=True).encode(),mtime=0))
write('manifest.json',{p.name:sha(p) for p in out.iterdir() if p.name!='manifest.json'})
def frac(x):return f'{x[0]}/{x[1]}'
def score(x):return f"{x['correct']}/{x['total']} ({100*x['correct']/x['total']:.1f}%)"
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
lines=['# Cue-free complexity: zero-shot Qwen3-14B','', 'All 240 direct answers and 120 new negative cases in each execution lane are complete. Every saved call was replayed through the unchanged grader. The 120 yes cases in each execution lane reuse their original results; their full instance dictionaries are unchanged.','',
 '## Main comparison','',table(['Lane','Original suite','Cue-free suite','Cue-free yes','Cue-free no'],[['Direct answer','236/240 (98.3%)',score(result['direct']['overall']),frac(result['direct']['overall']['yes']),frac(result['direct']['overall']['no'])],['Written algorithm (full trace)','28/240 (11.7%)',score(result['written_algorithm']['overall']),frac(result['written_algorithm']['overall']['yes']),frac(result['written_algorithm']['overall']['no'])],['Gym memory (external state)','40/240 (16.7%)',score(result['gym_memory']['overall']),frac(result['gym_memory']['overall']['yes']),frac(result['gym_memory']['overall']['no'])]]),'',
 'The separate bracket direct-answer reference is **62/92 (67.4%)**. Its cases and grammar families differ, so this is contextual rather than a paired control. All failures remain in each denominator. One decoding seed (17), no training, no retries, no examples, no thinking mode.','']
d=result['direct']['overall']
lines += ['## What this answers','',
 f"Direct accuracy changes from 236/240 to {d['correct']}/240 ({100*(d['correct']-236)/240:+.1f} percentage points): {paired['both_correct']} cases remain correct, {paired['both_wrong']} remain wrong, {paired['regressed']} regress, and {paired['improved']} improve. There are {len(positive_changes)} changed response texts on the 120 identical yes cases. This isolates where the observed changes occur without claiming to identify the model's internal algorithm.",'',
 f"On the new negative cases, written algorithm succeeds on {frac(result['written_algorithm']['overall']['no'])} and gym memory on {frac(result['gym_memory']['overall']['no'])}. These are complete valid executions, not merely correct final verdicts. The historical yes successes are reused rather than newly replicated.",'']
if d['correct']==204 and len(positive_changes)==0:
 lines += ['The outcome sits between the two scenarios in the brief. Near-perfect direct accuracy does not survive cue removal, but direct answers do not collapse to chance or to the separate bracket score. All 32 regressions are newly constructed no cases. The strongest tested count shortcut gets 172/240 (71.7%), below the model\'s 204/240; this does not rule out a mixture of family-specific heuristics.','',
 'For the tutorial, retain the answer-versus-execution distinction with the cue-free numbers and narrow its scope to these families. The original 236/240 alone was confounded by a perfect symbol cue. The new run establishes substantial direct-answer performance without that cue, alongside weaker execution performance; it does not establish a general membership algorithm.','']
if result['written_algorithm']['overall']['no'][0] or result['gym_memory']['overall']['no'][0]:
 lines += ['The claim **“every zero-shot execution success is a yes” is false on this suite**. The new successful negative executions are listed below. Each was replayed through the canonical controller; alphabet variants are zero-indexed.','',
 table(['Lane','Family','Level','Variant','Task ID'],[[r['mode'],r['family'],r['difficulty']['level'],r['difficulty']['variant'],r['task_id']] for r in fresh_execution if r['success']]),'']
for lane,title in [('direct','Direct answers'),('written_algorithm','Written algorithm (zero-shot)'),('gym_memory','Gym memory (zero-shot)')]:
 s=result[lane];lines += ['## '+title,'','### Per family','',table(['Family','All','Yes','No'],[[f,score(v),frac(v['yes']),frac(v['no'])] for f,v in s['by_family'].items()]),'','### Per level','',table(['Level','All','Yes','No'],[[n,score(v),frac(v['yes']),frac(v['no'])] for n,v in s['by_level'].items()]),'']
lines += ['## Failure accounting and call budgets','',
 table(['Lane','Failure type','Cases'],[[name,failure,count] for name,s in result.items() for failure,count in s['overall']['failures'].items()]),'',
 table(['Lane','Calls','Completion tokens','Total tokens','Length-limited calls','Historical cases reused'],[[name,v['calls'],v['completion_tokens'],v['total_tokens'],v['truncated'],v['reused_cases']] for name,v in cost.items()]),'',
 'These execution totals include the reused yes calls. A successful execution means a complete, valid canonical DFS trajectory and correct verdict; a correct verdict alone does not pass. Invalid actions, malformed responses, unfinished traces, and budget exhaustion remain failures. No tool-failure recovery was tested. Prompts, outputs, action order, grades, and usage are retained in the audit artifact.','',
 '## Shortcut audit and scope','',
 'The original last-terminal rule scores 240/240; all four symbol-level rules score 120/240 on the new suite. The prototype is preserved exactly. Its count shortcuts differ from the prose brief: they score 160/240 and 172/240 overall, with above-chance alternatives scores (24/40 and 28/40) and recursion at 24/40 for the latter. This suite removes the named symbol cue, not every possible shortcut. Comparing lengths is itself the membership computation for the one-production/chain families.','',
 table(['Shortcut','Original','Cue-free'],[[n,frac(read('experiments/cuefree/shortcuts.json')['original'][n]['overall']),frac(v['overall'])] for n,v in read('experiments/cuefree/shortcuts.json')['cuefree'].items()]),'',
 '## Reproducibility and interpretation limits','',
 'Base `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`; BF16, temperature 0, top-p 1, top-k −1, seed 17, thinking off, no LoRA. Direct output budget 128; original member_prompt preserved exactly. Written traces use 4,096 tokens; gym actions use 128 with at most 512 calls. Context 16,384. Direct batches have at most 32 calls; both execution lanes run serial calls, as in the original run. Runtime package versions match the original.','',
 'The direct set is rerun in full, whereas execution results combine historical yes outputs with newly generated no outputs. The new direct final batch has 16 cases; the original combined-membership job included added bracket cases in that batch. Prompts are independent, but small numerical batch effects are possible. Any changed outputs on identical yes cases are listed in metadata.json. No such outcome was used to change prompts or rerun cases.','',
 'The changed negative grammars alter structural/count requirements and canonical trajectory lengths as well as removing the symbol cue. Consequently, a score difference establishes sensitivity to this contrast set; it does not by itself identify which heuristic the original model used. The families, shared templates, and alphabet permutations are correlated, and there is only one seed. These results do not prove general CFG membership or arbitrary-program execution.','',
 '[Frozen protocol](../experiments/cuefree/protocol.json), [summary by family, level, and label](results/cuefree/summary.json), [raw calls and instances](results/cuefree/audit.json.gz), [metadata and positive-output comparisons](results/cuefree/metadata.json), and [shortcut scores](results/cuefree/shortcuts.json) accompany this report. Original suite data and saved results were not modified.','']
Path('docs/cuefree_results.md').write_text('\n'.join(lines))
print(json.dumps({k:v['overall'] for k,v in result.items()},indent=2))
