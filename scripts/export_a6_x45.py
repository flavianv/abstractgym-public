"""X4 interface comparison and X5 descriptive independence/history diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from abstractgym.a6 import summarize
from abstractgym.a6_trace import reference
from export_a6_followups import execution_summary,teacher_summary,save
from analyze_a6_x9 import ranks


def usage(records):
    return dict(calls=len(records),truncated=sum(r.get('finish_reason')=='length' for r in records),
        **{k:sum(r.get(k,r.get('usage',{}).get(k,0)) for r in records) for k in ('prompt_tokens','completion_tokens','reasoning_tokens','answer_tokens')})


def case_summary(records,rows):
    result=summarize(rows,[dict(id=r['id'],success=r['success'],first_error=r.get('first_error'),calls=[]) for r in records])
    result.pop('action_accuracy');result.pop('push_value_errors')
    result['valid_arrays']=[sum(r['valid_array'] for r in records),len(records)]
    result['action_accuracy']=[sum(r['correct_actions'] for r in records),sum(r['target_actions'] for r in records)]
    result['usage']=usage(records)
    return result


def old_live(path):
    out=[]
    for r in map(json.loads,path.read_text().splitlines()):
        if r['condition']!='external_stack':continue
        calls=[]
        for c in r['calls']:
            try:action=json.loads(c['text'])
            except ValueError:action=None
            calls.append(dict(c,action=action))
        out.append(dict(id=r['id'],success=r['verified'],calls=calls,first_error=None if r['verified'] else len(calls)))
    return out


def null_summary(rows,teacher,full,live):
    lookup={r['id']:r for r in rows};p={};po={}
    for alphabet in sorted({r['alphabet'] for r in rows}):
        rs=[r for r in teacher if lookup[r['id']]['alphabet']==alphabet]
        p[alphabet]=mean(r['correct'] for r in rs)
        po[alphabet]={op:mean(r['correct'] for r in rs if r['expected']['op']==op) for op in {r['expected']['op'] for r in rs}}
    full={r['id']:r for r in full};live={r['id']:r for r in live};predictions=[]
    for r in rows:
        gold=reference(r);q=1.
        for action in gold:q*=po[r['alphabet']][action['op']]
        predictions.append(dict(id=r['id'],alphabet=r['alphabet'],depth=r['depth'],length=len(gold),iid=p[r['alphabet']]**len(gold),action_stratified=q,full=full[r['id']]['success'],live=live[r['id']]['success']))
    cells=[]
    for alphabet in p:
        for n in sorted({r['length'] for r in predictions if r['alphabet']==alphabet}):
            rs=[r for r in predictions if r['alphabet']==alphabet and r['length']==n]
            cells.append(dict(alphabet=alphabet,length=n,total=len(rs),**{k:mean(r[k] for r in rs) for k in ('iid','action_stratified','full','live')}))
    return dict(p_by_alphabet=p,p_by_action=po,cases=predictions,by_length=cells,
        overall={k:mean(r[k] for r in predictions) for k in ('iid','action_stratified','full','live')})


def injection_summary(records,rows):
    result=case_summary(records,rows)
    result['post_prefix_error_rate']=1-result['action_accuracy'][0]/result['action_accuracy'][1]
    return result


def summarize_tiny(data):
    results=[];rows=data['instances'];lookup={r['id']:r for r in rows}
    for run in data['results']:
        rs=run['records'];cells={}
        for d in sorted({r['depth'] for r in rows}):
            selected=[r for r in rs if lookup[r['id']]['depth']==d]
            cells[str(d)]={k:[sum(r[k] for r in selected),len(selected)] for k in ('success','teacher_success')}
        results.append(dict(seed=run['seed'],full=[sum(r['success'] for r in rs),len(rs)],
            oracle_sequence=[sum(r['teacher_success'] for r in rs),len(rs)],
            teacher_actions=[sum(r['teacher_action_accuracy'][i] for r in rs) for i in (0,1)],by_depth=cells))
    return results



def tiny_x5(data,injected):
    results=[];lookup={r['id']:r for r in data['instances']}
    for run in data['results']:
        rs=run['records'];correct=sum(r['teacher_action_accuracy'][0] for r in rs);n=sum(r['teacher_action_accuracy'][1] for r in rs);p=correct/n
        eos=sum(r['teacher_prediction'][-1]==r['target'][-1] for r in rs)/len(rs)
        by_length=[]
        for length in sorted({len(r['target'])-1 for r in rs}):
            selected=[r for r in rs if len(r['target'])-1==length]
            by_length.append(dict(length=length,total=len(selected),predicted=p**length*eos,measured=mean(r['success'] for r in selected)))
        conditions={}
        for inj in injected['results']:
            if inj['seed']!=run['seed']:continue
            cases=inj['records'];ok=sum(r['correct_actions'] for r in cases);total=sum(r['target_actions'] for r in cases)
            conditions[str(inj['k'])]=dict(exact=[sum(r['success'] for r in cases),len(cases)],action_accuracy=[ok,total],post_prefix_error_rate=1-ok/total)
        results.append(dict(seed=run['seed'],p=p,eos_accuracy=eos,predicted=mean(p**(len(r['target'])-1)*eos for r in rs),measured=mean(r['success'] for r in rs),by_length=by_length,injection=conditions))
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--x12',type=Path,default=Path('build/a6-x12-2026-10-04'))
    p.add_argument('--tiny',type=Path,default=Path('build/a6-tiny-trace-2026-10-04'));p.add_argument('--output',type=Path,default=Path('docs/results'));a=p.parse_args()
    rows=json.loads((a.input/'instances.json').read_text());subset=[r for r in rows if r['depth']>=4];runs={};s4=[];s5=[]
    for size in ('14b','1.7b'):
        old=Path('build/a6-e2'+('-14b' if size=='14b' else '')+'-2026-10-03')
        for condition in ('zeroshot','lora'):
            name=size+'_'+condition
            full=json.loads((a.input/'x4'/name/'full.json').read_text())
            live=old_live(old/condition/'evaluation/records.jsonl')
            teacher=json.loads(((a.x12/'x2'/f'{size}.json') if condition=='lora' else a.input/'x4'/name/'teacher.json').read_text())
            for r in teacher:
                r.setdefault('historical_first_error',None);r.setdefault('after_historical_error',False)
            injection={str(k):json.loads((a.input/'x5'/name/f'k{k}.json').read_text()) for k in (0,1,2,4)}
            runs[name]=dict(full=full,teacher=teacher,live=live,injection=injection)
            s4.append(dict(model=name,full=case_summary(full,rows),oracle=teacher_summary(teacher,rows),live=execution_summary(live,rows)))
            s5.append(dict(model=name,independence=null_summary(rows,teacher,full,live),injection={k:injection_summary(v,subset) for k,v in injection.items()}))
    rankings={mode:dict(zip([r['model'] for r in s4],ranks([-r[mode]['held_out']['total'][0] if mode!='oracle' else -r[mode]['oracle_trace']['held_out']['total'][0] for r in s4]))) for mode in ('full','oracle','live')}
    inversions=[]
    for i,left in enumerate(s4):
        for right in s4[i+1:]:
            differences={m:rankings[m][left['model']]-rankings[m][right['model']] for m in rankings}
            if min(differences.values())<0<max(differences.values()):inversions.append(dict(left=left['model'],right=right['model'],rank_differences=differences))
    tiny=json.loads((a.tiny/'evaluation.json').read_text())
    consistency={}
    for name,run in runs.items():
        teacher={(r['id'],r['call']):r for r in run['teacher']};different=[];compared=0
        for r in run['live']:
            for i,c in enumerate(r['calls'],1):
                t=teacher[r['id'],i];assert t['model_prompt']==c['model_prompt'];compared+=1
                if t['action']!=c['action']:different.append(dict(id=r['id'],call=i,teacher=t['action'],live=c['action']))
        consistency[name]=dict(compared_calls=compared,different_calls=len(different),differences=different)
    meta=dict(batch_consistency=consistency,plan=json.loads((a.input/'plan.json').read_text()),completion=json.loads((a.input/'completion.json').read_text()),rankings=rankings,strict_rank_inversions=inversions,
        correction='Primary full-trace prompts explicitly specify action-object schemas. Earlier x45 run omitted schemas; stopped and excluded; kept only as a diagnostic.',
        tiny=dict(plan=json.loads((a.tiny/'plan.json').read_text()),checkpoints=json.loads((a.tiny/'frozen.json').read_text()),summary=summarize_tiny(tiny)))
    tiny_injected=json.loads((a.tiny/'injection.json').read_text())
    audit=dict(instances=rows,runs=runs,tiny=tiny,tiny_injection=tiny_injected)
    save(a.output/'2026-10-04-x4',s4,audit,meta)
    save(a.output/'2026-10-04-x5',s5,audit,dict(plan=meta['plan'],tiny=tiny_x5(tiny,tiny_injected),interpretation='Descriptive pooled p^n null fitted to the same oracle cases; action-stratified sensitivity. Forced wrong prefixes are a separate history-continuation intervention, not live state corruption.'))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for j,mode in enumerate(('full','oracle','live')):
        vals=[100*(r[mode]['held_out']['total'][0] if mode!='oracle' else r[mode]['oracle_trace']['held_out']['total'][0])/92 for r in s4]
        ax.bar([i+(j-1)*.25 for i in range(4)],vals,width=.25,label=mode)
    ax.set_xticks(range(4),[r['model'] for r in s4]);ax.set_ylabel('Exact cases (%)');ax.set_ylim(0,105);ax.legend();ax.set_title('X4: same Qwen checkpoint, different execution interfaces')
    fig.tight_layout();fig.savefig(a.output/'2026-10-04-x4/plot.png',dpi=160)
    fig,axes=plt.subplots(2,3,figsize=(13,7.5));axes=list(axes.flat)
    for ax,r in zip(axes,s5):
        cases=r['independence']['cases'];lengths=sorted({c['length'] for c in cases})
        for key,style in [('iid','--'),('action_stratified',':'),('live','-o'),('full','-s')]:
            ax.plot(lengths,[100*mean(c[key] for c in cases if c['length']==n) for n in lengths],style,label=key,markersize=3)
        ax.set(title=r['model'],xlabel='Correct trace length',ylabel='Success (%)',ylim=(-3,103))
    axes[0].legend(fontsize=7)
    for r in s5:
        axes[4].plot([0,1,2,4],[100*r['injection'][str(k)]['post_prefix_error_rate'] for k in (0,1,2,4)],marker='o',label=r['model'])
    for r in tiny_x5(tiny,tiny_injected):
        axes[5].plot([0,1,2,4],[100*r['injection'][str(k)]['post_prefix_error_rate'] for k in (0,1,2,4)],marker='o',label='tiny seed '+str(r['seed']))
    for ax,title in [(axes[4],'Qwen: corrupted history'),(axes[5],'Separate tiny trace controls')]:
        ax.set(title=title,xlabel='Flipped PUSH values (four-action prefix)',ylabel='Subsequent action error (%)',xticks=[0,1,2,4],ylim=(-3,103));ax.legend(fontsize=7)
    fig.suptitle('X5: descriptive independence nulls and history continuation');fig.tight_layout();fig.savefig(a.output/'2026-10-04-x5/plot.png',dpi=160)
    print(json.dumps(dict(x4=[dict(model=r['model'],full=r['full']['held_out']['total'],oracle=r['oracle']['oracle_trace']['held_out']['total'],live=r['live']['held_out']['total']) for r in s4],tiny=meta['tiny']['summary'],inversions=inversions),indent=2))


if __name__=='__main__':main()
