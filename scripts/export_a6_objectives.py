"""X7/X8 summaries with frozen per-cell decision baselines and cost accounting."""
import argparse
from collections import Counter
import json
from pathlib import Path
from abstractgym.controller import teacher_trajectory
from export_a6_followups import execution_summary,save
from export_a6_x45 import old_live,usage

MODES=('full_trace','oracle_step','external_state')


def parse(text):
    try:return json.loads(text)
    except ValueError:return None


def complexity_summary(records,rows):
    lookup={r['task_id']:r for r in rows};result={}
    def metrics(rs):
        n=len(rs);positive=sum(r['target_answer']=='accept' for r in rs);correct=sum(r['success'] for r in rs)
        groups={}
        for r in rs:
            row=lookup[r['task_id']];key=json.dumps({k:row[k] for k in ('grammar','input','search_config')},sort_keys=True)
            groups.setdefault(key,[]).append(r['success'])
        return dict(exact=[correct,n],label_blind=dict(always_accept=[positive,n],always_reject=[n-positive,n]),
            interpretation='beats both label-blind baselines' if correct>max(positive,n-positive) else 'chance / label-blind' if correct==max(positive,n-positive) else 'below label-blind',
            unique_cases=len(groups),unique_case_mean=sum(sum(g)/len(g) for g in groups.values())/len(groups),
            action_accuracy=[sum(r['correct_actions'] for r in rs),sum(r['target_actions'] for r in rs)],
            failures=dict(Counter(r['failure'] for r in rs if r['failure'])),usage=usage([c for r in rs for c in r['calls']]))
    for mode in MODES:
        rs=[r for r in records if r['mode']==mode]
        result[mode]=dict(overall=metrics(rs),by_family={family:metrics([r for r in rs if r['family']==family]) for family in sorted({r['family'] for r in rs})},
            cells=[dict(family=family,level=level,**metrics([r for r in rs if r['family']==family and r['difficulty']['level']==level])) for family in sorted({r['family'] for r in rs}) for level in sorted({r['difficulty']['level'] for r in rs if r['family']==family})])
    ops={}
    for r in records:
        if r['mode']!='oracle_step':continue
        gold=teacher_trajectory(lookup[r['task_id']])
        assert len(r['calls'])==len(gold)
        for c,g in zip(r['calls'],gold):
            op=g['action']['op'];counts=ops.setdefault(op,[0,0]);counts[0]+=parse(c.get('text',''))==g['action'];counts[1]+=1
    result['teacher_by_action']=ops
    return result


def direct_summary(records,rows):
    lookup={r['id']:r for r in rows};out={};blind={}
    for alphabet in sorted({r['alphabet'] for r in rows}):
        cells={};blind[alphabet]={}
        for depth in sorted({r['depth'] for r in rows if r['alphabet']==alphabet}):
            selected=[r for r in records if lookup[r['id']]['alphabet']==alphabet and lookup[r['id']]['depth']==depth]
            cells[str(depth)]=[sum(r['verified'] for r in selected),len(selected)]
            positive=sum(lookup[r['id']]['accept'] for r in selected)
            blind[alphabet][str(depth)]=dict(always_accept=[positive,len(selected)],always_reject=[len(selected)-positive,len(selected)])
        out[alphabet]=cells
    return dict(exact=[sum(r['verified'] for r in records),len(records)],by_label={str(label):[sum(r['verified'] for r in records if lookup[r['id']]['accept']==label),sum(lookup[r['id']]['accept']==label for r in records)] for label in (True,False)},by_cell=out,usage=usage(records),
        label_blind=dict(always_accept=[sum(r['accept'] for r in rows),len(rows)],always_reject=[sum(not r['accept'] for r in rows),len(rows)],by_cell=blind))


def main():
    p=argparse.ArgumentParser();p.add_argument('--x7',type=Path);p.add_argument('--x8',type=Path);p.add_argument('--output',type=Path,default=Path('docs/results'));a=p.parse_args()
    import matplotlib.pyplot as plt
    for name,root in (('x7',a.x7),('x8',a.x8)):
        if root is None:continue
        rows=json.loads((root/'instances.json').read_text());meta={n:json.loads((root/(n+'.json')).read_text()) for n in ('plan','completion')}
        meta['training']={n:json.loads((root/'lora'/(n+'.json')).read_text()) for n in ('training','reload_parity','initialization_parity','checkpoint_hashes')}
        if name=='x7':
            baseline=Path('build/cfg_controller_v1/ray/complexity/original/records.jsonl')
            runs={'zeroshot':[json.loads(l) for l in baseline.read_text().splitlines()], 'dfs_lora':[json.loads(l) for l in (root/'evaluation/records.jsonl').read_text().splitlines()]}
            summary={k:complexity_summary(v,rows) for k,v in runs.items()}
            save(a.output/'2026-10-04-x7',summary,dict(instances=rows,runs=runs,membership=json.loads((root/'membership.json').read_text())),meta)
            fig,axes=plt.subplots(2,3,figsize=(11,6),sharey=True)
            for ax,family in zip(axes.flat,summary['zeroshot']['external_state']['by_family']):
                for model,s in summary.items():
                    cells=[r for r in s['external_state']['cells'] if r['family']==family]
                    ax.plot([r['level'] for r in cells],[100*r['exact'][0]/r['exact'][1] for r in cells],marker='o',label=model)
                ax.set_title(family);ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.set_ylim(-3,103)
            axes.flat[0].legend(fontsize=8);fig.supylabel('Exact external-state execution (%)');fig.supxlabel('Family-specific complexity level');fig.suptitle('X7: development DFS-step SFT');fig.tight_layout();fig.savefig(a.output/'2026-10-04-x7/plot.png',dpi=160)
        else:
            old=Path('build/a6-e2-14b-2026-10-03');runs={}
            for condition,folder in [('zeroshot','zeroshot'),('step_lora','lora')]:
                f=old/folder/'evaluation/records.jsonl';records=[json.loads(l) for l in f.read_text().splitlines()]
                runs[condition]=dict(direct=[r for r in records if r['condition']=='grammar'],live=old_live(f))
            runs['membership_lora']={n:json.loads((root/(n+'.json')).read_text()) for n in ('direct','live')}
            summary={k:dict(direct=direct_summary(v['direct'],rows),live=execution_summary(v['live'],rows)) for k,v in runs.items()}
            save(a.output/'2026-10-04-x8',summary,dict(instances=rows,runs=runs),meta)
            fig,ax=plt.subplots(figsize=(8,4))
            for j,mode in enumerate(('direct','live')):
                vals=[100*(s[mode]['exact'][0] if mode=='direct' else s[mode]['held_out']['total'][0])/92 for s in summary.values()]
                ax.bar([i+(j-.5)*.3 for i in range(3)],vals,width=.3,label=mode)
            ax.set_xticks(range(3),list(summary));ax.axhline(50,color='gray',linestyle=':',label='Membership label-blind reference');ax.set_ylim(0,105);ax.set_ylabel('Correct cases (%)');ax.legend();ax.set_title('X8: answering versus executing after different SFT objectives');fig.tight_layout();fig.savefig(a.output/'2026-10-04-x8/plot.png',dpi=160)
        print(name,json.dumps({k:({m:v[m]['overall']['exact'] for m in MODES} if name=='x7' else dict(direct=v['direct']['exact'],live=v['live']['held_out']['total'])) for k,v in summary.items()}))


if __name__=='__main__':main()
