"""X3 paired replication summaries, preserving every seed and map."""
import argparse
import json
from pathlib import Path
from statistics import mean
from export_a6_followups import execution_summary,save


def summarize_records(records,rows):
    result=execution_summary(records,rows);lookup={r['id']:r for r in rows};groups={};baselines={}
    for alphabet in sorted({r['alphabet'] for r in rows}):
        cells={};blind={};interpret={}
        for depth in sorted({r['depth'] for r in rows if r['alphabet']==alphabet}):
            selected=[r for r in records if lookup[r['id']]['alphabet']==alphabet and lookup[r['id']]['depth']==depth]
            total=len(selected);correct=sum(r['success'] for r in selected);positive=sum(lookup[r['id']]['accept'] for r in selected)
            cells[str(depth)]=[correct,total];blind[str(depth)]=dict(always_accept=[positive,total],always_reject=[total-positive,total])
            interpret[str(depth)]='chance / label-blind' if correct==max(positive,total-positive) else 'beats both label-blind baselines' if correct>max(positive,total-positive) else 'below label-blind'
        groups[alphabet]=dict(by_depth=cells,total=[sum(v[0] for v in cells.values()),sum(v[1] for v in cells.values())],interpretation=interpret)
        baselines[alphabet]=blind
    result['by_alphabet']=groups;result['label_blind']['by_cell']=baselines
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,default=Path('docs/results/2026-10-04-x3'));a=p.parse_args();root=a.input
    instances={name:json.loads((root/(name+'_instances.json')).read_text()) for name in ('original','fresh')};entries=[];runs=[]
    for size in ('14b','1.7b'):
        for seed in (17,18,19):
            for name,rows in instances.items():
                records=json.loads((root/size/f'seed-{seed}'/(name+'.json')).read_text())
                entries.append(dict(model=size,seed=seed,test_set=name,**summarize_records(records,rows)))
                runs.append(dict(model=size,seed=seed,test_set=name,records=records))
    training={f'{size}/seed-{seed}':json.loads((root/size/f'seed-{seed}'/'lora/training.json').read_text()) for size in ('14b','1.7b') for seed in (18,19)}
    metadata=dict(training=training,plan=json.loads((root/'plan.json').read_text()),completion=json.loads((root/'completion.json').read_text()),checkpoints=json.loads((root/'frozen_checkpoints.json').read_text()),aggregates=[])
    for size in ('14b','1.7b'):
        for name in instances:
            values=[r['held_out']['total'][0]/r['held_out']['total'][1] for r in entries if r['model']==size and r['test_set']==name]
            metadata['aggregates'].append(dict(model=size,test_set=name,mean=mean(values),minimum=min(values),maximum=max(values)))
    save(a.output,entries,dict(instances=instances,runs=runs),metadata)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,name in zip(axes,instances):
        for size in ('14b','1.7b'):
            selected=[r for r in entries if r['model']==size and r['test_set']==name]
            ax.plot([17,18,19],[100*r['held_out']['total'][0]/r['held_out']['total'][1] for r in selected],marker='o',label=size)
        ax.set_title(name);ax.set_xlabel('Training seed');ax.set_xticks([17,18,19]);ax.axhline(50,color='gray',linestyle=':',label='Membership label-blind reference');ax.set_ylim(0,105)
    axes[0].set_ylabel('Exact executions (%)');axes[1].legend(fontsize=8);fig.suptitle('X3: replication across seeds and a newly frozen test set');fig.tight_layout();fig.savefig(a.output/'plot.png',dpi=160)
    for e in entries:print(e['model'],e['seed'],e['test_set'],e['held_out']['total'])


if __name__=='__main__':main()
