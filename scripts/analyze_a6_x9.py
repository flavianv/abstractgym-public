"""Paired answer/trace disagreement on the frozen complexity cases."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from export_a6_followups import save


def ranks(values):
    ordered=sorted(values)
    return [(ordered.index(x)+1+len(ordered)-ordered[::-1].index(x))/2 for x in values]


def correlation(a,b):
    if len(a)<2:return None
    a,b=ranks(a),ranks(b);aa=sum(a)/len(a);bb=sum(b)/len(b)
    numerator=sum((x-aa)*(y-bb) for x,y in zip(a,b))
    denominator=math.sqrt(sum((x-aa)**2 for x in a)*sum((y-bb)**2 for y in b))
    return numerator/denominator if denominator else None


def summarize(records):
    result=[]
    for model in sorted({r['model'] for r in records}):
        for mode in ('full_trace','oracle_step','external_state'):
            rs=[r for r in records if r['model']==model and r['mode']==mode]
            cells={f'answer_{a}_trace_{t}':[r['task_id'] for r in rs if r['answer_correct']==a and r['trace_correct']==t] for a in (False,True) for t in (False,True)}
            result.append(dict(model=model,mode=mode,total=len(rs),answer_correct=sum(r['answer_correct'] for r in rs),trace_correct=sum(r['trace_correct'] for r in rs),confusion={k:len(v) for k,v in cells.items()},case_ids=cells))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,default=Path('build/cfg_controller_v1/ray/complexity'))
    p.add_argument('--x7',type=Path);p.add_argument('--x10',type=Path);p.add_argument('--output',type=Path,default=Path('docs/results/2026-10-04-x9'));a=p.parse_args()
    rows=[json.loads(l) for l in (a.baseline/'instances.jsonl').read_text().splitlines()];ids={r['task_id'] for r in rows}
    sources=[('14b_zeroshot',a.baseline/'original/records.jsonl',a.baseline/'membership/records.jsonl')]
    if a.x7:sources.append(('14b_dfs_lora',a.x7/'evaluation/records.jsonl',a.x7/'membership.json'))
    if a.x10:sources.append(('14b_thinking_fixed_caps',a.x10/'complexity/records.jsonl',a.x10/'complexity/membership.json'))
    records=[];hashes={};answer_audit={}
    for model,trace_file,answer_file in sources:
        load=lambda f:[json.loads(l) for l in f.read_text().splitlines()] if f.suffix=='.jsonl' else json.loads(f.read_text())
        traces=load(trace_file);answers={r['task_id']:r for r in load(answer_file) if r['task_id'] in ids}
        answer_audit[model]=list(answers.values())
        assert set(answers)==ids and len(traces)==3*len(ids)
        assert {(r['task_id'],r['mode']) for r in traces}=={(i,m) for i in ids for m in ('full_trace','oracle_step','external_state')}
        for r in traces:records.append(dict(model=model,task_id=r['task_id'],mode=r['mode'],answer_correct=answers[r['task_id']]['success'],trace_correct=r['success']))
        for f in (trace_file,answer_file):hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    summary=summarize(records);correlations={}
    for entry in summary:
        entry['label_blind']={'always_accept':[sum(r['target_answer']=='accept' for r in rows),len(rows)],'always_reject':[sum(r['target_answer']=='reject' for r in rows),len(rows)],'by_cell':[]}
        for family in sorted({r['family'] for r in rows}):
            for level in sorted({r['difficulty']['level'] for r in rows if r['family']==family}):
                cell=[r for r in rows if r['family']==family and r['difficulty']['level']==level]
                positive=sum(r['target_answer']=='accept' for r in cell)
                entry['label_blind']['by_cell'].append(dict(family=family,level=level,always_accept=[positive,len(cell)],always_reject=[len(cell)-positive,len(cell)]))
    for mode in ('full_trace','oracle_step','external_state'):
        rs=[r for r in summary if r['mode']==mode]
        correlations[mode]=correlation([r['answer_correct']/r['total'] for r in rs],[r['trace_correct']/r['total'] for r in rs])
    save(a.output,summary,dict(records=records,instances=rows,answers=answer_audit),dict(source_hashes=hashes,rank_correlations=correlations,
        interpretation='Separate direct-answer and canonical trace invocations on identical cases; systems/configurations, not external benchmark rankings; null correlation means fewer than two systems or a constant ranking.'))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));rs=[r for r in summary if r['mode']=='full_trace'];bottom=[0]*len(rs)
    for key,label in [('answer_True_trace_True','Both correct'),('answer_True_trace_False','Answer only'),('answer_False_trace_True','Trace only'),('answer_False_trace_False','Neither')]:
        values=[r['confusion'][key] for r in rs];ax.bar(range(len(rs)),values,bottom=bottom,label=label);bottom=[x+y for x,y in zip(bottom,values)]
    names={'14b_zeroshot':'Base','14b_dfs_lora':'DFS step LoRA','14b_thinking_fixed_caps':'Thinking, fixed caps'}
    ax.set_xticks(range(len(rs)),[names[r['model']] for r in rs],fontsize=9);ax.set_ylabel('Frozen complexity cases');ax.set_title('X9: answer correctness versus exact canonical full trace');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(a.output/'plot.png',dpi=160)
    for row in summary:print(row['model'],row['mode'],row['confusion'])


if __name__=='__main__':main()
