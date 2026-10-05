"""Export X1/X2/X6 audits and recomputed measurements without inference."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from abstractgym.a6 import action_id,summarize,digest


def normalized(records,rows):
    lookup={r['id']:r for r in rows};out=[]
    for r in records:
        calls=[]
        for c in r['calls']:
            gold=action_id(c['expected'],lookup[r['id']])
            try:pred=action_id(c['action'],lookup[r['id']])
            except (ValueError,TypeError,KeyError):pred=-1
            if not c['correct'] and pred==gold:pred=-1
            calls.append(dict(expected=gold,prediction=pred,correct=c['correct']))
        out.append(dict(id=r['id'],success=r['success'],first_error=r['first_error'],calls=calls))
    return out


def execution_summary(records,rows):
    result=summarize(rows,normalized(records,rows))
    calls=[c for r in records for c in r['calls']]
    result['invalid_action_calls']=sum(not isinstance(c['action'],dict) for c in calls)
    result['push_value_errors']=sum(c['expected']['op']=='PUSH' and isinstance(c['action'],dict) and c['action'].get('op')=='PUSH' and c['action'].get('value')!=c['expected']['value'] for c in calls)
    result['usage']={k:sum(c.get(k,c.get('usage',{}).get(k,0)) for c in calls) for k in ('prompt_tokens','completion_tokens')}
    return result


def teacher_summary(records,rows):
    lookup={r['id']:r for r in rows};result={}
    for name,alphabet in [('letters','abcde'),('brackets','([)]#')]:
        selected=[r for r in records if lookup[r['id']]['alphabet']==alphabet]
        cell=lambda rs:[sum(r['correct'] for r in rs),len(rs)]
        ids={r['id'] for r in selected}
        result[name]=dict(action_accuracy=cell(selected),whole_trace=[sum(all(r['correct'] for r in selected if r['id']==i) for i in ids),len(ids)],
            by_action={op:cell([r for r in selected if r['expected']['op']==op]) for op in ('PUSH','CENTER','POP','ACCEPT','REJECT')},
            by_depth={str(d):cell([r for r in selected if lookup[r['id']]['depth']==d]) for d in sorted({lookup[r['id']]['depth'] for r in selected})},
            after_call1_failure=cell([r for r in selected if r['historical_first_error']==1 and r['call']>1]),
            after_any_historical_failure=cell([r for r in selected if r['after_historical_error']]))
    result['action_accuracy']=[sum(r['correct'] for r in records),len(records)]
    grouped=[]
    for row in rows:
        calls=[r for r in records if r['id']==row['id']]
        errors=[r['call'] for r in calls if not r['correct']]
        grouped.append(dict(id=row['id'],success=not errors,calls=calls,first_error=min(errors) if errors else None))
    result['oracle_trace']=execution_summary(grouped,rows)
    return result


def save(root,summary,audit,metadata):
    root.mkdir(parents=True,exist_ok=True)
    for name,value in [('summary.json',summary),('metadata.json',metadata)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    (root/'audit.json.gz').write_bytes(gzip.compress(json.dumps(audit,separators=(',',':'),sort_keys=True).encode(),mtime=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--x12',type=Path);p.add_argument('--x6',type=Path);p.add_argument('--output',type=Path,default=Path('docs/results'));args=p.parse_args()
    if args.x6:
        root=args.x6;data=json.loads((root/'evaluation.json').read_text());rows=data['instances'];summaries=[]
        for run in data['results']:
            s=execution_summary(run['records'],rows);s.update(seed=run['seed'],condition='tiny_raw_autoregressive',
                training=json.loads((root/f"train-{run['seed']}.json").read_text()),
                exhaustive={a:[sum(r['correct'] for r in run['exhaustive'] if r['alphabet']==a),36] for a in ('uvwxy','abcde','([)]#')})
            summaries.append(s)
        save(args.output/'2026-10-04-x6',summaries,data,dict(plan=json.loads((root/'plan.json').read_text()),checkpoints=json.loads((root/'frozen.json').read_text())))
    if args.x12:
        root=args.x12;rows=json.loads((root/'instances.json').read_text());plan=json.loads((root/'plan.json').read_text())
        assert digest(rows)==plan['held_out_hash']
        meta=dict(plan=plan,completion=json.loads((root/'completion.json').read_text()),models={n:json.loads((root/n/'metadata.json').read_text()) for n in ('14b','1.7b')})
        x1={name:json.loads((root/'x1'/f'{name}.json').read_text()) for name in ('constrained_json','role_id')}
        save(args.output/'2026-10-04-x1',[dict(condition=k,**execution_summary(v,rows)) for k,v in x1.items()],dict(instances=rows,conditions=x1),meta)
        x2={name:json.loads((root/'x2'/f'{name}.json').read_text()) for name in ('14b','1.7b')}
        save(args.output/'2026-10-04-x2',[dict(condition=k,**teacher_summary(v,rows)) for k,v in x2.items()],dict(instances=rows,conditions=x2),meta)


if __name__=='__main__':main()
