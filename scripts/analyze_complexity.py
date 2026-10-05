"""Summarize retrieved complexity artifacts; no scoring changes or repair."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import re
from pathlib import Path


def simple_accepting_derivation(row, text):
    """Conservative post-hoc witness check: only leftmost TRY* followed by ACC."""
    try:actions=json.loads(text)
    except ValueError:return False
    if not isinstance(actions,list) or not actions or actions[-1]!={'op':'ACC'}:return False
    grammar=row['grammar'];form=[grammar['start']]
    for action in actions[:-1]:
        if not isinstance(action,dict) or set(action)!={'op','production'} or action['op']!='TRY':return False
        pid=action['production']
        if not isinstance(pid,str) or not re.fullmatch(r'p(0|[1-9][0-9]*)',pid):return False
        index=int(pid[1:])
        if index>=len(grammar['productions']):return False
        rule=grammar['productions'][index]
        position=next((i for i,symbol in enumerate(form) if symbol in grammar['nonterminals']),None)
        if position is None or form[position]!=rule['lhs']:return False
        form=form[:position]+rule['rhs']+form[position+1:]
    return form==row['input']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory
    conditions=('original','clarified','clarified_fewshot','one_rule_specialized')
    rows={r['task_id']:r for r in map(json.loads,(root/'instances.jsonl').read_text().splitlines())}
    result={'runtime':json.loads((root/'runtime.json').read_text()),
            'primitive':json.loads((root/'primitive_summary.json').read_text()),'conditions':{},'batch_consistency':{},
            'posthoc_simple_derivations':{}}
    baseline={};parity={}
    def unique_metrics(records):
        groups=defaultdict(list)
        for r in records:
            row=rows[r['task_id']]
            key=json.dumps({k:row[k] for k in ('grammar','input','search_config')},sort_keys=True)
            groups[key].append(r)
        return {'total':len(groups),'mean_success_rate':sum(sum(r['success'] for r in g)/len(g) for g in groups.values())/len(groups),
                'discordant_duplicate_groups':sum(len({r['success'] for r in g})>1 for g in groups.values())}
    for condition in conditions:
        records=list(map(json.loads,(root/condition/'records.jsonl').read_text().splitlines()))
        cells=defaultdict(list)
        checked=0
        for record in records:
            row=rows[record['task_id']]
            key=(record['task_id'],record['mode'])
            cells[(record['family'],record['difficulty']['level'],record['mode'])].append(record)
            if record['mode']!='external_state':
                observations=[c['model_prompt'].split('Observation:\n')[-1] for c in record['calls']]
                budgets=[c['max_tokens'] for c in record['calls']]
                if condition=='original':baseline[key]=(observations,budgets)
                else:
                    assert baseline[key]==(observations,budgets),(condition,key)
                    checked+=len(observations)
            if record['mode']=='full_trace':
                obs=json.loads(record['calls'][0]['model_prompt'].split('Observation:\n')[-1])
                assert obs['grammar']==row['grammar'] and obs['input']==row['input']
        parity[condition]=checked
        positive=[r for r in records if r['mode']=='full_trace' and r['target_answer']=='accept']
        witnesses=[r for r in positive if simple_accepting_derivation(rows[r['task_id']],r['calls'][0]['text'])]
        result['posthoc_simple_derivations'][condition]={'positive_cases':len(positive),
            'verified_witnesses':len(witnesses),'failed_canonical_but_verified':sum(not r['success'] for r in witnesses),
            'task_ids_failed_canonical_but_verified':[r['task_id'] for r in witnesses if not r['success']]}
        by_case=defaultdict(dict)
        for r in records:by_case[r['task_id']][r['mode']]=r
        compared=0;differences=[]
        for task,modes in by_case.items():
            for i,(teacher,live) in enumerate(zip(modes['oracle_step']['calls'],modes['external_state']['calls'])):
                assert teacher['model_prompt']==live['model_prompt']
                compared+=1
                try:same=json.loads(teacher['text'])==json.loads(live['text'])
                except ValueError:same=teacher['text']==live['text']
                if not same:differences.append({'task_id':task,'step':i,'teacher':teacher['text'],'live':live['text']})
        result['batch_consistency'][condition]={'compared':compared,'different':len(differences),'examples':differences}
        result['conditions'][condition]=[{'family':family,'level':level,'mode':mode,
            'correct':sum(r['success'] for r in rs),'total':len(rs),
            'unique_cases':unique_metrics(rs),
            'by_label':{label:{'correct':sum(r['success'] for r in rs if r['target_answer']==label),
                        'total':sum(r['target_answer']==label for r in rs)} for label in ('accept','reject')},
            'correct_actions':sum(r['correct_actions'] for r in rs),
            'target_actions':sum(r['target_actions'] for r in rs),
            'failures':dict(Counter(r['failure'] for r in rs if r['failure'])),
            'calls':sum(len(r['calls']) for r in rs),
            'truncated':sum(c['finish_reason']=='length' for r in rs for c in r['calls']),
            'completion_tokens':sum(c['usage']['completion_tokens'] for r in rs for c in r['calls'])}
            for (family,level,mode),rs in sorted(cells.items())]
    result['matched_observation_and_budget_calls']=parity
    membership=root/'membership'
    if (membership/'summary.json').exists():
        result['membership']=json.loads((membership/'summary.json').read_text())
        member_rows=list(map(json.loads,(membership/'instances.jsonl').read_text().splitlines()))
        assert list(rows.values())==member_rows[:len(rows)]
        replay=json.loads((membership/'batch_replay.json').read_text())
        result['inference_replay']={}
        for run in replay['runs']:
            index=replay['step'] if run['label']=='teacher_batch' else 0
            responses=[batch[index] for batch in run['outputs']]
            assert all(r['model_prompt']==replay['prompt'] for r in responses)
            result['inference_replay'][run['label']]=dict(Counter(r['text'] for r in responses))
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Qwen3-14B increasing complexity','',
        'Strict completed cases / total. Each grammar cell contains four accept/reject pairs.',
        'Levels are family-specific parameters, not a common difficulty unit.','']
    for mode in ('full_trace','external_state','oracle_step'):
        lines+=['## '+mode,'','| Family / condition | 1 | 2 | 4 | 8 | 16 |','|---|---:|---:|---:|---:|---:|']
        for family in ('terminal_length','chain_depth','alternatives','sequence','recursion','nesting'):
            for condition,cells in result['conditions'].items():
                selected={c['level']:c for c in cells if c['family']==family and c['mode']==mode}
                if selected:lines+=['| '+family+' / '+condition+' | '+' | '.join(f"{selected[n]['correct']}/{selected[n]['total']}" for n in (1,2,4,8,16))+' |']
        lines+=['']
    if 'membership' in result:
        lines+=['## Direct membership','','| Family | 1 | 2 | 4 | 8 | 16 |','|---|---:|---:|---:|---:|---:|']
        for family in sorted({r['family'] for r in result['membership']}):
            selected={r['level']:r for r in result['membership'] if r['family']==family}
            lines+=['| '+family+' | '+' | '.join(f"{selected[n]['correct']}/{selected[n]['total']}" if n in selected else '—' for n in (1,2,4,8,16))+' |']
    (root/'tables.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'original':'#9b9b9b','clarified':'#0072B2','clarified_fewshot':'#D55E00','one_rule_specialized':'#009E73'}
    labels={'original':'Original','clarified':'Clarified','clarified_fewshot':'Clarified + examples','one_rule_specialized':'One-rule instructions'}
    for mode in ('full_trace','external_state'):
        fig,axes=plt.subplots(2,3,figsize=(12,6.7),sharey=True)
        for ax,family in zip(axes.flat,('terminal_length','chain_depth','alternatives','sequence','recursion','nesting')):
            for condition,cells in result['conditions'].items():
                selected=sorted([c for c in cells if c['family']==family and c['mode']==mode],key=lambda c:c['level'])
                if selected:ax.plot([c['level'] for c in selected],[100*c['unique_cases']['mean_success_rate'] for c in selected],'-o',color=colors[condition],label=labels[condition],lw=2,ms=4)
            ax.set_title(family.replace('_',' ').capitalize(),loc='left',fontsize=11)
            ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],['1','2','4','8','16']);ax.set_ylim(-4,104);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.18)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('Qwen3-14B: '+('complete traces' if mode=='full_trace' else 'execution with exact external state'),fontsize=15,x=.055,ha='left')
        fig.supylabel('Successful complete executions (%) — unique cases',fontsize=11)
        fig.supxlabel('Complexity level (different meaning in each panel)',y=.065,fontsize=11)
        handles,legendlabels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legendlabels,loc='lower center',ncol=4,frameon=False,fontsize=9)
        fig.tight_layout(rect=(.015,.085,1,.94));fig.savefig(root/(mode+'.png'),dpi=160,facecolor='white');plt.close(fig)
    print(json.dumps({condition:{mode:sum(c['correct'] for c in cells if c['mode']==mode) for mode in ('full_trace','external_state','oracle_step')} for condition,cells in result['conditions'].items()},indent=2))


if __name__=='__main__':main()
