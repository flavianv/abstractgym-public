"""X4 full traces and X5 controlled assistant-history prefixes."""
import json
from abstractgym.brackets import STACK_PROTOCOL
from abstractgym.a6_followup import oracle_states


def trace_prompt(row):
    protocol=STACK_PROTOCOL.split('Output only ')[0].replace('Execute ONE transition','Execute ALL transitions')
    schema=STACK_PROTOCOL.split('Output only ')[1].strip().removesuffix('.')
    return protocol+'Start with opening mode and an empty stack. Stop at ACCEPT or REJECT. Return one JSON array containing all actions, with raw JSON only. Each array element must be exactly one of '+schema+'. Do not add other keys.\n'+json.dumps({k:row[k] for k in ('pairs','center','input')})


def reference(row):return [s['expected'] for s in oracle_states(row)]


def injected_prefix(row,k):
    assert row['depth']>=4 and k in (0,1,2,4)
    prefix=[dict(a) for a in reference(row)[:4]]
    assert all(a['op']=='PUSH' for a in prefix)
    for i in range(k):
        prefix[i]['value']=next(v for v in row['pairs'].values() if v!=prefix[i]['value'])
    return '['+','.join(json.dumps(a,separators=(',',':')) for a in prefix)+','


def score(text,row,prefix='',injected=False):
    try:actions=json.loads(prefix+text)
    except ValueError:actions=None
    gold=reference(row);offset=4 if injected else 0
    valid=isinstance(actions,list) and all(isinstance(a,dict) for a in actions)
    suffix=actions[offset:] if valid else []
    target=gold[offset:]
    matches=[i<len(suffix) and suffix[i]==a for i,a in enumerate(target)]
    exact=valid and suffix==target
    first=next((i+1+offset for i,v in enumerate(matches) if not v),None)
    if not exact and first is None:first=len(gold)+1
    return dict(success=exact,valid_array=valid,first_error=first,
        correct_actions=sum(matches),target_actions=len(target),matches=matches,actions=actions,
        forced_prefix_length=offset)
