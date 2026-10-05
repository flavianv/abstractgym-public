"""Frozen X1 output contracts and X2 oracle observations, without inference."""
import json
from abstractgym.a6 import ACTIONS
from abstractgym.brackets import STACK_PROTOCOL,advance,expected,stack_prompt


def legal_actions(row):
    # Output domain only. Never filter using a state or the reference action.
    return [dict(op='PUSH',value=v) for v in row['pairs'].values()]+[
        dict(op=op) for op in ('CENTER','POP','ACCEPT','REJECT')]


def choices(row,interface):
    if interface=='role_id':return list(ACTIONS)
    assert interface=='constrained_json'
    return [json.dumps(a,separators=(',',':')) for a in legal_actions(row)]


def prompt(row,pos,mode,stack,interface='free_json'):
    original=stack_prompt(row,pos,mode,stack)
    if interface!='role_id':return original
    protocol=STACK_PROTOCOL.split('Output only ')[0]
    instruction='Output exactly one role ID: PUSH_0, PUSH_1, CENTER, POP, ACCEPT, or REJECT. '
    instruction+='PUSH_0 means PUSH value '+json.dumps(list(row['pairs'].values())[0])+'. '
    instruction+='PUSH_1 means PUSH value '+json.dumps(list(row['pairs'].values())[1])+'.\n'
    return protocol+instruction+original[len(STACK_PROTOCOL):]


def parse_action(text,row,interface='free_json'):
    if interface=='role_id':
        return legal_actions(row)[ACTIONS.index(text)] if text in ACTIONS else None
    try:return json.loads(text)
    except (ValueError,TypeError):return None


def oracle_states(row):
    result=[];pos,mode,stack=0,'opening',[]
    while True:
        action=expected(row,pos,mode,stack)
        result.append(dict(id=row['id'],call=len(result)+1,pos=pos,mode=mode,
                           stack=stack.copy(),expected=action))
        if action['op'] in ('ACCEPT','REJECT'):return result
        pos,mode,stack=advance(action,pos,mode,stack)
