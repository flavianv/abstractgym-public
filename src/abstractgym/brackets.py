"""Small, independently verified nested-pair diagnostics (not general Dyck)."""
import itertools
import json
from abstractgym.cfg import Grammar, Production
from abstractgym.oracle import accepts


def suite():
    rows=[]
    for split,alphabet in [('development','uvwxy'),('test','abcde'),('test','([)]#')]:
        a,c,b,d,e=alphabet
        pairs={a:b,c:d}
        for depth in (0,1,2,3,4,8,16):
            patterns=sorted(set(tuple(p) for p in ([a]*depth,[c]*depth,[a,c]*(depth//2)+[a]*(depth%2),[c,a]*(depth//2)+[c]*(depth%2))))
            for k,opens in enumerate(patterns):
                good=[pairs[x] for x in reversed(opens)]
                bad=good.copy()
                if depth>=2 and len(set(bad))==2:
                    j=next(i for i in range(1,len(bad)) if bad[i]!=bad[0]);bad[0],bad[j]=bad[j],bad[0]
                    negative='order_only'
                elif depth:
                    bad[0]=d if bad[0]==b else b;negative='wrong_type'
                else:
                    bad=[b];negative='extra_close'
                grammar=Grammar('S',frozenset(['S']),frozenset(alphabet),tuple(Production('S',rhs) for rhs in ((a,'S',b),(c,'S',d),(e,))))
                for positive,closing in ((True,good),(False,bad)):
                    tokens=list(opens)+[e]+closing
                    assert accepts(grammar,tuple(tokens))==positive
                    rows.append(dict(id=f'{split}-{alphabet}-{depth}-{k}-{positive}',split=split,alphabet=alphabet,depth=depth,pairs=pairs,center=e,input=tokens,grammar=grammar.to_dict(),accept=positive,negative=None if positive else negative))
    return rows


def direct_prompt(row,explicit=False,witness=False):
    text='Does this context-free grammar generate exactly this input from S?\n'+json.dumps({'grammar':row['grammar'],'input':row['input']})+'\n'
    if explicit:
        text+='These rules create nested pairs around one center token. The closing tokens must match the opening tokens in REVERSE order, using the pair mapping '+json.dumps(row['pairs'])+'. The center token is '+json.dumps(row['center'])+'.\n'
    if witness:
        text+='Return only {"accept":true,"rules":[0,1,2]} with a valid leftmost derivation as zero-based production indices, or {"accept":false,"rules":[]}. The example indices illustrate the format only. Any valid derivation is allowed; no search order is imposed.'
    else:
        text+='Return only raw JSON {"accept":true} or {"accept":false}.'
    return text


def score_answer(row,text,witness=False):
    try: obj=json.loads(text)
    except (ValueError,TypeError): return {'schema':False,'decision':False,'verified':False}
    schema=isinstance(obj,dict) and type(obj.get('accept')) is bool and set(obj)==({'accept','rules'} if witness else {'accept'})
    if witness: schema=schema and isinstance(obj.get('rules'),list) and all(type(x) is int for x in obj['rules'])
    decision=bool(schema and obj['accept']==row['accept']);verified=decision
    if witness and schema:
        if obj['accept']:
            form=['S'];rules=row['grammar']['productions']
            for index in obj['rules']:
                if index<0 or index>=len(rules) or 'S' not in form: verified=False;break
                i=form.index('S');rule=rules[index]
                form=form[:i]+list(rule['rhs'])+form[i+1:]
            verified=verified and form==row['input']
        else: verified=verified and obj['rules']==[]
    return dict(schema=bool(schema),decision=decision,verified=bool(verified))


STACK_PROTOCOL='''Execute ONE transition of a nested-pair recognizer. Stack stores expected closing tokens, bottom first, top LAST. In opening mode: an opening token means PUSH its mapped closing token; center means CENTER (switch to closing mode); anything else means REJECT. In closing mode: if the current token equals the nonempty stack top, POP; otherwise REJECT. At end of input: ACCEPT exactly when mode is closing and stack is empty; otherwise REJECT. Output only {"op":"PUSH","value":"token"}, {"op":"CENTER"}, {"op":"POP"}, {"op":"ACCEPT"}, or {"op":"REJECT"}.\n'''


def expected(row,pos,mode,stack):
    if pos==len(row['input']):return {'op':'ACCEPT' if mode=='closing' and not stack else 'REJECT'}
    token=row['input'][pos]
    if mode=='opening':
        if token in row['pairs']:return {'op':'PUSH','value':row['pairs'][token]}
        if token==row['center']:return {'op':'CENTER'}
    elif stack and token==stack[-1]:return {'op':'POP'}
    return {'op':'REJECT'}


def stack_prompt(row,pos,mode,stack):
    return STACK_PROTOCOL+json.dumps(dict(pairs=row['pairs'],center=row['center'],mode=mode,stack=stack,end=pos==len(row['input']),token=row['input'][pos] if pos<len(row['input']) else None))


def advance(action,pos,mode,stack):
    stack=list(stack)
    if action['op']=='PUSH':stack.append(action['value'])
    elif action['op']=='POP':stack.pop()
    elif action['op']=='CENTER':mode='closing'
    return pos+1,mode,stack
