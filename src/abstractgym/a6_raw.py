"""Raw-symbol observation/output encoding for the X6 ablation."""
import itertools
from abstractgym.a6 import DEV
from abstractgym.brackets import expected

# Fixed printable ASCII vocabulary, not selected using held-out labels.
VOCAB=('PAD','CLS','opening','closing','END','EMPTY','SEP','EOS',
       'PUSH','CENTER','POP','ACCEPT','REJECT')+tuple(chr(i) for i in range(32,127))


def observation(row,pos,mode,stack):
    values=['CLS',mode,row['input'][pos] if pos<len(row['input']) else 'END',
            stack[-1] if stack else 'EMPTY']
    for opening,closing in row['pairs'].items():values.extend([opening,closing])
    values.extend([row['center'],'SEP'])
    return [VOCAB.index(x) for x in values]


def target(action):
    values=[action['op']]+([action['value']] if action['op']=='PUSH' else [])+['EOS']
    return [VOCAB.index(x) for x in values]


def parse(ids):
    values=[VOCAB[i] for i in ids]
    if len(values)==3 and values[0]=='PUSH' and len(values[1])==1 and values[2]=='EOS':
        return dict(op='PUSH',value=values[1])
    if len(values)==2 and values[0] in ('CENTER','POP','ACCEPT','REJECT') and values[1]=='EOS':
        return dict(op=values[0])
    return None


def domain(row):
    """36 raw observations per supplied map; all in-domain top roles."""
    states=[]
    for mode,token,top in itertools.product(('opening','closing'),list(row['pairs'])+list(row['pairs'].values())+[row['center'],None],[None]+list(row['pairs'].values())):
        r=dict(row,input=[] if token is None else [token]);stack=[] if top is None else [top]
        states.append(dict(input_ids=observation(r,0,mode,stack),expected=expected(r,0,mode,stack)))
    return states
