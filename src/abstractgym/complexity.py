"""Paired diagnostic ladders; fixed structures, independently checked labels."""
from __future__ import annotations
import itertools
import json
import random
from abstractgym.cfg import Grammar, Production
from abstractgym.controller import PROTOCOL, teacher_trajectory
from abstractgym.oracle import accepts
from abstractgym.schema import stable_id

LEVELS = (1, 2, 4, 8, 16)
FAMILIES = ('terminal_length', 'chain_depth', 'alternatives', 'sequence', 'recursion', 'nesting')


def build_complexity_suite(levels=LEVELS, variants=4):
    if not levels or any(n < 1 for n in levels) or not 1 <= variants <= 24:
        raise ValueError('positive levels and 1..24 variants required')
    alphabets=list(itertools.permutations('abcd'))
    random.Random(71).shuffle(alphabets)
    rows=[]
    for level in levels:
        for family in FAMILIES:
            for variant,(a,b,c,d) in enumerate(alphabets[:variants]):
                pair=stable_id('complexity_pair',[family,level,a,b,c,d])
                for positive in (True,False):
                    end=a if positive else d
                    tokens=[a]
                    if family=='terminal_length':
                        tokens=[a]*level
                        rules=[('S',tuple([a]*(level-1)+[end]))]
                    elif family=='chain_depth':
                        names=['S']+[f'N{i}' for i in range(1,level+1)]
                        rules=[(x,(y,)) for x,y in zip(names,names[1:])]+[(names[-1],(end,))]
                    elif family=='alternatives':
                        rules=[('S',tuple([b]*i)) for i in range(1,level)]+[('S',(end,))]
                    elif family=='sequence':
                        tokens=[a]*level
                        rules=[('S',tuple(['A']*level)),('A',(end,))]
                    elif family=='recursion':
                        tokens=[a]*(level-1)+[c]
                        rules=[('S',(a,'S')),('S',(c if positive else d,))]
                    else:
                        tokens=[a]*level+[c]+[b]*level
                        rules=[('S',(a,'S',b)),('S',(c if positive else d,))]
                    grammar=Grammar('S',frozenset(lhs for lhs,_ in rules),frozenset((a,b,c,d)),
                                    tuple(Production(lhs,rhs) for lhs,rhs in rules))
                    assert accepts(grammar,tuple(tokens))==positive
                    row={'task_id':stable_id('complexity',[pair,positive]),'pair_id':pair,'split':'test',
                         'family':family,'depth_band':f'level_{level:02}',
                         'structure_id':f'{family}_{level}', 'grammar':grammar.to_dict(),
                         'input':tokens,'target_answer':'accept' if positive else 'reject',
                         'search_config':{'max_steps':512,'max_depth':level+4},
                         'difficulty':{'level':level,'variant':variant,'string_length':len(tokens),
                                       'rule_count':len(rules)}}
                    steps=teacher_trajectory(row)
                    row['difficulty'].update(search_steps=len(steps),
                        backtracks=sum(s['action']['op']=='BT' for s in steps),
                        derivation_depth=max(len(s['observation']['stack'])-1 for s in steps))
                    rows.append(row)
    return rows


CLARIFIED_PROTOCOL='''Execute deterministic leftmost depth-first search using the observation.
The stack is ordered root first, current node LAST. Only stack[-1] is the current
frame. A form is a list of grammar symbols. The empty terminal prefix matches
any input. Production p0 is productions[0], p1 is productions[1], and so on.

Choose the next action by the following priority:
1. If the current frame phase is "await_bt", output {"op":"BT"}. Do not inspect
   the failed child or prune the parent. BT changes this frame to "expanding".
2. If phase is "entered", classify the CURRENT FORM in this order:
   a. Depth len(stack)-1 > max_depth: {"op":"CUT","reason":"max_depth"}.
   b. Its terminal prefix BEFORE its first nonterminal is not a prefix of input:
      {"op":"PRN","reason":"pref"}. An empty prefix never fails this test.
   c. Total terminal symbols anywhere in the form exceeds input length:
      {"op":"PRN","reason":"long"}.
   d. No nonterminals and form equals input: {"op":"ACC"}.
   e. No nonterminals and form differs from input: {"op":"REJ","reason":"term"}.
   f. Leftmost nonterminal has no productions: {"op":"REJ","reason":"no_prod"}.
   If none applies, proceed to step 3.
3. For phase "expanding", skip classification and proceed directly here.
   Find the leftmost nonterminal in the current form. Scan productions in their
   given array order for rules with that lhs whose p-index is absent from this
   frame's tried list. If a rule exists, output {"op":"TRY","production":"pN"}
   using its actual index. If none exists, output {"op":"DONE"}.

State transitions for complete execution:
TRY marks that rule tried in the parent and pushes its substituted child form
with tried=[] and phase="entered". Replace only the leftmost nonterminal.
PRN, REJ, CUT, and DONE pop the current frame. If a parent remains, its phase
becomes "await_bt". If none remains, execution ends: reject, or unknown after CUT.
ACC immediately ends the ENTIRE execution with acceptance. Never emit DONE or
any other action after ACC. BT does not pop a frame. It sets phase="expanding".
Stop when execution ends or max_steps is reached; the latter is unknown.
Return only action JSON, using exactly the specified op names and reason strings.
'''


def clarify_prompt(prompt):
    if not prompt.startswith(PROTOCOL):
        raise ValueError('expected a controller prompt')
    return CLARIFIED_PROTOCOL+prompt[len(PROTOCOL):]


def build_primitive_ladder(levels=(1,2,4,8,16,32),variants=8):
    rng=random.Random(119)
    tasks=[]
    def add(category,level,variant,prompt,expected):
        tasks.append({'id':stable_id('primitive',[category,level,variant,prompt]),
            'category':category,'level':level,'variant':variant,'prompt':prompt+
            '\nReturn only the requested raw JSON. No Markdown or explanation.', 'expected':expected})
    for n in levels:
        for v in range(variants):
            s=''.join(rng.choice('abcd') for _ in range(n))
            pos=(v*n)//variants
            alt=s[:pos]+next(c for c in 'abcd' if c!=s[pos])+s[pos+1:]
            for same in (True,False):
                other=s if same else alt
                add('equality',n,v,f'Are the two strings exactly equal? A={json.dumps(s)}, B={json.dumps(other)}. '
                    'Return {"match": true} if equal, otherwise {"match": false}.',{'match':same})
                add('prefix',n,v,f'Does {json.dumps(s+"ab")} start with {json.dumps(other)}? '
                    'Return {"match": true} if it does, otherwise {"match": false}.',{'match':same})
            ids=[f'p{i}' for i in range(n)]
            rng.shuffle(ids)
            cases=[[],ids[:n//2],ids[1:],ids]
            for tried in cases:
                first=next((p for p in ids if p not in tried),None)
                add('first_untried',n,v,
                    'Choose the first item in applicable, in its given order, that is not in tried. '
                    'If one exists return {"op":"TRY","production":item}; otherwise return {"op":"DONE"}.\n'+
                    json.dumps({'applicable':ids,'tried':tried}),
                    {'op':'TRY','production':first} if first else {'op':'DONE'})
    # Avoid giving duplicate trivial prompts extra weight at short lengths.
    return list({(t['category'],t['level'],t['prompt']):t for t in tasks}.values())


def build_balanced_membership_suite(levels=(2,4,8,16),variants=4):
    """Two bracket types; swap closing partners while preserving terminal counts."""
    if any(n<2 or n%2 for n in levels) or not 1<=variants<=5:
        raise ValueError('even levels >=2 and 1..5 variants required')
    rows=[]
    for n in levels:
        for v in range(variants):
            alphabet='abcde'[v:]+'abcde'[:v]
            a,b,c,d,e=alphabet
            opens=[a]*(n//2)+[c]*(n//2)
            random.Random(307+n*13+v).shuffle(opens)
            close={a:b,c:d}
            tokens=opens+[e]+[close[x] for x in reversed(opens)]
            pair=stable_id('balanced_pair',[n,v,tokens])
            for positive in (True,False):
                rules=[('S',(a,'S',b if positive else d)),
                       ('S',(c,'S',d if positive else b)),('S',(e,))]
                grammar=Grammar('S',frozenset(['S']),frozenset(alphabet),
                    tuple(Production(lhs,rhs) for lhs,rhs in rules))
                assert accepts(grammar,tuple(tokens))==positive
                row={'task_id':stable_id('balanced',[pair,positive]),'pair_id':pair,'split':'test',
                    'family':'paired_brackets','depth_band':f'level_{n:02}',
                    'structure_id':f'paired_brackets_{n}', 'grammar':grammar.to_dict(),
                    'input':tokens,'target_answer':'accept' if positive else 'reject',
                    'search_config':{'max_steps':512,'max_depth':n+4},
                    'difficulty':{'level':n,'variant':v,'string_length':len(tokens),'rule_count':3,
                                  'balanced_pair_counts':True}}
                steps=teacher_trajectory(row)
                row['difficulty'].update(search_steps=len(steps),backtracks=sum(s['action']['op']=='BT' for s in steps),
                    derivation_depth=max(len(s['observation']['stack'])-1 for s in steps))
                rows.append(row)
    return rows


def _cuefree_rules(family, level, a, b, c, positive):
    n = level
    if family == 'terminal_length':          # string a^n ; no: rule writes a^(n+1)
        return [a] * n, [('S', tuple([a] * (n if positive else n + 1)))]
    if family == 'chain_depth':              # string a ; no: the chain ends in a a
        names = ['S'] + [f'N{i}' for i in range(1, n + 1)]
        return [a], [(x, (y,)) for x, y in zip(names, names[1:])] + [(names[-1], (a,) if positive else (a, a))]
    if family == 'alternatives':             # original yes: b | bb | ... | a on string a ; no: the last option is a a
        return [a], [('S', tuple([b] * i)) for i in range(1, n)] + [('S', (a,) if positive else (a, a))]
    if family == 'sequence':                 # string a^n ; no: n+1 placeholders
        return [a] * n, [('S', tuple(['A'] * (n if positive else n + 1))), ('A', (a,))]
    if family == 'recursion':                # string a^(n-1) c ; no: recursion adds two a's at a time, parity flipped
        k = n - 1
        if positive:
            return [a] * k + [c], [('S', (a, 'S')), ('S', (c,))]
        step = (a, a, 'S')
        base = (c,) if k % 2 else (a, c)      # reaches only the other parity
        return [a] * k + [c], [('S', step), ('S', base)]
    if family == 'nesting':                  # string a^n c b^n ; no: each a needs two b's
        return [a] * n + [c] + [b] * n, [('S', (a, 'S', b) if positive else (a, 'S', b, b)), ('S', (c,))]
    raise ValueError(family)


def build_cuefree_complexity_suite(levels=LEVELS, variants=4):
    """Original yes rows verbatim; structurally rejecting partners without odd symbols.

    Pair IDs retain the original pairing. Negative task IDs are distinct from the
    old suite so saved outputs cannot silently join to changed grammars.
    """
    from copy import deepcopy
    original = build_complexity_suite(levels, variants)
    alphabets = list(itertools.permutations('abcd'))
    random.Random(71).shuffle(alphabets)
    rows = []
    for old in original:
        row = deepcopy(old)
        positive = row['target_answer'] == 'accept'
        a, b, c, d = alphabets[row['difficulty']['variant']]
        tokens, rules = _cuefree_rules(row['family'], row['difficulty']['level'], a, b, c, positive)
        grammar = Grammar('S', frozenset(lhs for lhs, _ in rules), frozenset((a,b,c,d)),
                          tuple(Production(lhs, rhs) for lhs, rhs in rules))
        assert accepts(grammar, tuple(tokens)) == positive
        assert tokens == old['input']
        if positive:
            assert grammar.to_dict() == old['grammar']
            assert row == old
        else:
            row['grammar'] = grammar.to_dict()
            row['task_id'] = stable_id('cuefree_complexity', [old['task_id'], row['grammar']])
            steps = teacher_trajectory(row)
            row['difficulty'].update(search_steps=len(steps),
                backtracks=sum(s['action']['op']=='BT' for s in steps),
                derivation_depth=max(len(s['observation']['stack'])-1 for s in steps))
        rows.append(row)
    assert [r for r in rows if r['target_answer']=='accept'] == [r for r in original if r['target_answer']=='accept']
    return rows
