"""A6 data and scoring. No Torch dependency; frozen brackets remain unchanged."""
import hashlib
import itertools
import json
from statistics import median
from abstractgym.brackets import suite, expected, advance
from abstractgym.cfg import Grammar, Production
from abstractgym.oracle import accepts

PHASES = ('opening', 'closing')
ROLES = ('OPEN_0', 'OPEN_1', 'CLOSE_0', 'CLOSE_1', 'CENTER', 'END')
TOPS = ('EMPTY', 'CLOSE_0', 'CLOSE_1')
ACTIONS = ('PUSH_0', 'PUSH_1', 'CENTER', 'POP', 'ACCEPT', 'REJECT')
VOCAB = ('PAD', 'CLS', 'opening', 'closing') + ROLES + ('EMPTY', 'DIRECT')
DEV = dict(pairs={'u': 'w', 'v': 'x'}, center='y', alphabet='uvwxy')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def role_map(row):
    mapping = {row['center']: 'CENTER'}
    for i, (opening, closing) in enumerate(row['pairs'].items()):
        mapping[opening] = f'OPEN_{i}'
        mapping[closing] = f'CLOSE_{i}'
    return mapping


def abstract_state(row, pos, mode, stack):
    roles = role_map(row)
    return (mode, roles[row['input'][pos]] if pos < len(row['input']) else 'END',
            roles[stack[-1]] if stack else 'EMPTY')


def encode_step(key):
    return [VOCAB.index(x) for x in ('CLS',) + tuple(key)]


def encode_direct(row):
    roles = role_map(row)
    return [VOCAB.index(x) for x in ['CLS', 'DIRECT'] + [roles[t] for t in row['input']] + ['END']]


def action_id(action, row):
    if action['op'] == 'PUSH':
        return ACTIONS.index('PUSH_' + str(list(row['pairs'].values()).index(action['value'])))
    return ACTIONS.index(action['op'])


def decode_action(index, row):
    action = ACTIONS[index]
    if action.startswith('PUSH_'):
        return {'op': 'PUSH', 'value': list(row['pairs'].values())[int(action[-1])]}
    return {'op': action}


def grammar_for(row):
    return Grammar('S', frozenset(['S']), frozenset(role_map(row)), tuple(
        [Production('S', (a, 'S', b)) for a, b in row['pairs'].items()] + [Production('S', (row['center'],))]))


def dev_data():
    """All dev openings/closings through depth 4, plus coverage traces <= depth 2."""
    sequences = set()
    for depth in range(5):
        for opens in itertools.product(DEV['pairs'], repeat=depth):
            for closes in itertools.product(DEV['pairs'].values(), repeat=depth):
                sequences.add(opens + (DEV['center'],) + closes)
    sequences.add(('y', 'w'))  # negative at depth zero
    # Reach every abstract input through real reference transitions, not injected states.
    inverse = {v: k for k, v in role_map(DEV).items()}
    for phase, token, top in itertools.product(PHASES, ROLES, TOPS):
        prefix = [] if top == 'EMPTY' else [list(DEV['pairs'])[TOPS.index(top) - 1]]
        if phase == 'closing':
            prefix.append(DEV['center'])
        if token != 'END':
            prefix.append(inverse[token])
        sequences.add(tuple(prefix))
    rows = []; steps = []; unique = {}
    grammar = grammar_for(DEV)
    for i, tokens in enumerate(sorted(sequences)):
        row = dict(DEV, id=f'dev-{i}', input=list(tokens), accept=accepts(grammar, tokens))
        rows.append(row)
        pos, mode, stack = 0, 'opening', []
        for _ in range(len(tokens) + 1):
            key = abstract_state(row, pos, mode, stack)
            gold = expected(row, pos, mode, stack)
            label = action_id(gold, row)
            steps.append(dict(trajectory=row['id'], position=pos, state=list(key), action=label))
            if key in unique:
                assert unique[key] == label
            unique[key] = label
            if gold['op'] in ('ACCEPT', 'REJECT'):
                assert (gold['op'] == 'ACCEPT') == row['accept']
                break
            pos, mode, stack = advance(gold, pos, mode, stack)
            assert len(stack) <= 4
    assert set(unique) == set(itertools.product(PHASES, ROLES, TOPS))
    return rows, steps, unique


def exhaustive_rows():
    """All 36 role states, including states absent from the frozen case templates."""
    inverse = {v: k for k, v in role_map(DEV).items()}
    result = []
    for key in itertools.product(PHASES, ROLES, TOPS):
        phase, token, top = key
        row = dict(DEV, input=[] if token == 'END' else [inverse[token]])
        stack = [] if top == 'EMPTY' else [inverse[top]]
        result.append((key, action_id(expected(row, 0, phase, stack), row)))
    return result


def extra_rows():
    rows = []
    for template in (r for r in suite() if r['split'] == 'test' and r['depth'] == 16):
        for depth in (32, 64):
            # Extend the four fixed patterns, retain the existing negative transformation.
            opens = (template['input'][:16] * (depth // 16))
            closes = [template['pairs'][x] for x in reversed(opens)]
            if not template['accept']:
                if len(set(closes)) == 2:
                    j = next(i for i in range(1, len(closes)) if closes[i] != closes[0])
                    closes[0], closes[j] = closes[j], closes[0]
                else:
                    closes[0] = next(x for x in template['pairs'].values() if x != closes[0])
            row = dict(template, id=template['id'] + f'-extra-{depth}', depth=depth,
                       input=opens + [template['center']] + closes)
            assert accepts(grammar_for(row), tuple(row['input'])) == row['accept']
            rows.append(row)
    return rows


def evaluate_execution(rows, predict):
    """predict sees abstract observations only; first-error stopping matches frozen harness."""
    active = [dict(row=r, pos=0, mode='opening', stack=[], calls=[]) for r in rows]
    records = []
    while active:
        keys = [abstract_state(s['row'], s['pos'], s['mode'], s['stack']) for s in active]
        predictions = predict(keys)
        assert len(predictions) == len(active)
        continuing = []
        for s, key, prediction in zip(active, keys, predictions):
            row = s['row']; action = decode_action(prediction, row)
            gold = expected(row, s['pos'], s['mode'], s['stack'])
            correct = action == gold
            s['calls'].append(dict(state=list(key), prediction=prediction, expected=action_id(gold,row), correct=correct))
            if not correct or gold['op'] in ('ACCEPT', 'REJECT'):
                records.append(dict(id=row['id'], success=correct, calls=s['calls'],
                                    first_error=None if correct else len(s['calls'])))
            else:
                # Apply the model's action, not a reference substitution.
                s['pos'], s['mode'], s['stack'] = advance(action, s['pos'], s['mode'], s['stack'])
                continuing.append(s)
        active = continuing
    return records


def summarize(rows, records):
    lookup = {r['id']: r for r in rows}
    result = {}
    baselines = {}
    for name, alphabet in [('letters', 'abcde'), ('brackets', '([)]#')]:
        depths = sorted({r['depth'] for r in rows if r['alphabet'] == alphabet})
        by_depth = {}; blind = {}; status = {}
        for depth in depths:
            cases = [r for r in records if lookup[r['id']]['alphabet'] == alphabet and lookup[r['id']]['depth'] == depth]
            n = len(cases); ok = sum(r['success'] for r in cases)
            yes = sum(lookup[r['id']]['accept'] for r in cases)
            by_depth[str(depth)] = [ok, n]
            blind[str(depth)] = {'always_accept':[yes,n], 'always_reject':[n-yes,n]}
            status[str(depth)] = 'beats both label-blind baselines' if ok > max(yes,n-yes) else ('chance / label-blind' if ok == max(yes,n-yes) else 'below label-blind')
        result[name] = {'by_depth':by_depth, 'interpretation':status}
        baselines[name] = blind
    result['total'] = [sum(r['success'] for r in records), len(records)]
    calls = [c for r in records for c in r['calls']]
    errors = [r['first_error'] for r in records if r['first_error'] is not None]
    push_errors = sum(c['expected'] in (0,1) and c['prediction'] in (0,1) and not c['correct'] for c in calls)
    return dict(held_out=result, action_accuracy=[sum(c['correct'] for c in calls),len(calls)],
                first_error_median=median(errors) if errors else None,
                push_value_errors=push_errors,
                label_blind={'always_accept':[sum(r['accept'] for r in rows),len(rows)],
                             'always_reject':[sum(not r['accept'] for r in rows),len(rows)],'by_cell':baselines})
