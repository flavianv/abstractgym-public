"""Order-only paired puzzles; count invariance gates all model evaluation."""
from collections import Counter
import itertools
import math
import random

from abstractgym.cfg import Grammar, Production
from abstractgym.complexity import FAMILIES, LEVELS
from abstractgym.controller import teacher_trajectory
from abstractgym.cuefree import shortcut_predictions
from abstractgym.oracle import accepts
from abstractgym.schema import stable_id


def _row(suite, family, level, variant, grammar, tokens, positive, pair, **extra):
    grammar.validate_v0_constraints()
    assert accepts(grammar, tuple(tokens)) == positive
    row = dict(task_id=stable_id(suite, [pair, positive, grammar.to_dict(), tokens]),
        pair_id=pair, split='test', family=family, depth_band=f'level_{level:02}',
        structure_id=f'{suite}_{family}_{level}', grammar=grammar.to_dict(), input=list(tokens),
        target_answer='accept' if positive else 'reject',
        search_config=dict(max_steps=512, max_depth=max(level + 4, len(tokens) + 2)),
        difficulty=dict(level=level, variant=variant, string_length=len(tokens),
            rule_count=len(grammar.productions), **extra))
    steps = teacher_trajectory(row)
    row['difficulty'].update(search_steps=len(steps),
        backtracks=sum(s['action']['op'] == 'BT' for s in steps),
        derivation_depth=max(len(s['observation']['stack']) - 1 for s in steps),
        accepting_depth=len(steps[-1]['observation']['stack']) - 1 if positive else None)
    return row


def _grammar(rules, alphabet):
    return Grammar('S', frozenset(lhs for lhs, _ in rules), frozenset(alphabet),
        tuple(Production(lhs, tuple(rhs)) for lhs, rhs in rules))


def build_suite_a(levels=LEVELS, variants=4):
    if any(n < 1 for n in levels) or not 1 <= variants <= 24:
        raise ValueError('positive levels and 1..24 letterings required')
    alphabets = list(itertools.permutations('abcde', 5))
    random.Random(71).shuffle(alphabets)
    rows = []
    for n in levels:
        for family in FAMILIES:
            for v, alphabet in enumerate(alphabets[:variants]):
                a, b, c, d, e = alphabet
                pair = stable_id('count_a_pair', [family, n, v])
                for yes in (True, False):
                    tokens = [a, b] * n
                    if family == 'terminal_length':
                        rhs = tokens.copy()
                        if not yes:
                            rhs[0], rhs[1] = rhs[1], rhs[0]
                        rules = [('S', rhs)]
                    elif family == 'chain_depth':
                        tokens = [a, b]
                        names = ['S'] + [f'N{i}' for i in range(1, n + 1)]
                        rules = [(x, (y,)) for x, y in zip(names, names[1:])]
                        rules += [(names[-1], (a, b) if yes else (b, a))]
                    elif family == 'alternatives':
                        k = 1
                        while math.comb(2 * k, k) <= n:
                            k += 1
                        tokens = [a] * k + [b] * k
                        options = []
                        for positions in itertools.combinations(range(2 * k), k):
                            value = tuple(a if i in positions else b for i in range(2 * k))
                            if value != tuple(tokens):
                                options.append(value)
                            if len(options) == n:
                                break
                        rules = [('S', value) for value in options[:n - 1]]
                        rules += [('S', tuple(tokens) if yes else options[n - 1])]
                    elif family == 'sequence':
                        rhs = ['A', 'B'] * n
                        if not yes:
                            rhs[0], rhs[1] = rhs[1], rhs[0]
                        rules = [('S', rhs), ('A', (a,)), ('B', (b,))]
                    elif family == 'recursion':
                        tokens += [c]
                        rules = [('S', (a, b, 'S') if yes else (b, a, 'S')), ('S', (c,))]
                    else:
                        opens = [a, c] * n
                        close = {a: b, c: d}
                        tokens = opens + [e] + [close[x] for x in reversed(opens)]
                        if not yes:
                            tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
                        rules = [('S', (a, 'S', b)), ('S', (c, 'S', d)), ('S', (e,))]
                    rows.append(_row('count_a', family, n, v, _grammar(rules, alphabet),
                        tokens, yes, pair, order_only=True))
    validate_pairs(rows)
    return rows


def _sample_random(rng):
    names = ['S'] + [f'N{i}' for i in range(1, rng.randint(3, 6))]
    alphabet = tuple(rng.sample('abcd', rng.randint(3, 4)))
    rules = []
    for name in names:
        count = rng.randint(2, 3)
        candidates = [tuple(rng.choices(alphabet, k=rng.randint(2, 4)))]
        while len(candidates) < count:
            rhs = tuple(rng.choices(alphabet, k=rng.randint(1, 2)))
            rhs += (rng.choice(names),) + tuple(rng.choices(alphabet, k=rng.randint(0, 1)))
            if rhs not in candidates:
                candidates.append(rhs)
        rng.shuffle(candidates)
        rules.extend((name, rhs) for rhs in candidates)
    return _grammar(rules, alphabet)


def build_suite_b(levels=LEVELS, pairs_per_level=24, seed=417, max_attempts=20000):
    rng = random.Random(seed)
    rows, seen = [], set()
    attempts, reasons = 0, Counter()
    for n in levels:
        accepted = 0
        while accepted < pairs_per_level:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f'Suite B generation budget exhausted: {dict(reasons)}')
            grammar = _sample_random(rng)
            form = ('S',)
            for depth in range(n):
                index = next(i for i, s in enumerate(form) if s in grammar.nonterminals)
                choices = [p for p in grammar.productions_for(form[index])
                    if any(s in grammar.nonterminals for s in p.rhs) == (depth < n - 1)]
                rule = rng.choice(choices)
                form = form[:index] + rule.rhs + form[index + 1:]
            positions = [i for i in range(len(form) - 1) if form[i] != form[i + 1]]
            rng.shuffle(positions)
            negative = None
            for i in positions:
                swapped = list(form)
                swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                if not accepts(grammar, tuple(swapped)):
                    negative = swapped
                    break
            if negative is None:
                reasons['no_rejecting_swap'] += 1
                continue
            content = stable_id('grammar', grammar.to_dict())
            if content in seen:
                reasons['duplicate_grammar'] += 1
                continue
            pair = stable_id('count_b_pair', [n, content, form])
            try:
                yes = _row('count_b', 'random_multi_rule', n, accepted, grammar, form, True, pair,
                    witness_derivation_depth=n, adjacent_swap_index=i)
                no = _row('count_b', 'random_multi_rule', n, accepted, grammar, negative, False, pair,
                    witness_derivation_depth=n, adjacent_swap_index=i)
            except ValueError:
                reasons['bounded_solver_disagreement_or_limit'] += 1
                continue
            if yes['difficulty']['accepting_depth'] != n:
                reasons['different_canonical_accepting_depth'] += 1
                continue
            rows.extend((yes, no))
            seen.add(content)
            accepted += 1
    validate_pairs(rows)
    return rows, dict(seed=seed, attempts=attempts, retained_pairs=len(rows) // 2,
        rejected=dict(reasons), generation_policy='random terminal/terminal-prefix single-child rules; rejection sampled',
        nonterminals=[3, 6], rules_per_nonterminal=[2, 3], vocabulary=[3, 4])


def count_features(row):
    grammar = Grammar.from_dict(row['grammar'])
    features = Counter({'input_length': len(row['input']), 'rules': len(grammar.productions),
        'nonterminals': len(grammar.nonterminals), 'vocabulary': len(grammar.terminals)})
    features.update({'input:' + s: n for s, n in Counter(row['input']).items()})
    for index, p in enumerate(grammar.productions):
        features['rule_length:' + str(len(p.rhs))] += 1
        features['placeholder_count:' + str(sum(s in grammar.nonterminals for s in p.rhs))] += 1
        features[f'rule:{index}:length'] = len(p.rhs)
        features[f'rule:{index}:placeholders'] = sum(s in grammar.nonterminals for s in p.rhs)
        for symbol in p.rhs:
            features[('placeholder:' if symbol in grammar.nonterminals else 'terminal:') + symbol] += 1
            features[f'rule:{index}:symbol:{symbol}'] += 1
    return dict(features)


def validate_pairs(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault(row['pair_id'], []).append(row)
    for pair in pairs.values():
        assert len(pair) == 2 and {r['target_answer'] for r in pair} == {'accept', 'reject'}
        assert count_features(pair[0]) == count_features(pair[1]), pair[0]['pair_id']
        assert Counter(pair[0]['input']) == Counter(pair[1]['input'])
        assert pair[0]['family'] == pair[1]['family']
        assert pair[0]['difficulty']['level'] == pair[1]['difficulty']['level']
    return pairs


def predictions(row):
    result = shortcut_predictions(row)
    rules = row['grammar']['productions']
    nts = set(row['grammar']['nonterminals'])
    tokens, length = Counter(row['input']), len(row['input'])
    terms = Counter(s for p in rules for s in p['rhs'] if s not in nts)
    lengths = [len(p['rhs']) for p in rules]
    placeholders = [sum(s in nts for s in p['rhs']) for p in rules]
    result.update({'always no': False, 'any rule length equals input': length in lengths,
        'longest rule length equals input': max(lengths) == length,
        'last rule length equals input': lengths[-1] == length,
        'longest full rule no longer than input': max(lengths) <= length,
        'any placeholder count equals input': length in placeholders,
        'total placeholders equal input': sum(placeholders) == length,
        'terminal multisets equal': terms == tokens,
        'total terminal count equals input': sum(terms.values()) == length,
        'terminal/input length parity equal': sum(terms.values()) % 2 == length % 2,
        'rule/input length parity equal': sum(lengths) % 2 == length % 2})
    alphabet = 'abcde'
    for a in alphabet:
        result['count equality:' + a] = terms[a] == tokens[a]
        result['count parity:' + a] = terms[a] % 2 == tokens[a] % 2
        for b in alphabet:
            for ratio in (1, 2, 3):
                result[f'input ratio {a}/{b}={ratio}'] = tokens[a] == ratio * tokens[b]
            result[f'grammar/input ratio {a}/{b}'] = terms[a] * tokens[b] == terms[b] * tokens[a]
    return result
