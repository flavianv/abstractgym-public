"""Build and gate count-preserving suites before any model submission."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from abstractgym.countpreserving import (build_suite_a, build_suite_b, count_features,
    predictions, validate_pairs)


def scores(rows, predicted):
    def cell(indices):
        return [sum(predicted[i] == (rows[i]['target_answer'] == 'accept') for i in indices), len(indices)]
    return dict(overall=cell(range(len(rows))),
        by_family={f: cell([i for i, r in enumerate(rows) if r['family'] == f])
            for f in sorted({r['family'] for r in rows})},
        by_size={str(n): cell([i for i, r in enumerate(rows) if r['difficulty']['level'] == n])
            for n in sorted({r['difficulty']['level'] for r in rows})},
        family_size={f: {str(n): cell([i for i, r in enumerate(rows)
            if r['family'] == f and r['difficulty']['level'] == n])
            for n in sorted({r['difficulty']['level'] for r in rows if r['family'] == f})}
            for f in sorted({r['family'] for r in rows})})


def bag_classifier(rows):
    import sklearn
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    result, folds = [None] * len(rows), []
    for fold in range(4):
        held = [i for i, r in enumerate(rows) if r['difficulty']['variant'] % 4 == fold]
        train = [i for i in range(len(rows)) if i not in held]
        assert not {rows[i]['pair_id'] for i in held} & {rows[i]['pair_id'] for i in train}
        vectorizer = DictVectorizer(sparse=True)
        x = vectorizer.fit_transform([count_features(rows[i]) for i in train])
        model = LogisticRegression(random_state=17, max_iter=1000, solver='lbfgs')
        model.fit(x, [rows[i]['target_answer'] == 'accept' for i in train])
        y = model.predict(vectorizer.transform([count_features(rows[i]) for i in held]))
        for i, value in zip(held, y):
            result[i] = bool(value)
        folds.append(dict(fold=fold, train_pairs=sorted({rows[i]['pair_id'] for i in train}),
            heldout_pairs=sorted({rows[i]['pair_id'] for i in held}), rows=len(held)))
    assert all(v is not None for v in result)
    return result, dict(package='scikit-learn', version=sklearn.__version__, seed=17,
        solver='lbfgs', max_iter=1000, four_fold_pair_grouped=True, folds=folds,
        features='grammar/input counts including individual production bags; no order, labels, or trace statistics')


def battery(rows):
    pairs = validate_pairs(rows)
    raw = [predictions(r) for r in rows]
    names = sorted(raw[0])
    assert all(set(p) == set(names) for p in raw)
    classifier, metadata = bag_classifier(rows)
    result = {name: scores(rows, [p[name] for p in raw]) for name in names}
    result['bag-of-symbols logistic regression (held-out pairs)'] = scores(rows, classifier)
    bad = []
    for name, scored in result.items():
        cells = [('overall', scored['overall'])]
        cells += [('family:' + k, v) for k, v in scored['by_family'].items()]
        cells += [('size:' + k, v) for k, v in scored['by_size'].items()]
        cells += [(f'{f}/{n}', v) for f, ns in scored['family_size'].items() for n, v in ns.items()]
        for cell, (correct, total) in cells:
            if not .45 <= correct / total <= .55:
                bad.append(dict(rule=name, cell=cell, score=[correct, total]))
    return dict(passed=not bad, violations=bad, tolerance=[.45, .55], scores=result,
        classifier=metadata, count_invariant_pairs=len(pairs),
        invariance='Identical per-rule bags and input bags make every deterministic function of these counts label-balanced within pairs.')


def main():
    folder = Path('experiments/countpreserving')
    folder.mkdir(parents=True, exist_ok=True)
    a = build_suite_a()
    b, generation = build_suite_b()
    suites = dict(A=a, B=b)
    audited = {name: battery(rows) for name, rows in suites.items()}
    assert all(len(rows) == 240 for rows in suites.values())
    def save(name, value):
        (folder / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    for name, rows in suites.items():
        save(name + '.json', rows)
    save('shortcuts.json', audited)
    save('generation.json', generation)
    save('gate.json', dict(passed=all(a['passed'] for a in audited.values()),
        model_calls=0, suites={name: dict(rows=len(rows), labels=dict(Counter(r['target_answer'] for r in rows)),
            max_search_steps=max(r['difficulty']['search_steps'] for r in rows)) for name, rows in suites.items()}))
    save('manifest.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'manifest.json'})
    assert all(a['passed'] for a in audited.values()), 'SHORTCUT GATE FAILED: do not submit model evaluation'
    print(json.dumps({name: dict(passed=a['passed'], rules=len(a['scores']),
        invariant_pairs=a['count_invariant_pairs']) for name, a in audited.items()}, indent=2))


if __name__ == '__main__':
    main()
