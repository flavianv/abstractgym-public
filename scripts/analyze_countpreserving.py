"""Replay baseline artifacts and summarize outcomes without dropping failures."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from abstractgym.controller import teacher_trajectory
from report_wholerun import LANES, cell, cost, read, replay
from run_ray_countpreserving import SETTINGS, verify_inputs

OPERATORS = ('DONE', 'TRY', 'PRN', 'BT', 'ACC', 'REJ', 'CUT')


def direct_bias(records):
    replies = Counter()
    for record in records:
        try:
            action = json.loads(record['text'])
        except ValueError:
            action = None
        if isinstance(action, dict) and set(action) == {'accept'} and type(action['accept']) is bool:
            replies['yes' if action['accept'] else 'no'] += 1
        else:
            replies['malformed'] += 1
    return dict(cases=len(records), yes=replies['yes'], no=replies['no'], malformed=replies['malformed'],
        yes_rate_all_cases=replies['yes'] / len(records) if records else None,
        yes_rate_valid_replies=replies['yes'] / (replies['yes'] + replies['no'])
            if replies['yes'] + replies['no'] else None)


def summarize(records):
    def scored(selected):
        result = cell(selected)
        if selected and selected[0]['mode'] == 'direct':
            result['reply_bias'] = direct_bias(selected)
        return result
    families = sorted({r['family'] for r in records})
    levels = sorted({r['difficulty']['level'] for r in records})
    return dict(overall=scored(records),
        by_family={f: scored([r for r in records if r['family'] == f]) for f in families},
        by_level={str(n): scored([r for r in records if r['difficulty']['level'] == n]) for n in levels},
        family_level={f: {str(n): scored([r for r in records
            if r['family'] == f and r['difficulty']['level'] == n]) for n in levels} for f in families})


def operator_stats(rows, records):
    lookup = {r['task_id']: r for r in rows}
    result = {op: dict(correct=0, total=0, mistakes={}) for op in OPERATORS}
    for record in records:
        if record['mode'] != 'oracle_step':
            continue
        gold = teacher_trajectory(lookup[record['task_id']])
        if len(record['calls']) != len(gold):
            raise ValueError('incomplete teacher-state evidence')
        for step, call in zip(gold, record['calls']):
            expected = step['action']
            metric = result[expected['op']]
            metric['total'] += 1
            try:
                action = json.loads(call.get('text', ''))
            except ValueError:
                action = None
            if action == expected:
                metric['correct'] += 1
                continue
            error = action.get('op') if isinstance(action, dict) else None
            error = error if error in OPERATORS else 'malformed_or_unknown'
            if error == 'TRY' and action.get('production') in step['observation']['stack'][-1]['tried']:
                error = 'TRY_already_tried'
            metric['mistakes'][error] = metric['mistakes'].get(error, 0) + 1
    return result


def assemble(root):
    suites, manifest = verify_inputs()
    original = read('docs/results/wholerun/audit.json.gz')
    historical_sha = hashlib.sha256(Path('docs/results/wholerun/audit.json.gz').read_bytes()).hexdigest()
    expected = read('docs/results/wholerun/manifest.json')['audit.json.gz']
    if historical_sha != expected:
        raise ValueError('historical evidence hash mismatch')
    instances, runs, metadata = {}, {}, {}
    for suite in ('original', 'cuefree', 'A', 'B'):
        if suite in suites:
            folder = root / suite / 'results'
            completion = read(folder / 'completion.json')
            if not completion['completed'] or completion['direct_cases'] != 240 or completion['execution_cases'] != 720:
                raise ValueError('incomplete baseline suite: ' + suite)
            runtime = read(folder / 'runtime.json')
            if runtime['settings'] != SETTINGS or runtime['manifest'] != manifest or runtime['suite'] != suite:
                raise ValueError('baseline runtime mismatch')
            staging = read(root / suite / 'staging.json')
            if any(sha != staging['files'][name] for name, sha in runtime['source_sha256'].items()):
                raise ValueError('deployed source mismatch')
            instances[suite] = suites[suite]
            records = read(folder / 'execution.jsonl') + read(folder / 'direct.json')
            records = [dict(r, provenance='fresh') for r in records]
            metadata[suite] = dict(runtime=runtime, completion=completion, staging=staging,
                preflight=read(folder / 'preflight.json'))
        else:
            instances[suite] = original['instances'][suite]
            records = [dict(r, provenance='historical whole-run published zero-shot evidence')
                for r in original['runs'][suite]['zeroshot']]
            metadata[suite] = dict(historical_audit_sha256=historical_sha, fresh_cases=0)
        lookup = {r['task_id']: r for r in instances[suite]}
        keys = [(r['task_id'], r['mode']) for r in records]
        if len(keys) != 960 or len(set(keys)) != 960 or set(keys) != {(i, m) for i in lookup for m in LANES}:
            raise ValueError('missing or duplicated baseline evidence')
        for record in records:
            row = lookup[record['task_id']]
            record.update(family=row['family'], difficulty=row['difficulty'], target_answer=row['target_answer'])
            replay(row, record)
        runs[suite] = records
    return instances, runs, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('build/countpreserving-20261005'))
    parser.add_argument('--output', type=Path, default=Path('build/countpreserving-20261005/baseline_analysis'))
    args = parser.parse_args()
    instances, runs, metadata = assemble(args.root)
    values = dict(summary={s: {m: summarize([r for r in records if r['mode'] == m]) for m in LANES}
        for s, records in runs.items()}, operators={s: operator_stats(instances[s], records) for s, records in runs.items()},
        costs={s: {m: cost([r for r in records if r['mode'] == m]) for m in LANES} for s, records in runs.items()},
        metadata=metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        (args.output / (name + '.json')).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    (args.output / 'audit.json.gz').write_bytes(gzip.compress(json.dumps(dict(instances=instances, runs=runs),
        sort_keys=True).encode(), mtime=0))
    print('baseline_analysis_complete: all 3840 cases replayed; historical comparisons labelled')


if __name__ == '__main__':
    main()
