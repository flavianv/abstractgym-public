from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from analyze_countpreserving import operator_stats, summarize
from report_countpreserving_offloading import guard_stats, live_errors
from report_wholerun import cost
from run_ray_offloading import CONDITIONS, SUITES, verify_study_inputs

ROOT = Path('docs/results/countpreserving_offloading')


@pytest.fixture(scope='module')
def audit():
    return json.loads(gzip.decompress((ROOT / 'audit.json.gz').read_bytes()))


def read(name):
    return json.loads((ROOT / (name + '.json')).read_text())


def test_published_manifest_and_ordered_completion():
    manifest = read('manifest')
    required = {'audit.json.gz', 'summary.json', 'operators.json', 'guards.json', 'costs.json',
        'live_errors.json', 'comparisons.json', 'metadata.json', 'baseline_summary.json',
        'baseline_operators.json', 'baseline_costs.json', 'baseline_metadata.json'}
    assert set(manifest) == required
    for name, sha in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    metadata = read('metadata')
    for suite in SUITES:
        job = metadata[suite]['job']
        assert job['status'] == 'SUCCEEDED'
        assert datetime.fromisoformat(job['started_at']) >= max(
            datetime.fromisoformat(j['ended_at']) for j in metadata[suite]['baseline_jobs'].values())
        assert metadata[suite]['completion']['cases'] == 2880
        assert all(c['completed'] and c['cases'] == 480 for c in metadata[suite]['conditions'].values())


def test_every_case_and_failure_is_retained(audit):
    instances, protocol = verify_study_inputs()
    assert audit['instances'] == instances and audit['protocol'] == protocol
    summary, ops, guards, costs, errors = (read(n) for n in ('summary', 'operators', 'guards', 'costs', 'live_errors'))
    total = 0
    for suite in SUITES:
        ids = {r['task_id'] for r in instances[suite]}
        assert len(audit['baseline'][suite]) == 960
        for condition, config in CONDITIONS.items():
            records = audit['offloading'][suite][condition]
            assert len(records) == 480
            assert {(r['task_id'], r['mode']) for r in records} == {
                (i, m) for i in ids for m in ('external_state', 'oracle_step')}
            assert all(r['provenance'] == 'fresh' and 0 < r['model_calls'] == len(r['calls']) <= 512 for r in records)
            assert all(r['hints'] == config['hints'] and r['guard_requested'] == config['legality_guard'] for r in records)
            for mode in ('external_state', 'oracle_step'):
                selected = [r for r in records if r['mode'] == mode]
                assert summary[suite][condition][mode] == summarize(selected)
                assert costs[suite][condition][mode] == cost(selected)
                overall = summary[suite][condition][mode]['overall']
                assert overall['total'] == 240
                assert overall['correct'] + sum(overall['failures'].values()) == 240
            assert ops[suite][condition] == operator_stats(instances[suite], records)
            assert guards[suite][condition] == guard_stats(records)
            assert errors[suite][condition] == live_errors(instances[suite], records)
            total += len(records)
    assert total == 11520


def test_baseline_metrics_reproduce_from_the_published_raw_audit(audit):
    summary, ops, costs = (read('baseline_' + n) for n in ('summary', 'operators', 'costs'))
    for suite in SUITES:
        records = audit['baseline'][suite]
        for mode in ('direct', 'full_trace', 'external_state', 'oracle_step'):
            selected = [r for r in records if r['mode'] == mode]
            assert summary[suite][mode] == summarize(selected)
            assert costs[suite][mode] == cost(selected)
        assert ops[suite] == operator_stats(audit['instances'][suite], records)


def test_report_contains_all_required_sections_and_links():
    path = Path('docs/countpreserving_offloading_results.md')
    text = path.read_text()
    for heading in ('Main comparison', 'What this answers', 'Shortcut gate',
                    'Part 2: baseline atlas', 'Part 3: offloading atlas',
                    'Guard outcomes', 'Live first errors versus teacher-state errors',
                    'Failure accounting and call budgets', 'Reproducibility and interpretation limits'):
        assert heading in text
    assert text.count('| Operator | Correct / total |') == 28
    import re
    for target in re.findall(r'\]\(([^)]+)\)', text):
        assert (path.parent / target).exists()
    assert 'countpreserving_offloading_results.md' in Path('README.md').read_text()
