import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_wholerun import replay, cost, diagnostic_summary
from abstractgym.wholerun import written_diagnostic

ROOT = Path(__file__).resolve().parents[1]


def test_original_references_replay_in_all_four_lanes():
    x7 = json.loads(gzip.decompress((ROOT / 'docs/results/2026-10-04-x7/audit.json.gz').read_bytes()))
    cue = json.loads(gzip.decompress((ROOT / 'docs/results/cuefree/audit.json.gz').read_bytes()))
    lookup = {r['task_id']: r for r in x7['instances']}
    for records in x7['runs'].values():
        for record in records:
            replay(lookup[record['task_id']], record)
    for record in x7['membership'] + cue['historical_direct']:
        replay(lookup[record['task_id']], dict(record, mode='direct'))


def test_cost_counts_fresh_calls_inside_a_partly_reused_lane():
    def record(provenance, tokens):
        return dict(provenance=provenance, mode='full_trace', calls=[dict(
            usage=dict(prompt_tokens=10, completion_tokens=tokens, total_tokens=10 + tokens),
            finish_reason='stop', inference_seconds=.5)])
    c = cost([record('fresh', 7), record('identical original yes', 11)])
    assert c['completion_tokens'] == 18 and c['fresh_completion_tokens'] == 7
    assert c['total_tokens'] == 38 and c['fresh_total_tokens'] == 17
    assert c['reused_cases'] == c['fresh_cases'] == 1


def test_x7_failure_categories_partition_all_cases():
    from collections import Counter
    a = json.loads(gzip.decompress((ROOT / 'docs/results/2026-10-04-x7/audit.json.gz').read_bytes()))
    rows = {r['task_id']: r for r in a['instances']}
    ds = [written_diagnostic(rows[r['task_id']], r) for r in a['runs']['dfs_lora'] if r['mode'] == 'full_trace']
    assert Counter(d['category'] for d in ds) == dict(one_move_then_stop=133, cap_length_loop=47,
        malformed_invalid_or_trailing=51, other_incomplete=9)


def test_pair_statistics_exclude_missing_moves_from_wrong_move_medians():
    pairs = [dict(written_success=False, gym_success=False, first_wrong_move=None, first_missing_move=3,
        first_wrong_fraction=None, canonical_actions=8, correct_prefix_actions=3, gym_first_error=0, gym_first_error_fraction=0),
        dict(written_success=False, gym_success=False, first_wrong_move=4, first_missing_move=None,
            first_wrong_fraction=.5, canonical_actions=8, correct_prefix_actions=4, gym_first_error=2, gym_first_error_fraction=.25),
        dict(written_success=True, gym_success=True)]
    s = diagnostic_summary(pairs)
    assert s['failed'] == 2 and s['missing_without_wrong'] == 1
    assert s['paired_observed_errors'] == 1 and s['paired_written_index'] == 4
    assert s['paired_gym_index'] == 2 and s['paired_written_fraction'] == .5
    assert s['total'] == 3 and s['written_correct'] == s['gym_correct'] == 1


def test_published_wholerun_results_replay_and_keep_all_denominators():
    import hashlib
    from report_wholerun import analyze, read
    folder = ROOT / 'docs/results/wholerun'
    for name, sha in read(folder / 'manifest.json').items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == sha
    audit = read(folder / 'audit.json.gz')
    for suite, arms in audit['runs'].items():
        lookup = {r['task_id']: r for r in audit['instances'][suite]}
        for records in arms.values():
            assert len(records) == 960
            assert len({(r['task_id'], r['mode']) for r in records}) == 960
            for record in records:
                replay(lookup[record['task_id']], record)
    summary, diagnostics, costs = analyze(audit['instances'], audit['runs'])
    assert summary == read(folder / 'summary.json')
    assert diagnostics == read(folder / 'diagnostics.json')
    assert costs == read(folder / 'costs.json')
    for arms in summary.values():
        for lanes in arms.values():
            for lane in lanes.values():
                assert lane['overall']['total'] == 240
                assert lane['overall']['yes'][1] == lane['overall']['no'][1] == 120
    assert [summary[s]['mixed_sft'][m]['overall']['correct']
        for s in ('original', 'cuefree') for m in ('full_trace', 'external_state')] == [100, 156, 89, 128]
