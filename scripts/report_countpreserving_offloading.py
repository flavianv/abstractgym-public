"""Replay the ordered study and publish the complete, failure-inclusive atlas."""
import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path

from abstractgym.controller import teacher_trajectory
from abstractgym.experiment import run_case
from analyze_countpreserving import OPERATORS, assemble as assemble_baseline, operator_stats, summarize
from report_wholerun import LANES, cost, read, score, table
from run_ray_offloading import CONDITIONS, SUITES, verify_study_inputs

LABELS = dict(direct='Direct answer', full_trace='Full execution in one reply',
    external_state='Gym memory', oracle_step='Correct-state check (all actions correct)')


def replay(row, record, config):
    index = 0
    def adapter(prompt):
        nonlocal index
        if index >= len(record['calls']):
            raise ValueError('missing response evidence')
        call = record['calls'][index]
        index += 1
        assert prompt == call['prompt']
        if 'text' not in call:
            assert call.get('error_type')
            raise RuntimeError('recorded inference failure')
        assert prompt == call['model_prompt']
        assert call['max_tokens'] == 128
        expected_batch = (min(32, len(record['calls']) - (index - 1) // 32 * 32)
                          if record['mode'] == 'oracle_step' else 1)
        assert call['batch_size'] == expected_batch
        return {k: v for k, v in call.items() if k not in ('guard_rejected', 'seconds', 'prompt')}
    result = run_case(row, record['mode'], adapter, max_calls=512, **config)
    assert index == len(record['calls']) <= 512
    for key in ('success', 'failure', 'first_error', 'correct_actions', 'target_actions',
                'outcome', 'model_calls', 'controller_calls'):
        assert result[key] == record[key], (row['task_id'], record['condition'], key)
    assert result.get('guard') == record.get('guard')
    assert [c.get('guard_rejected', False) for c in result['calls']] == [
        c.get('guard_rejected', False) for c in record['calls']]


def guard_stats(records):
    guarded = [r for r in records if r.get('guard_effective')]
    return dict(cases=len(guarded), activated_cases=sum(r['guard']['activations'] > 0 for r in guarded),
        **{key: sum(r['guard'][key] for r in guarded)
           for key in ('activations', 'reasks', 'second_correct', 'budget_blocked')})


def live_errors(rows, records):
    lookup = {r['task_id']: r for r in rows}
    expected, substitutions, examples = Counter(), Counter(), []
    failed = [r for r in records if r['mode'] == 'external_state' and not r['success']]
    repeated, before_first_done = 0, 0
    for record in failed:
        gold = teacher_trajectory(lookup[record['task_id']])
        index = record['first_error']
        if index is None or index >= len(gold):
            expected['budget_or_unresolved'] += 1
            continue
        step = gold[index]
        before_first_done += not any(s['action']['op'] == 'DONE' for s in gold[:index+1])
        expected[step['action']['op']] += 1
        try:
            action = json.loads(record['calls'][-1].get('text', ''))
        except ValueError:
            action = None
        op = action.get('op') if isinstance(action, dict) else 'malformed'
        if op == 'TRY':
            retried = action.get('production') in step['observation']['stack'][-1]['tried']
            repeated += retried
            op = 'TRY_already_tried' if retried else 'TRY_other_or_premature'
        elif op in ('PRN', 'REJ', 'CUT'):
            op += '(' + str(action.get('reason')) + ')'
        key = step['action']['op'] + ' -> ' + str(op)
        substitutions[key] += 1
        if len(examples) < 6 and not any(e['substitution'] == key for e in examples):
            examples.append(dict(task_id=record['task_id'], family=record['family'],
                target_answer=record['target_answer'], step=index, expected=step['action'], predicted=action,
                current_frame=step['observation']['stack'][-1], substitution=key))
    return dict(failed_cases=len(failed), expected_operator=dict(expected), substitutions=dict(substitutions),
        final_wrong_repeated_try=repeated, before_first_done=before_first_done, examples=examples)


def changes(old, new):
    before = {r['task_id']: r for r in old}
    after = {r['task_id']: r for r in new}
    if set(before) != set(after):
        raise ValueError('unpaired comparison')
    texts = lambda r: [c.get('text') for c in r['calls']]
    return dict(cases=len(before), improved=sum(not before[i]['success'] and after[i]['success'] for i in before),
        regressed=sum(before[i]['success'] and not after[i]['success'] for i in before),
        output_changed_cases=sum(texts(before[i]) != texts(after[i]) for i in before))


def assemble(root, baseline):
    instances, protocol = verify_study_inputs()
    frozen_baseline = read(baseline / 'baseline_analysis/audit.json.gz')
    checked_instances, checked_baselines, checked_metadata = assemble_baseline(baseline)
    if (frozen_baseline['instances'] != instances or checked_instances != instances
            or frozen_baseline['runs'] != checked_baselines):
        raise ValueError('baseline audit does not match the frozen suites')
    derived = dict(summary={s: {m: summarize([r for r in rs if r['mode'] == m]) for m in LANES}
        for s, rs in checked_baselines.items()},
        operators={s: operator_stats(instances[s], rs) for s, rs in checked_baselines.items()},
        costs={s: {m: cost([r for r in rs if r['mode'] == m]) for m in LANES}
            for s, rs in checked_baselines.items()}, metadata=checked_metadata)
    for name, value in derived.items():
        if read(baseline / 'baseline_analysis' / (name + '.json')) != value:
            raise ValueError('baseline analysis does not reproduce: ' + name)
    for suite in ('A', 'B'):
        path = baseline / suite / 'results/completion.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol['baseline_gate']['completion_sha256'][suite]:
            raise ValueError('baseline completion evidence changed')
    baseline_jobs = {s: read(baseline / s / 'status.json')['data']['job'] for s in ('A', 'B')}
    if any(job['status'] != 'SUCCEEDED' or not job['ended_at'] for job in baseline_jobs.values()):
        raise ValueError('baseline jobs are not terminal successes')
    baseline_end = max(datetime.fromisoformat(job['ended_at']) for job in baseline_jobs.values())
    runs, metadata = {}, {}
    for suite in SUITES:
        folder = root / suite / 'results'
        job = read(root / suite / 'status.json')['data']['job']
        if (job['status'] != 'SUCCEEDED' or job['id'] != 'abstractgym-offloading-' + suite.lower() + '-20261005'
                or datetime.fromisoformat(job['started_at']) < baseline_end):
            raise ValueError('hint job is incomplete or started before baseline completion')
        completion = read(folder / 'completion.json')
        if not completion['completed'] or completion['cases'] != 2880:
            raise ValueError('incomplete hint suite: ' + suite)
        runtime = read(folder / 'runtime.json')
        if runtime['suite'] != suite or runtime['protocol'] != protocol:
            raise ValueError('hint runtime mismatch')
        if any(runtime['packages'][name] != version for name, version in
               dict(vllm='0.11.2', transformers='4.57.6', huggingface_hub='0.36.0').items()):
            raise ValueError('serving package mismatch')
        staging = read(root / suite / 'staging.json')
        if any(sha != staging['files'][name] for name, sha in runtime['source_sha256'].items()):
            raise ValueError('deployed source mismatch')
        for name in ('controller.py', 'experiment.py', 'cfg.py', 'trace.py', 'oracle.py'):
            path = Path('src/abstractgym') / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != runtime['source_sha256'][str(path)]:
                raise ValueError('local replay differs from deployed core: ' + name)
        if staging['files']['experiments/offloading/protocol.json'] != hashlib.sha256(
                Path('experiments/offloading/protocol.json').read_bytes()).hexdigest():
            raise ValueError('deployed protocol mismatch')
        lookup = {r['task_id']: r for r in instances[suite]}
        runs[suite], per_condition = {}, {}
        for condition, config in CONDITIONS.items():
            marker = read(folder / (condition + '-completion.json'))
            if not marker['completed'] or marker['cases'] != 480:
                raise ValueError('incomplete hint condition')
            records = read(folder / (condition + '.jsonl'))
            keys = [(r['task_id'], r['mode']) for r in records]
            if (len(keys) != 480 or len(set(keys)) != 480 or set(keys) !=
                    {(i, m) for i in lookup for m in ('external_state', 'oracle_step')}):
                raise ValueError('missing or duplicate hint evidence')
            for record in records:
                if (record['condition'] != condition or record['hints'] != config['hints']
                        or record['guard_requested'] != config['legality_guard']
                        or record['guard_effective'] != (config['legality_guard'] and record['mode'] == 'external_state')):
                    raise ValueError('condition metadata mismatch')
                row = lookup[record['task_id']]
                if any(record[key] != row[key] for key in ('family', 'difficulty', 'target_answer')):
                    raise ValueError('instance metadata mismatch')
                replay(row, record, config)
                record['provenance'] = 'fresh'
            runs[suite][condition] = records
            per_condition[condition] = marker
            print('replayed ' + suite + ' ' + condition + ': 480 cases', flush=True)
        check = read(folder / 'preflight.json')
        if (not check['context_check_passed'] or len(check['cases']) != 240
                or {r['task_id'] for r in check['cases']} != set(lookup)
                or any(max(r['max_prompt_tokens'].values()) + 128 > 16384 for r in check['cases'])):
            raise ValueError('preflight evidence mismatch')
        metadata[suite] = dict(runtime=runtime, staging=staging, completion=completion,
            conditions=per_condition, preflight=check, job=job, baseline_jobs=baseline_jobs)
    return instances, frozen_baseline['runs'], runs, metadata, protocol


def analyze(instances, baselines, runs, metadata):
    values = dict(summary={}, operators={}, guards={}, costs={}, comparisons={}, live_errors={}, metadata=metadata)
    for suite in SUITES:
        for key in ('summary', 'operators', 'guards', 'costs', 'comparisons', 'live_errors'):
            values[key][suite] = {}
        for condition, records in runs[suite].items():
            by_mode = {m: [r for r in records if r['mode'] == m] for m in ('external_state', 'oracle_step')}
            values['summary'][suite][condition] = {m: summarize(rs) for m, rs in by_mode.items()}
            values['operators'][suite][condition] = operator_stats(instances[suite], records)
            values['guards'][suite][condition] = guard_stats(records)
            values['live_errors'][suite][condition] = live_errors(instances[suite], records)
            values['costs'][suite][condition] = {m: cost(rs) for m, rs in by_mode.items()}
            values['comparisons'][suite][condition] = {m: changes(
                [r for r in runs[suite]['H0'] if r['mode'] == m], rs) for m, rs in by_mode.items()}
        values['comparisons'][suite]['historical_vs_fresh_H0'] = {m: changes(
            [r for r in baselines[suite] if r['mode'] == m],
            [r for r in runs[suite]['H0'] if r['mode'] == m]) for m in ('external_state', 'oracle_step')}
    return values


def counts(cell):
    return f"{cell['correct']}/{cell['total']}"


def labels(cell):
    return counts(cell) + '; ' + '/'.join(map(str, cell['yes'])) + '; ' + '/'.join(map(str, cell['no']))


def operator_table(metrics):
    return table(['Operator', 'Correct / total', 'Accuracy', 'Wrong substitutions (counts)'], [
        [op, f"{m['correct']}/{m['total']}", f"{100*m['correct']/m['total']:.1f}%" if m['total'] else 'n/a',
         ', '.join(f'{k}: {v}' for k, v in sorted(m['mistakes'].items())) or 'none']
        for op in OPERATORS for m in [metrics[op]]])


def render(baseline, values, protocol):
    base = read(baseline / 'baseline_analysis/summary.json')
    base_ops = read(baseline / 'baseline_analysis/operators.json')
    base_cost = read(baseline / 'baseline_analysis/costs.json')
    shortcuts = read('experiments/countpreserving/shortcuts.json')
    generation = read('experiments/countpreserving/generation.json')
    parts = ['# Count-preserving puzzles and operator offloading: zero-shot Qwen3-14B',
        'All ordered parts are complete: construction and the shortcut gate, four zero-shot interfaces on A/B, '
        'then fresh H0-H5 in two interfaces on all four suites. The 480 new baseline cases per interface and '
        '11,520 hint-study cases are retained, including every failed response. The 3,840-case baseline comparison '
        'and all hint responses were replayed through the grader. No training, thinking mode, examples, repair, or '
        'model-outcome-driven case selection. Only H2/H3 live execution permits one bookkeeping re-ask.',
        '## Main comparison', table(['Interface', *SUITES], [
            [LABELS[m], *[score(base[s][m]['overall']) for s in SUITES]]
            for m in LANES]),
        'Original/cue-free baselines reuse the committed whole-run zero-shot audit; A/B are fresh. '
        'Direct-answer correctness is not execution correctness. Gym/full-trace success requires a complete '
        'valid canonical trajectory and the correct membership verdict. Correct-state all-case success requires '
        'every independently queried teacher-state action to be correct; it is not a deployed rollout.',
        '## What this answers',
        'A direct answer is correct on 187/240 (77.9%) and B on 146/240 (60.8%), versus chance 120/240. '
        'Every yes/no pair has identical count features, so a deterministic count-only rule cannot produce '
        'these above-chance results. This does not establish a general membership algorithm: A contains simple '
        'order-sensitive structures and B samples bounded linear CFGs. B also emits only 66 yes answers out of '
        '240 (27.5%), with 46/120 yes and 100/120 no correct. A emits 129/240 yes (53.8%), but its nesting '
        'family emits no on every case. An overall score must therefore be read with family/size and reply-bias tables.',
        'The following are matched, fresh H0 comparisons. Each value is all; yes; no, with failures retained. '
        'These are scaffolded zero-shot executions, not new model capabilities established without the scaffold.',
        table(['Suite', *CONDITIONS], [[s, *[labels(values['summary'][s][c]['external_state']['overall'])
            for c in CONDITIONS]] for s in SUITES])]
    for s in SUITES:
        before = values['summary'][s]['H0']['external_state']['overall']['correct']
        after = max(values['summary'][s][c]['external_state']['overall']['correct'] for c in CONDITIONS)
        best = ', '.join(c for c in CONDITIONS if values['summary'][s][c]['external_state']['overall']['correct'] == after)
        done = values['operators'][s]
        live = {c: values['summary'][s][c]['external_state']['overall']['correct'] for c in CONDITIONS}
        parts.append(f"{s}: the best complete-run condition(s) are {best}, {after}/240 versus fresh H0 {before}/240. "
            f"DONE accuracy is {done['H0']['DONE']['correct']}/{done['H0']['DONE']['total']} in H0, "
            f"{done['H1']['DONE']['correct']}/{done['H1']['DONE']['total']} in H1 and "
            f"{done['H3']['DONE']['correct']}/{done['H3']['DONE']['total']} in H3. "
            f"Adding comparisons (H4 versus H1) changes complete runs by {live['H4'] - live['H1']:+d}; "
            f"adding next_untried (H5 versus H4) changes them by {live['H5'] - live['H4']:+d}. "
            'Use the per-operator and wrong-substitution tables below to distinguish exhaustion assistance '
            'from comparison or production-selection assistance.')
    activations = sum(values['guards'][s][c]['activations'] for s in SUITES for c in ('H2', 'H3'))
    second_correct = sum(values['guards'][s][c]['second_correct'] for s in SUITES for c in ('H2', 'H3'))
    max_done = max(values['operators'][s][c]['DONE']['correct'] / values['operators'][s][c]['DONE']['total']
                   for s in SUITES for c in CONDITIONS)
    negative_completions = sum(values['summary'][s][c]['external_state']['overall']['no'][0]
                               for s in SUITES for c in CONDITIONS)
    failed_before_done = sum(values['live_errors'][s]['H0']['before_first_done'] for s in SUITES)
    h0_failed = sum(values['live_errors'][s]['H0']['failed_cases'] for s in SUITES)
    parts.append(f"The guard activates {activations} times across 1,920 guarded live cases. "
        f"Its second answer is correct {second_correct}/{activations} times. "
        f"In fresh H0, {failed_before_done}/{h0_failed} failed live cases stop before the first required DONE. "
        'These deployed first errors must be distinguished from frequent DONE mistakes on independently '
        'supplied teacher states. Listing remaining rules does not by itself establish an exhaustion solution; '
        'the DONE rows and complete negative-run counts are the relevant evidence.')
    parts.append(f"The highest DONE accuracy in any suite/condition is {100*max_done:.1f}%, not near ceiling. "
        f"Across the hint study there are only {negative_completions} complete negative runs; the table above "
        'locates them by suite and condition. H5 adds no complete runs over H4 on any suite. '
        'The proposed exhaustion rescue is therefore not observed under this fixed zero-shot protocol. '
        'The strongest supported conclusion is narrower: remaining-rule hints improve some positive '
        'executions, while earlier action/argument mistakes still dominate deployed failures.')
    parts += ['The separately trained X7 step adapter completes 133/240 original and 118/240 cue-free cases. '
        'Those scores provide scale only: X7 received training; H0-H5 did not. The new count-preserving study '
        'does not evaluate X7 on A/B. Strong H5 explicitly delegates first-untried production selection to code.',
        '## Shortcut gate',
        'Before any model call, all 129 checks per suite scored exactly 50% overall and in every nonempty '
        'family, size, and family/size cell. The rule battery includes symbol-presence checks, lengths, letter '
        'counts, rule/placeholder counts, parity, ratios, constants, and a bag-of-symbols logistic classifier. '
        'Classifier predictions are four-fold out-of-fold, holding entire pairs together and fitting the '
        'vectorizer only on training folds. Pairwise feature identity makes the chance result stronger than '
        'a finite battery: any deterministic function of these count features predicts both labels identically.',
        table(['Suite', 'Cell', 'Cases', 'Minimum over checks', 'Maximum over checks'], [
            [s, 'overall', 240, '50.0%', '50.0%'] for s in ('A', 'B')] + [
            [s, 'family: ' + f, cell[1], '50.0%', '50.0%']
            for s in ('A', 'B') for f, cell in next(iter(shortcuts[s]['scores'].values()))['by_family'].items()] + [
            [s, 'size: ' + n, cell[1], '50.0%', '50.0%']
            for s in ('A', 'B') for n, cell in sorted(
                next(iter(shortcuts[s]['scores'].values()))['by_size'].items(), key=lambda item: int(item[0]))]),
        '### Every shortcut (overall)', table(['Check', 'A', 'B'], [
            [name, '/'.join(map(str, shortcuts['A']['scores'][name]['overall'])),
             '/'.join(map(str, shortcuts['B']['scores'][name]['overall']))]
            for name in sorted(shortcuts['A']['scores'])]),
        '### Construction and scope',
        'A has six families, five sizes (1, 2, 4, 8, 16), four letterings and balanced labels. '
        'Order-sensitive blocks contain two symbols, so length/sequence level n has input length 2n; '
        'nesting uses n openings of each of two types, actual nesting 2n, even at level one. '
        'B has 120 independent random linear grammars, 3-6 nonterminals, 2-3 rules per nonterminal, '
        '3-4 declared symbols, and successful canonical depths 1, 2, 4, 8, 16 with 24 pairs per depth. '
        'A B-negative swaps two different adjacent input symbols; only Earley-rejected, bounded-DFS-completing '
        'pairs survive. Both partners retain all bags/counts. No epsilon or left recursion; maximum 512 search '
        'steps. Grammar reuse, correlated letterings and rejection sampling limit generalization.',
        '```json\n' + json.dumps(generation, sort_keys=True, indent=2) + '\n```',
        '## Part 2: baseline atlas',
        'Each entry is all correct/total; yes correct/total; no correct/total. A/B sizes are not '
        'paired structural equivalents of original/cue-free sizes; inspect actual string lengths and accepting '
        'depths in the frozen instances. Missing families are n/a, not zero-score cases.']
    for mode in LANES:
        parts += ['### ' + LABELS[mode], table(['Suite', 'All', 'Yes', 'No'], [
            [s, counts(base[s][mode]['overall']), '/'.join(map(str, base[s][mode]['overall']['yes'])),
             '/'.join(map(str, base[s][mode]['overall']['no']))] for s in SUITES]),
            '#### Per family', table(['Family', *SUITES], [[f, *[
                labels(base[s][mode]['by_family'][f]) if f in base[s][mode]['by_family'] else 'n/a'
                for s in SUITES]] for f in sorted({f for s in SUITES for f in base[s][mode]['by_family']})]),
            '#### Per size', table(['Size', *SUITES], [[n, *[labels(base[s][mode]['by_level'][str(n)])
                for s in SUITES]] for n in (1, 2, 4, 8, 16)])]
    parts += ['### Direct-answer yes-rate and malformed replies',
        'The all-case yes-rate counts malformed replies as neither yes nor no. The valid-reply rate is '
        'reported separately; no malformed response is removed from accuracy denominators.']
    for group in ('by_family', 'by_level'):
        parts += ['#### ' + ('Per family' if group == 'by_family' else 'Per size'),
            table(['Suite', 'Cell', 'Yes / all cases', 'Yes-rate (all)', 'Yes-rate (valid)', 'Malformed'], [
                [s, k, f"{v['reply_bias']['yes']}/{v['reply_bias']['cases']}",
                 f"{100*v['reply_bias']['yes_rate_all_cases']:.1f}%",
                 f"{100*v['reply_bias']['yes_rate_valid_replies']:.1f}%" if v['reply_bias']['yes_rate_valid_replies'] is not None else 'n/a',
                 v['reply_bias']['malformed']] for s in SUITES for k, v in sorted(
                    base[s]['direct'][group].items(), key=lambda item: int(item[0]) if group == 'by_level' else item[0])])]
    parts += ['### Baseline correct-state operator accuracy',
        'DONE is shown first. TRY_already_tried is a wrong TRY naming a production already in the current '
        "teacher frame's tried list, not just an arbitrary wrong production. Every queried state remains in the denominator."]
    for s in SUITES:
        parts += ['#### ' + s, operator_table(base_ops[s])]
    parts += ['## Part 3: offloading atlas', table(['Condition', 'Hints', 'Live guard'], [
        [c, ', '.join(config['hints']) or 'none', 'one re-ask' if config['legality_guard'] else 'off']
        for c, config in CONDITIONS.items()]),
        'Each suite has one serving job with fresh H0-H5 in order. The current frame alone gets enabled '
        'fields; no hint names an opcode. Enabled prompts add the same one-sentence explanation, independent '
        'of condition. The unmodified prompt and observation are preserved byte-for-byte with hints off. '
        'untried_rules and next_untried use grammar order and the frame tried list; prefix_matches uses the '
        'validator classification without its depth cutoff; complete_match requires exact terminal equality. '
        'Hints do not mutate the stored state or the grader.',
        'The guard applies only in gym memory. A repeated TRY is not executed; the same state is queried '
        'once more with exactly "pK was already tried here". The second answer is final, even if it repeats '
        'the error or is malformed. All calls, including rejected first answers and second answers, count '
        'toward 512; no correct opcode or reference action is supplied. The activation decision uses only '
        'the tried list. H2 correct-state checks therefore equal the H0 interface; H3 equals H1. They are '
        'fresh queries, not reused responses, and deliberately do not measure guard-assisted accuracy.',
        '### Correct-state all-case and action-level accuracy',
        table(['Suite', 'Condition', 'All actions correct (cases)', 'Correct actions / all states'], [
            [s, c, counts(values['summary'][s][c]['oracle_step']['overall']),
             '/'.join(map(str, values['summary'][s][c]['oracle_step']['overall']['action_accuracy']))]
            for s in SUITES for c in CONDITIONS])]
    for s in SUITES:
        parts += ['### ' + s + ': complete gym-memory runs',
            'Entries are all; yes; no. Each complete-run denominator is 240 overall, with 120 per label.',
            table(['Condition', 'All', 'Yes', 'No'], [[c,
                counts(values['summary'][s][c]['external_state']['overall']),
                '/'.join(map(str, values['summary'][s][c]['external_state']['overall']['yes'])),
                '/'.join(map(str, values['summary'][s][c]['external_state']['overall']['no']))]
                for c in CONDITIONS])]
        for group in ('by_family', 'by_level'):
            cells = values['summary'][s]['H0']['external_state'][group]
            parts += ['#### ' + ('Per family' if group == 'by_family' else 'Per size'),
                table(['Cell', *CONDITIONS], [[k, *[labels(values['summary'][s][c]['external_state'][group][k])
                    for c in CONDITIONS]] for k in cells])]
        for c in CONDITIONS:
            parts += ['#### ' + s + ' ' + c + ': raw correct-state operators', operator_table(values['operators'][s][c])]
    parts += ['### Guard outcomes', table(['Suite', 'Condition', 'Cases activated', 'Activations', 'Re-asks',
        'Second answers correct', 'Retries blocked by call cap'], [[s, c,
            f"{v['activated_cases']}/{v['cases']}", v['activations'], v['reasks'], v['second_correct'], v['budget_blocked']]
            for s in SUITES for c in ('H2', 'H3') for v in [values['guards'][s][c]]]),
        'Second-answer correctness is the unchanged canonical action test at the same state, not merely '
        'the absence of another repeated rule. A corrected second action does not guarantee the whole run finishes.',
        '### Live first errors versus teacher-state errors',
        'Teacher-state queries deliberately reach states that a flawed live execution may never reach. '
        'A large teacher-state DONE error count is therefore not automatically the first deployed bottleneck. '
        'The following counts classify the expected operator at the first final live error; corrected guard '
        'attempts are not counted as final failures. Wrong reason strings and wrong production IDs remain '
        'strict protocol failures even when an opcode appears plausible.',
        table(['Suite', 'Condition', 'Failed live cases', *OPERATORS, 'Budget / unresolved'], [[s, c,
            values['live_errors'][s][c]['failed_cases'],
            *[values['live_errors'][s][c]['expected_operator'].get(op, 0) for op in OPERATORS],
            values['live_errors'][s][c]['expected_operator'].get('budget_or_unresolved', 0)]
            for s in SUITES for c in CONDITIONS]),
        'First-error substitutions and inspectable examples are saved in '
        '[live_errors.json](results/countpreserving_offloading/live_errors.json). '
        'If a guard condition records zero activations, it supplied no re-ask or changed prompt; any score '
        'difference from its unguarded counterpart is not evidence for guard effectiveness. '
        'This study must distinguish latent exhaustion errors from earlier action/argument errors.',
        '### Matched changes versus fresh H0', table(['Suite', 'Condition', 'Improved gym cases', 'Regressed gym cases'], [
            [s, c, values['comparisons'][s][c]['external_state']['improved'],
             values['comparisons'][s][c]['external_state']['regressed']] for s in SUITES for c in CONDITIONS]),
        '### Historical baseline versus fresh H0', table(['Suite', 'Interface', 'Outputs changed (cases)', 'Improved', 'Regressed'], [
            [s, LABELS[m], v['output_changed_cases'], v['improved'], v['regressed']]
            for s in SUITES for m, v in values['comparisons'][s]['historical_vs_fresh_H0'].items()]),
        'Fresh H0, not a historical score, is the comparator for hint effects. Prompts and nominal batching '
        'are matched; greedy inference can still have numerical/runtime differences. No output was used '
        'to choose a rerun, prompt change, or case exclusion.',
        '## Failure accounting and call budgets',
        table(['Part', 'Suite', 'Condition', 'Interface', 'Failure', 'Cases'], [
            ['baseline', s, 'zero-shot', LABELS[m], failure, n] for s in SUITES for m in LANES
            for failure, n in base[s][m]['overall']['failures'].items()] + [
            ['offloading', s, c, LABELS[m], failure, n] for s in SUITES for c in CONDITIONS
            for m in ('external_state', 'oracle_step')
            for failure, n in values['summary'][s][c][m]['overall']['failures'].items()]),
        'Malformed responses, invalid actions, incomplete traces and exhausted budgets count as failures. '
        'Correct-state failures count per state and per all-actions-correct case; teacher-state checks continue '
        'after an error without revealing the answer. Live execution stops on the first final invalid action. '
        'There was no injected infrastructure/tool-failure recovery test.',
        '### Baseline costs', table(['Suite', 'Interface', 'Calls', 'Completion tokens', 'Total tokens', 'Length-limited', 'Fresh cases'], [
            [s, LABELS[m], v['calls'], v['completion_tokens'], v['total_tokens'], v['length_limited_calls'], v['fresh_cases']]
            for s in SUITES for m, v in base_cost[s].items()]),
        '### Hint-study costs', table(['Suite', 'Condition', 'Interface', 'Calls', 'Prompt tokens', 'Completion tokens',
            'Total tokens', 'Length-limited', 'Inference seconds'], [[s, c, LABELS[m], v['calls'],
                v['prompt_tokens'], v['completion_tokens'], v['total_tokens'], v['length_limited_calls'],
                f"{v['inference_seconds']:.1f}"] for s in SUITES for c in CONDITIONS
                for m, v in values['costs'][s][c].items()]),
        '### Evaluation wall time per condition', table(['Suite', *CONDITIONS], [[s, *[
            f"{values['metadata'][s]['conditions'][c]['seconds']:.1f}s" for c in CONDITIONS]] for s in SUITES]),
        'Inference seconds are batch wall time apportioned over independent prompts, then summed; '
        'evaluation wall time includes grading and durable snapshots, not cluster startup, dependencies, '
        'model loading or preflight. Four suites run concurrently on one B200 per job. Early failed runs '
        'can cost less than successful runs simply because they stop sooner. Token totals include every guard call.',
        '## Reproducibility and interpretation limits',
        'Base Qwen/Qwen3-14B at revision ' + protocol['settings']['revision'] + '; BF16, temperature 0, '
        'top-p 1, top-k -1, seed 17, thinking off, LoRA disabled. Direct cap 128, full trace 4,096, '
        'actions 128, context 16,384, at most 512 calls including retries. Independent direct and '
        'correct-state batches have at most 32 prompts; teacher batches stay within a case, matching the '
        'baseline harness. Live and full-trace calls are serial. Runtime package versions, hardware, '
        'source hashes, immutable inputs, prompts, outputs, grades, token use and timing are retained.',
        'One seed, correlated variants, selected bounded grammars, and structural changes across suites '
        'limit generalization. The shortcut result excludes deterministic count-only decisions on these '
        'matched pairs, not structural/order-sensitive shortcuts. B is a rejection-sampled linear-CFG '
        'subset, not arbitrary CFG membership. Offloading measures the coupled model-plus-harness system; '
        'it does not by itself show that the model learned exhaustion, comparison, search or backtracking. '
        'The strong next_untried hint delegates selection to code. No fine-tuning or RL was run in this study. '
        'Existing whole-run/cue-free results, prompts and data were not rewritten.',
        '[Frozen hint protocol](../experiments/offloading/protocol.json), '
        '[count-preserving instances and gate](../experiments/countpreserving/manifest.json), '
        '[full shortcut cells](../experiments/countpreserving/shortcuts.json), '
        '[baseline summary](results/countpreserving_offloading/baseline_summary.json), '
        '[hint summary](results/countpreserving_offloading/summary.json), '
        '[operator substitutions](results/countpreserving_offloading/operators.json), '
        '[guard outcomes](results/countpreserving_offloading/guards.json), '
        '[costs](results/countpreserving_offloading/costs.json), '
        '[runtime and deployment metadata](results/countpreserving_offloading/metadata.json), '
        '[paired comparisons](results/countpreserving_offloading/comparisons.json), '
        '[raw calls and instances](results/countpreserving_offloading/audit.json.gz) accompany this report.']
    return '\n\n'.join(parts) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('build/offloading-20261005'))
    parser.add_argument('--baseline', type=Path, default=Path('build/countpreserving-20261005'))
    parser.add_argument('--output', type=Path, default=Path('docs/results/countpreserving_offloading'))
    parser.add_argument('--report', type=Path, default=Path('docs/countpreserving_offloading_results.md'))
    args = parser.parse_args()
    instances, baselines, runs, metadata, protocol = assemble(args.root, args.baseline)
    values = analyze(instances, baselines, runs, metadata)
    report = render(args.baseline, values, protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        (args.output / (name + '.json')).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    for name in ('summary', 'operators', 'costs', 'metadata'):
        (args.output / ('baseline_' + name + '.json')).write_bytes(
            (args.baseline / 'baseline_analysis' / (name + '.json')).read_bytes())
    audit = dict(instances=instances, baseline=baselines, offloading=runs, protocol=protocol)
    (args.output / 'audit.json.gz').write_bytes(gzip.compress(json.dumps(audit, sort_keys=True).encode(), mtime=0))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                if p.name != 'manifest.json'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    args.report.write_text(report)
    print('report_complete: all 11520 hint cases replayed, 3840 baseline comparison cases retained')


if __name__ == '__main__':
    main()
