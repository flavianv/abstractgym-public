"""Replay all whole-run evidence, preserve denominators, and write the atlas."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from abstractgym.complexity import FAMILIES, LEVELS
from abstractgym.experiment import run_case
from abstractgym.wholerun import ARMS, written_diagnostic
from a6_prompts import member_prompt
from analyze_a6_x9 import correlation

LANES = ('direct', 'full_trace', 'external_state', 'oracle_step')
CONDITIONS = ('zeroshot', 'dfs_lora') + ARMS


def read(path):
    path = Path(path)
    if path.name.endswith('.gz'):
        return json.loads(gzip.decompress(path.read_bytes()))
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def replay(row, record):
    if record['mode'] == 'direct':
        assert record.get('prompt', record['model_prompt']) == record['model_prompt'] == member_prompt(row)
        assert record['max_tokens'] == 128
        try:
            parsed = json.loads(record['text'])
        except ValueError:
            parsed = None
        expected = {'accept': row['target_answer'] == 'accept'}
        success = isinstance(parsed, dict) and type(parsed.get('accept')) is bool and parsed == expected
        assert success == record['success']
        return
    index = 0
    def adapter(prompt):
        nonlocal index
        call = record['calls'][index]
        index += 1
        assert prompt == call['prompt'] == call['model_prompt']
        assert call['max_tokens'] == (4096 if record['mode'] == 'full_trace' else 128)
        assert call['batch_size'] == (min(32, len(record['calls']) - (index - 1) // 32 * 32)
            if record['mode'] == 'oracle_step' else 1)
        return call
    result = run_case(row, record['mode'], adapter, max_calls=512)
    assert index == len(record['calls'])
    for key in ('success', 'failure', 'first_error', 'correct_actions', 'target_actions', 'outcome'):
        assert result[key] == record[key], (row['task_id'], record['mode'], key)


def assemble(root):
    data = Path('experiments/wholerun')
    protocol = read(data / 'protocol.json')
    for name, sha in read(data / 'manifest.json').items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == sha
    for name, sha in protocol['historical_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
    x7 = read('docs/results/2026-10-04-x7/audit.json.gz')
    cue = read('docs/results/cuefree/audit.json.gz')
    instances = {s: read(data / (s + '.json')) for s in ('original', 'cuefree')}
    assert instances['original'] == x7['instances'] and instances['cuefree'] == cue['instances']
    runs = {s: {} for s in instances}
    meta = {}
    def decorate(records, mode=None, provenance='fresh'):
        return [dict(r, **({'mode': mode} if mode else {}), provenance=provenance) for r in records]
    for arm in CONDITIONS:
        folder = root / arm
        assert read(folder / 'completion.json')['completed'], arm
        meta[arm] = {name: read(folder / (name + '.json')) for name in ('runtime', 'completion')}
        assert meta[arm]['runtime']['protocol'] == protocol
        if arm in ARMS:
            for name in ('training', 'initialization_parity', 'reload_parity', 'checkpoint_hashes'):
                meta[arm][name] = read(folder / (name + '.json'))
            for name in ('initialization_parity', 'reload_parity'):
                assert meta[arm][name]['allclose'] and meta[arm][name]['mean_abs'] <= .05
            for suite in instances:
                records = decorate(read(folder / suite / 'execution.jsonl'))
                records += decorate(read(folder / suite / 'direct.json'), 'direct')
                runs[suite][arm] = records
        else:
            if arm == 'dfs_lora':
                meta[arm]['checkpoint_hashes'] = read(folder / 'checkpoint_hashes.json')
                assert meta[arm]['checkpoint_hashes'] == read(
                    'docs/results/2026-10-04-x7/metadata.json')['training']['checkpoint_hashes']
            historical = x7['runs'][arm]
            direct = x7['membership'] if arm == 'dfs_lora' else cue['historical_direct']
            runs['original'][arm] = decorate(historical, provenance='historical original') + decorate(direct, 'direct', 'historical original')
            missing = decorate(read(folder / 'cuefree/execution.jsonl'))
            if arm == 'zeroshot':
                missing += decorate(cue['execution'], provenance='cue-free prior audit')
                missing += decorate([r for r in historical if r['mode'] == 'oracle_step' and r['target_answer'] == 'accept'], provenance='identical original yes')
                missing += decorate(cue['direct'], 'direct', 'cue-free prior audit')
            else:
                missing += decorate([r for r in historical if r['target_answer'] == 'accept'], provenance='identical original yes')
                missing += decorate(read(folder / 'cuefree/direct.json'), 'direct')
            runs['cuefree'][arm] = missing
    for suite, arms in runs.items():
        lookup = {r['task_id']: r for r in instances[suite]}
        for arm, records in arms.items():
            assert len(records) == 960, (suite, arm, len(records))
            assert {(r['task_id'], r['mode']) for r in records} == {(i, m) for i in lookup for m in LANES}
            for r in records:
                row = lookup[r['task_id']]
                r.update(family=row['family'], difficulty=row['difficulty'], target_answer=row['target_answer'])
                replay(row, r)
    for arm in CONDITIONS:
        comparison = {}
        for mode in LANES:
            old = {r['task_id']: r for r in runs['original'][arm]
                if r['mode'] == mode and r['target_answer'] == 'accept'}
            new = {r['task_id']: r for r in runs['cuefree'][arm]
                if r['mode'] == mode and r['target_answer'] == 'accept'}
            assert set(old) == set(new) and len(old) == 120
            texts = lambda r: [r['text']] if mode == 'direct' else [c.get('text') for c in r['calls']]
            comparison[mode] = dict(cases=120,
                output_changed_task_ids=[i for i in old if texts(old[i]) != texts(new[i])],
                success_changed_task_ids=[i for i in old if old[i]['success'] != new[i]['success']])
        meta[arm]['identical_yes_comparison'] = comparison
    return instances, runs, meta, protocol


def cell(records):
    return dict(correct=sum(r['success'] for r in records), total=len(records),
        yes=[sum(r['success'] for r in records if r['target_answer'] == 'accept'), sum(r['target_answer'] == 'accept' for r in records)],
        no=[sum(r['success'] for r in records if r['target_answer'] == 'reject'), sum(r['target_answer'] == 'reject' for r in records)],
        failures=dict(Counter(r.get('failure') or 'incorrect_answer' for r in records if not r['success'])),
        action_accuracy=[sum(r.get('correct_actions', 0) for r in records), sum(r.get('target_actions', 0) for r in records)])


def summarize(records):
    return dict(overall=cell(records),
        by_family={f: cell([r for r in records if r['family'] == f]) for f in FAMILIES},
        by_level={str(n): cell([r for r in records if r['difficulty']['level'] == n]) for n in LEVELS},
        family_level={f: {str(n): cell([r for r in records if r['family'] == f and r['difficulty']['level'] == n])
            for n in LEVELS} for f in FAMILIES})


def cost(records):
    calls = [c for r in records for c in ([r] if r['mode'] == 'direct' else r['calls'])]
    fresh = [c for r in records if r['provenance'] == 'fresh'
        for c in ([r] if r['mode'] == 'direct' else r['calls'])]
    return dict(calls=len(calls), completion_tokens=sum(c['usage']['completion_tokens'] for c in calls),
        prompt_tokens=sum(c['usage']['prompt_tokens'] for c in calls),
        total_tokens=sum(c['usage']['total_tokens'] for c in calls),
        length_limited_calls=sum(c['finish_reason'] == 'length' for c in calls),
        inference_seconds=sum(c['inference_seconds'] for c in calls),
        fresh_completion_tokens=sum(c['usage']['completion_tokens'] for c in fresh),
        fresh_total_tokens=sum(c['usage']['total_tokens'] for c in fresh),
        fresh_cases=sum(r['provenance'] == 'fresh' for r in records),
        reused_cases=sum(r['provenance'] != 'fresh' for r in records))


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def diagnostic_summary(pairs):
    failed = [r for r in pairs if not r['written_success']]
    both = [r for r in failed if r['first_wrong_move'] is not None and r['gym_first_error'] is not None]
    observed = [r for r in failed if r['first_wrong_move'] is not None]
    return dict(total=len(pairs), written_correct=sum(r['written_success'] for r in pairs),
        gym_correct=sum(r['gym_success'] for r in pairs),
        failed=len(failed), observed_wrong=sum(r['first_wrong_move'] is not None for r in failed),
        completed_prefix_format_failures=sum(r.get('canonical_prefix_completed', False) for r in failed),
        trailing_first_errors=sum(r.get('first_wrong_is_trailing', False) for r in failed),
        missing_without_wrong=sum(r['first_missing_move'] is not None for r in failed),
        no_observable_move_error=sum(r['first_wrong_move'] is None and r['first_missing_move'] is None for r in failed),
        median_wrong_index=median([r['first_wrong_move'] for r in failed]),
        median_wrong_fraction=median([r['first_wrong_fraction'] for r in failed]),
        median_correct_prefix=median([r['correct_prefix_actions'] for r in failed]),
        wrong_index_length_spearman=correlation([r['canonical_actions'] for r in observed],
            [r['first_wrong_move'] for r in observed]),
        paired_observed_errors=len(both),
        paired_written_index=median([r['first_wrong_move'] for r in both]),
        paired_gym_index=median([r['gym_first_error'] for r in both]),
        paired_written_fraction=median([r['first_wrong_fraction'] for r in both]),
        paired_gym_fraction=median([r['gym_first_error_fraction'] for r in both]))


def analyze(instances, runs):
    summary, diagnostics, costs = {}, {}, {}
    for suite, arms in runs.items():
        lookup = {r['task_id']: r for r in instances[suite]}
        summary[suite], diagnostics[suite], costs[suite] = {}, {}, {}
        for arm, records in arms.items():
            summary[suite][arm] = {m: summarize([r for r in records if r['mode'] == m]) for m in LANES}
            costs[suite][arm] = {m: cost([r for r in records if r['mode'] == m]) for m in LANES}
            gym = {r['task_id']: r for r in records if r['mode'] == 'external_state'}
            pairs = []
            for record in records:
                if record['mode'] != 'full_trace':
                    continue
                row = lookup[record['task_id']]
                live = gym[record['task_id']]
                d = written_diagnostic(row, record)
                pairs.append(dict(d, family=row['family'], level=row['difficulty']['level'],
                    target_answer=row['target_answer'], written_success=record['success'], gym_success=live['success'],
                    gym_first_error=live['first_error'], gym_first_error_fraction=
                    live['first_error'] / d['canonical_actions'] if live['first_error'] is not None else None))
            diagnostics[suite][arm] = dict(pairs=pairs, overall=diagnostic_summary(pairs),
                failure_categories=dict(Counter(r['category'] for r in pairs if r['category'])),
                by_family={f: diagnostic_summary([r for r in pairs if r['family'] == f]) for f in FAMILIES},
                by_length={label: diagnostic_summary([r for r in pairs if lo <= r['canonical_actions'] <= hi])
                    for label, lo, hi in [('1-8', 1, 8), ('9-16', 9, 16), ('17-32', 17, 32), ('33-64', 33, 64), ('65+', 65, 100000)]})
    return summary, diagnostics, costs


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def frac(pair):
    return str(pair[0]) + '/' + str(pair[1])


def score(c):
    return f"{c['correct']}/{c['total']} ({100 * c['correct'] / c['total']:.1f}%)" if c['total'] else '0/0'


def number(v):
    return 'not observed' if v is None else f'{v:.3f}'


def report(summary, diagnostics, costs, meta, protocol):
    lane_names = dict(direct='Direct answer', full_trace='Written run', external_state='Gym memory', oracle_step='Correct-state check')
    lines = ['# Whole written runs: Qwen3-14B LoRA', '',
        'All three new adapters and both reference conditions have complete four-lane results on both 240-case suites. '
        'Every saved response was replayed through the unchanged grader. Original-suite reference outputs and identical '
        'cue-free yes executions are reused explicitly; all new-arm cases are freshly evaluated.', '',
        '## Main comparison', '']
    for suite in summary:
        lines += ['### ' + suite.capitalize() + ' suite', '', table(
            ['Arm', 'Direct /240', 'Written /240', 'Gym /240', 'Correct-state /240'],
            [[arm] + [score(summary[suite][arm][m]['overall']) for m in LANES] for arm in CONDITIONS]), '']
    lines += ['All failures remain in every denominator. One training/decoding seed (17), thinking off, greedy decoding, '
        'no examples, no retries, and no repairs. Written and gym success require complete valid canonical DFS execution; '
        'direct answer accuracy is a separate invocation. Correct-state checks supply each canonical state independently and '
        'require every action to be correct; they do not establish autonomous completion. '
        'There are no exact grammar/input matches between the 64 training instances and either evaluation suite.', '', '## What this answers', '']
    for suite in summary:
        lengths = [r['canonical_actions'] for r in diagnostics[suite]['zeroshot']['pairs']]
        lines += [f'The {suite} canonical traces span {min(lengths)}-{max(lengths)} actions; '
            f'{sum(n > 512 for n in lengths)} targets exceed the 512-action-call budget.', '']
    for suite in summary:
        arms = summary[suite]
        lines += [f"On {suite}, written runs are " + ', '.join(f"{arm}: {arms[arm]['full_trace']['overall']['correct']}/240" for arm in CONDITIONS) + '.', '']
        mixed = arms['mixed_sft']['external_state']['overall']['correct']
        reference = arms['dfs_lora']['external_state']['overall']['correct']
        written = arms['mixed_sft']['full_trace']['overall']['correct']
        lines += [f'Mixed training gets {mixed}/240 gym-memory completions versus the matched-suite X7 reference {reference}/240 '
            f'({mixed - reference:+d} cases), with {written}/240 written completions. The original X7 reference is 133/240.', '']
        baseline = arms['zeroshot']['full_trace']['overall']['correct']
        lines += [f'Written mixed-training completions improve by {written - baseline} cases over zero-shot '
            f'({baseline}/240), but remain {mixed - written} cases below the same adapter with computer-held state. '
            f'Direct verdict accuracy is {arms["mixed_sft"]["direct"]["overall"]["correct"]}/240 versus '
            f'{arms["zeroshot"]["direct"]["overall"]["correct"]}/240 zero-shot; better execution does not imply '
            'better direct verdicts.', '']
    lines += ['This is an intermediate outcome, not a clean confirmation of either extreme in the brief. '
        'Whole-run practice materially improves complete written executions, so X7\'s zero cannot by itself '
        'establish an inability to execute in the model\'s own output. The mixed arm retains and improves '
        'the computer-held-state gain, while written completion remains substantially lower. The tested '
        'full-only budgets are just six and eight optimizer updates; the mixed arm also weights its rare '
        'full examples differently. This does not establish an optimization-independent state-carrying limit.', '']
    lines += ['The tables remove the specific absence-of-whole-run-training confound at the tested budgets. '
        'Assess the written-run gains alongside failure categories and paired first-error positions below. '
        'A later observed mistake in long traces is consistent with difficulty maintaining state; early format errors '
        'or a correct prefix that stops support a different explanation. These outputs do not identify an internal mechanism, '
        'and this one-seed study does not establish a general CFG membership or arbitrary-program execution procedure.', '']
    lines += ['The whole-trace-only arms saw no single-step output targets. Low next-action or gym-memory scores '
        'for those arms can therefore reflect the reverse interface-format confound. The mixed arm trains both '
        'interfaces and provides the most direct test of gaining written completions while retaining X7\'s '
        'gym-memory behavior. Training amounts and loss weighting still differ, as recorded below.', '']
    lines += ['## Training amount', '', table(['Arm', 'Rows', 'Epochs touched', 'Updates', 'Target tokens',
        'Processed tokens', 'Train seconds', 'Last loss'], [[arm, meta[arm]['training']['training_examples'],
        meta[arm]['training']['epochs_touched'], meta[arm]['training']['optimizer_updates'],
        meta[arm]['training']['scheduled_supervised_tokens'], meta[arm]['training']['scheduled_processed_tokens'],
        round(meta[arm]['training']['training_seconds']), f"{meta[arm]['training']['last_train_loss']:.6g}"] for arm in ARMS]), '',
        'X7 processed 295,089 target tokens including EOS, 27,750,309 nonpadding tokens, and 1,032 optimizer updates. '
        f"Its historical training took {protocol['training']['training_seconds']:.0f} seconds; that adapter is reused, not retrained. "
        'The matched arm matches target tokens at the first effective batch reaching that budget; it does not match optimizer updates. '
        'Full targets contain the unchanged raw action array through completion, whose final action supplies the controller verdict. '
        'The mixed arm includes every X7 state example plus all 64 full traces, shuffled jointly in each of three epochs.', '']
    matched_tokens = meta['fulltrace_sft_matched']['training']['scheduled_supervised_tokens']
    lines += [f'The matched arm exceeds the target-token budget by {matched_tokens - 295089:,} tokens '
        f'({100 * (matched_tokens / 295089 - 1):.2f}%). Matching target tokens leaves a large optimizer-update '
        'difference because whole traces aggregate many actions into each example. Loss is averaged over unmasked '
        'tokens within each microbatch, following X7. In mixed training the 64 full traces are only 0.58% of examples; '
        'their length and microbatch loss weighting mean row share and target-token share are different quantities.', '']
    for suite in summary:
        for mode in LANES:
            lines += ['## ' + suite.capitalize() + ': ' + lane_names[mode], '', '### Per arm and label', '', table(
                ['Arm', 'All', 'Yes', 'No', 'Action accuracy'], [[arm, score(summary[suite][arm][mode]['overall']),
                frac(summary[suite][arm][mode]['overall']['yes']), frac(summary[suite][arm][mode]['overall']['no']),
                frac(summary[suite][arm][mode]['overall']['action_accuracy']) if mode != 'direct' else 'n/a'] for arm in CONDITIONS]), '',
                '### Per family', '', table(['Family', 'Arm', 'All', 'Yes', 'No'],
                [[f, arm, score(summary[suite][arm][mode]['by_family'][f]),
                    frac(summary[suite][arm][mode]['by_family'][f]['yes']), frac(summary[suite][arm][mode]['by_family'][f]['no'])]
                    for f in FAMILIES for arm in CONDITIONS]), '',
                '### Per level', '', table(['Level', 'Arm', 'All', 'Yes', 'No'],
                [[n, arm, score(summary[suite][arm][mode]['by_level'][str(n)]),
                    frac(summary[suite][arm][mode]['by_level'][str(n)]['yes']), frac(summary[suite][arm][mode]['by_level'][str(n)]['no'])]
                    for n in LEVELS for arm in CONDITIONS]), '']
    lines += ['## Failure accounting', '', table(['Suite', 'Arm', 'Lane', 'Failure', 'Cases'],
        [[suite, arm, mode, failure, n] for suite in summary for arm in CONDITIONS for mode in LANES
            for failure, n in summary[suite][arm][mode]['overall']['failures'].items()]), '',
        '## Written-run failure accounting', '', table(['Suite', 'Arm', 'One move then stop',
        'Cap-length loop', 'Malformed/invalid/trailing', 'Other incomplete', 'Failed /240'],
        [[suite, arm] + [diagnostics[suite][arm]['failure_categories'].get(k, 0) for k in
            ('one_move_then_stop', 'cap_length_loop', 'malformed_invalid_or_trailing', 'other_incomplete')] +
            [diagnostics[suite][arm]['overall']['failed']] for suite in summary for arm in CONDITIONS]), '',
        'Categories are mutually exclusive, with cap-length repeated-action loops classified first. '
        'X7 reproduces 133 one-move stops, 47 capped loops, and 60 remaining failures (51 malformed/invalid/trailing '
        'and 9 other incomplete). Some capped loops include TRY/PRN as well as DONE/BT. '
        'The repeated-block heuristic is descriptive: legitimate repetitions before another error can also trigger it. '
        'A readable prefix of malformed output is diagnostic only and never changes the grade.', '',
        '## First wrong moves versus gym memory', '',
        'Indices are zero-based; fractions divide by canonical action count. A correct prefix that simply stops has '
        'a missing move, not an observed wrong action. Paired medians include only the same failed written cases '
        'with an observable wrong move in both lanes. The full per-case paired audit retains missing moves and '
        'cases with no observable action error. A fully correct canonical prefix followed by trailing actions or '
        'bad JSON closure remains a failure, but is identified separately as a completion-format failure; a trailing '
        'first error has fraction 1 and is not evidence that state was lost before completion.', '',
        table(['Suite', 'Arm', 'Wrong observed', 'Missing only', 'Unobservable', 'Completed-prefix format failures', 'Paired errors', 'Written index', 'Gym index',
            'Written fraction', 'Gym fraction'], [[suite, arm, d['observed_wrong'], d['missing_without_wrong'],
            d['no_observable_move_error'],
            d['completed_prefix_format_failures'],
            d['paired_observed_errors'], number(d['paired_written_index']), number(d['paired_gym_index']),
            number(d['paired_written_fraction']), number(d['paired_gym_fraction'])]
            for suite in summary for arm in CONDITIONS for d in [diagnostics[suite][arm]['overall']]]), '']
    for suite in summary:
        lines += ['### ' + suite.capitalize() + ': failed written runs by family', '', table(
            ['Family', 'Arm', 'Failures', 'Wrong observed', 'Missing only', 'Median wrong index', 'Median fraction', 'Index/length rank correlation'],
            [[family, arm, d['failed'], d['observed_wrong'], d['missing_without_wrong'], number(d['median_wrong_index']),
                number(d['median_wrong_fraction']), number(d['wrong_index_length_spearman'])]
                for family in FAMILIES for arm in CONDITIONS for d in [diagnostics[suite][arm]['by_family'][family]]]), '']
        lines += ['### ' + suite.capitalize() + ': failed written runs by canonical length', '', table(
            ['Length', 'Arm', 'Written complete', 'Gym complete', 'Failures', 'Wrong observed', 'Missing only', 'Median wrong index', 'Median fraction'],
            [[length, arm, frac([d['written_correct'], d['total']]), frac([d['gym_correct'], d['total']]),
                d['failed'], d['observed_wrong'], d['missing_without_wrong'], number(d['median_wrong_index']),
                number(d['median_wrong_fraction'])] for length in ('1-8', '9-16', '17-32', '33-64', '65+')
                for arm in CONDITIONS for d in [diagnostics[suite][arm]['by_length'][length]]]), '']
    lines += ['## Call budgets and costs', '', table(['Suite', 'Arm', 'Lane', 'Calls', 'Completion tokens',
        'Total tokens', 'Inference seconds', 'Capped calls', 'Reused cases'],
        [[suite, arm, mode, c['calls'], c['completion_tokens'], c['total_tokens'], round(c['inference_seconds']),
            c['length_limited_calls'], c['reused_cases']] for suite in summary for arm in CONDITIONS for mode in LANES
            for c in [costs[suite][arm][mode]]]), '',
        table(['Arm', 'Arm-process wall seconds', 'New training wall seconds', 'Fresh completion tokens'],
            [[arm, round(meta[arm]['completion']['seconds']), round(meta[arm].get('training', {}).get('training_seconds', 0)),
            sum(c['fresh_completion_tokens'] for suite in summary for c in costs[suite][arm].values())]
            for arm in CONDITIONS]), '',
        'Inference seconds sum the backend-reported amortized per-call times. Historical calls contribute to lane '
        'token/time totals but were not rerun; arm-process wall time includes model setup, HDFS persistence, and checkpoint parity checks, '
        'but starts after dependency/environment bootstrap. Full Ray submission wall times are retained in the execution manifest. '
        'Identical reused yes calls appear in both suite totals and are not independent replications. '
        'Raw provenance enables exact deduplication. No tool-failure recovery was tested.', '',
        '## Reproducibility and interpretation limits', '',
        f"Base `{protocol['model']}`, revision `{protocol['revision']}`; ordinary LoRA rank 16, alpha 32, "
        'dropout 0, q/k/v/o/gate/up/down targets; BF16 base and FP32 adapters. AdamW, learning rate 2e-4, '
        'weight decay .01, cosine schedule, 5% warmup, gradient clip 1; microbatch 8, accumulation 4, '
        'effective batch 32 with X7 partial final-batch behavior. Gradient checkpointing, response-only loss, '
        'no target truncation or packing. Initialization and native checkpoint reload parity checks passed.', '',
        'Greedy temperature 0, top-p 1, top-k -1, seed 17, thinking off, context 16,384. '
        'Direct cap 128, written cap 4,096, action cap 128, maximum 512 action calls. Prompts and grading '
        'are unchanged. Direct/correct-state batches have at most 32 calls; written/gym calls are serial. '
        'Native adapters use X7\'s job-scoped vLLM 0.11.2 B200 PDL workaround. The historical zero-shot reference '
        'has its LoRA engine disabled; this runtime difference remains a comparison limit.', '',
        'The cue-free suite preserves the 120 yes instances exactly and changes the negative grammars. '
        'Output and success changes on identical yes instances are listed per arm/lane in metadata.json. '
        'It removes the last-terminal/symbol-presence cue, not every possible shortcut; some count/length rules remain above chance. Families and alphabet permutations '
        'share templates; one seed, few full trajectories, training-token versus update differences, and fixed '
        'completion budgets limit claims. The optional 92-case bracket full-trace test was skipped because its '
        'action protocol differs and new training data are out of scope. Existing results were not modified.', '',
        '[Frozen protocol](wholerun_protocol.md), [machine-readable protocol](../experiments/wholerun/protocol.json), '
        '[family/level/label atlas](results/wholerun/summary.json), [paired diagnostics](results/wholerun/diagnostics.json), '
        '[raw calls and instances](results/wholerun/audit.json.gz), [training/runtime metadata](results/wholerun/metadata.json), '
        '[execution manifest](results/wholerun/execution_manifest.json), '
        'and [cost accounting](results/wholerun/costs.json) accompany this report.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('build/wholerun-20261005/results'))
    parser.add_argument('--output', type=Path, default=Path('docs/results/wholerun'))
    args = parser.parse_args()
    instances, runs, meta, protocol = assemble(args.root)
    summary, diagnostics, costs = analyze(instances, runs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary', summary), ('diagnostics', diagnostics), ('costs', costs), ('metadata', meta)]:
        (args.output / (name + '.json')).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    (args.output / 'audit.json.gz').write_bytes(gzip.compress(json.dumps(dict(instances=instances, runs=runs), sort_keys=True).encode(), mtime=0))
    (args.output / 'manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir() if p.name != 'manifest.json'}, indent=2) + '\n')
    Path('docs/wholerun_results.md').write_text(report(summary, diagnostics, costs, meta, protocol))
    print(json.dumps({suite: {arm: {m: v['overall']['correct'] for m, v in lanes.items()}
        for arm, lanes in arms.items()} for suite, arms in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
