"""Whole-run format ablation on X7's unchanged training trajectories."""
import json
import random

from abstractgym.a6_objectives import dfs_examples
from abstractgym.benchmark import build_suite
from abstractgym.controller import ControllerEnv, teacher_trajectory
from abstractgym.experiment import prompt_for

ARMS = ('fulltrace_sft', 'fulltrace_sft_matched', 'mixed_sft')


def fulltrace_examples():
    examples = []
    for row in build_suite()[0]:
        if row['split'] != 'train':
            continue
        actions = [s['action'] for s in teacher_trajectory(row)]
        examples.append(dict(trajectory=row['task_id'], kind='full_trace',
            prompt=prompt_for(ControllerEnv(row).observation(), 'full_trace'),
            answer=json.dumps(actions, separators=(',', ':'))))
    return examples


def training_examples(arm):
    if arm not in ARMS:
        raise ValueError('unknown whole-run arm')
    full = fulltrace_examples()
    if arm != 'mixed_sft':
        return full
    return [dict(e, kind='single_step') for e in dfs_examples()] + full


def training_schedule(data, arm, *, target_budget=295089, seed=17, batch_size=32):
    """Match supervised tokens at whole-update boundaries, including EOS."""
    if arm not in ARMS or not data or batch_size < 1 or target_budget < 1:
        raise ValueError('invalid schedule configuration')
    schedule = []
    tokens = 0
    epoch = 0
    while epoch < 3 or arm == 'fulltrace_sft_matched':
        order = list(range(len(data)))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            group = order[start:start + batch_size]
            schedule.append(dict(epoch=epoch + 1, indices=group))
            tokens += sum(data[i]['answer_tokens'] for i in group)
            if arm == 'fulltrace_sft_matched' and tokens >= target_budget:
                return schedule
        epoch += 1
    return schedule


def diagnostic_actions(text):
    """Read complete JSON values from the array prefix; never repair grades."""
    text = text.lstrip()
    decoder = json.JSONDecoder()
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        parsed = None
    if isinstance(parsed, list):
        return parsed, True
    if not text.startswith('['):
        return ([parsed] if isinstance(parsed, dict) else []), False
    actions, pos = [], 1
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            return actions, False
        if text[pos] == ']':
            return actions, not text[pos + 1:].strip()
        try:
            value, end = decoder.raw_decode(text, pos)
        except (ValueError, RecursionError):
            return actions, False
        if not isinstance(value, dict):
            return actions + [value], False
        actions.append(value)
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos] == ',':
            pos += 1
            if text[pos:].lstrip().startswith(']'):
                return actions, False
        elif pos < len(text) and text[pos] == ']':
            return actions, not text[pos + 1:].strip()
        else:
            return actions, False


def written_diagnostic(row, record):
    gold = [s['action'] for s in teacher_trajectory(row)]
    call = record['calls'][0]
    actions, valid_array = diagnostic_actions(call.get('text', ''))
    prefix = 0
    for action, expected in zip(actions, gold):
        if action != expected:
            break
        prefix += 1
    # No wrong move is observable when a correct prefix simply stops.
    wrong = prefix if prefix < min(len(actions), len(gold)) or len(actions) > len(gold) else None
    loop = any(actions[i:i + width] == actions[i + width:i + 2 * width] == actions[i + 2 * width:i + 3 * width]
        for width in range(1, 9) for i in range(max(0, len(actions) - 3 * width + 1)))
    if record['success']:
        category = None
    elif call.get('finish_reason') == 'length' and loop:
        category = 'cap_length_loop'
    elif len(actions) == 1 and call.get('finish_reason') != 'length':
        category = 'one_move_then_stop'
    elif not valid_array or record['failure'] == 'invalid_or_trailing_action':
        category = 'malformed_invalid_or_trailing'
    else:
        category = 'other_incomplete'
    return dict(task_id=row['task_id'], category=category, valid_array=valid_array,
        loop_detected=loop, loop_ops=sorted({a.get('op') if isinstance(a, dict) and isinstance(a.get('op'), str)
            else '<invalid-op>' for a in actions}),
        emitted_complete_actions=len(actions), correct_prefix_actions=prefix,
        canonical_actions=len(gold), first_wrong_move=wrong,
        first_wrong_is_trailing=wrong == len(gold),
        canonical_prefix_completed=prefix == len(gold),
        first_wrong_fraction=wrong / len(gold) if wrong is not None else None,
        first_missing_move=prefix if not record['success'] and wrong is None and prefix < len(gold)
            and (prefix > 0 or valid_array) else None,
        diagnostic_only=True)
