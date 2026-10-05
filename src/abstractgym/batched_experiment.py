"""Batch independent calls; replay every response through the strict legacy grader."""
import json
from abstractgym.controller import ControllerEnv, teacher_trajectory
from abstractgym.experiment import prompt_for, run_case

class Replay:
    def __init__(self, calls): self.calls = iter(calls); self.used = 0
    def __call__(self, prompt):
        expected, response = next(self.calls)
        assert prompt == expected, 'replay prompt mismatch'
        self.used += 1
        return response

def evaluate_batched(rows, adapter, on_record=lambda r: None):
    records = []
    def finish(row, mode, calls):
        replay = Replay(calls)
        record = run_case(row, mode, replay, max_calls=512)
        assert replay.used == len(calls)
        # run_case seconds here measure grading, not inference. Sum amortized batch time explicitly.
        record['grading_seconds'] = record['seconds']
        record['seconds'] = sum(o.get('inference_seconds', 0) for _,o in calls)
        records.append(record); on_record(record)
    prompts = [prompt_for(ControllerEnv(r).observation(), 'full_trace') for r in rows]
    replies = adapter.generate_batch(prompts); assert len(replies) == len(rows)
    for row,p,o in zip(rows,prompts,replies): finish(row,'full_trace',[(p,o)])
    # Preserve each state occurrence, even if text-identical to a different case.
    prompts = [prompt_for(s['observation'],'oracle_step') for r in rows for s in teacher_trajectory(r)]
    replies = adapter.generate_batch(prompts); assert len(replies) == len(prompts)
    offset = 0
    for row in rows:
        n = len(teacher_trajectory(row)); finish(row,'oracle_step',list(zip(prompts[offset:offset+n],replies[offset:offset+n]))); offset += n
    active = [(r,ControllerEnv(r),[]) for r in rows]
    while active:
        prompts = [prompt_for(env.observation(),'external_state') for _,env,_ in active]
        replies = adapter.generate_batch(prompts); assert len(replies) == len(active)
        keep = []
        for (row,env,calls),p,o in zip(active,prompts,replies):
            calls.append((p,o))
            try: env.step(json.loads(o['text'])); failed = False
            except (ValueError,TypeError,KeyError): failed = True
            if failed or env.outcome is not None or len(calls)>=512: finish(row,'external_state',calls)
            else: keep.append((row,env,calls))
        active = keep
    return records
