"""Paired controller experiments with auditable prompts and failure accounting."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from urllib.request import Request, urlopen

from abstractgym.controller import ControllerEnv, PROTOCOL, teacher_trajectory

MODES = ("full_trace", "oracle_step", "external_state")


def prompt_for(observation, mode):
    instruction = ("Return one JSON array containing all actions through completion."
                   if mode == "full_trace" else "Return exactly one next-action JSON object.")
    instruction += " Output raw JSON only: no Markdown fences, headings, explanations, or reasoning text."
    return PROTOCOL + "\n" + instruction + "\nObservation:\n" + json.dumps(observation, sort_keys=True)


class ChatCompletionsAdapter:
    """Explicit endpoint/model selection; no credentials enter run artifacts."""
    def __init__(self, *, base_url, model, api_key_env="OPENAI_API_KEY", max_tokens=8192, timeout=120):
        if max_tokens < 1 or timeout <= 0:
            raise ValueError("token budget and timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.temperature = 0

    def __call__(self, prompt):
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": self.max_tokens}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "")
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        choice = body["choices"][0]
        return {"text": choice["message"]["content"], "usage": body.get("usage") or {},
                "finish_reason": choice.get("finish_reason")}


def run_case(row, mode, adapter=None, *, reference=False, max_calls=4096):
    if mode not in MODES or (not reference and adapter is None):
        raise ValueError("select a supported mode and a model adapter or explicit reference control")
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    env = ControllerEnv(row)
    gold = teacher_trajectory(row)
    calls = []
    failure = None
    first_error = None
    correct_actions = 0
    started = time.perf_counter()
    batch_responses = None

    def call(observation, reference_output):
        prompt = prompt_for(observation, mode)
        begin = time.perf_counter()
        entry = {"prompt": prompt}
        # The adapter gets only the prompt. reference_output never crosses its
        # interface and is only used in explicitly labelled plumbing tests.
        try:
            response = ({"text": json.dumps(reference_output), "usage": {}} if reference
                        else next(batch_responses) if batch_responses is not None else adapter(prompt))
            entry.update(response)
            parsed = json.loads(response["text"])
        except Exception as exc:
            # Do not log exception strings from network clients (may include
            # credentials/headers). Keep the type for infrastructure diagnosis.
            entry["error_type"] = type(exc).__name__
            entry["status_code"] = getattr(exc, "code", None)
            raise ValueError("call_or_parse_error") from None
        finally:
            entry["seconds"] = time.perf_counter() - begin
            calls.append(entry)
        return parsed

    if mode == "full_trace":
        try:
            actions = call(env.observation(), [step["action"] for step in gold])
            if not isinstance(actions, list):
                raise ValueError("expected_action_array")
            for index, action in enumerate(actions):
                try:
                    env.step(action)
                    correct_actions += 1
                except ValueError:
                    first_error = index
                    raise ValueError("invalid_or_trailing_action") from None
            if env.outcome is None:
                failure = "incomplete_trace"
                first_error = len(actions)
        except ValueError as exc:
            failure = str(exc)
    elif mode == "oracle_step":
        if not reference and hasattr(adapter, "generate_batch"):
            # Independent teacher states may share a GPU batch, never a conversation.
            # Only prompts cross the adapter boundary, never reference actions.
            prompts = [prompt_for(step["observation"], mode) for step in gold[:max_calls]]
            responses = adapter.generate_batch(prompts)
            if len(responses) != len(prompts):
                raise ValueError("batch response count mismatch")
            batch_responses = iter(responses)
        for index, step in enumerate(gold):
            if len(calls) >= max_calls:
                failure = "call_budget_exhausted"
                break
            try:
                action = call(step["observation"], step["action"])
                correct = action == step["action"]
            except ValueError:
                correct = False
                failure = failure or "call_or_parse_error"
            if correct:
                correct_actions += 1
            else:
                first_error = index if first_error is None else first_error
                failure = failure or "invalid_action"
        # This is teacher-state accuracy, deliberately not a deployed rollout.
    else:
        while env.outcome is None:
            if len(calls) >= max_calls:
                failure = "call_budget_exhausted"
                break
            try:
                action = call(env.observation(), env.reference_action() if reference else None)
                env.step(action)
                correct_actions += 1
            except ValueError as exc:
                failure = str(exc)
                first_error = env.steps
                break

    if mode == "oracle_step":
        success = failure is None and correct_actions == len(gold)
        outcome = None
    else:
        outcome = env.outcome
        success = failure is None and outcome == row["target_answer"]
        if not success and failure is None:
            failure = "budget_exhausted" if outcome == "unknown" else "wrong_verdict"
    return {"task_id": row["task_id"], "pair_id": row["pair_id"], "split": row["split"],
            "family": row["family"], "depth_band": row["depth_band"], "structure_id": row["structure_id"],
            "difficulty": row["difficulty"], "mode": mode,
            "success": success, "outcome": outcome, "target_answer": row["target_answer"],
            "failure": failure, "first_error": first_error,
            "correct_actions": correct_actions, "target_actions": len(gold),
            "action_accuracy": correct_actions / len(gold), "model_calls": 0 if reference else len(calls),
            "controller_calls": len(calls), "seconds": time.perf_counter() - started, "calls": calls}


def summarize(records):
    summaries = {}
    for mode in MODES:
        selected = [r for r in records if r["mode"] == mode]
        if not selected:
            continue
        usage_available = all("total_tokens" in call.get("usage", {}) for r in selected for call in r["calls"])
        summaries[mode] = {"cases": len(selected),
            "all_case_success_rate": sum(r["success"] for r in selected) / len(selected),
            "success_definition": "all teacher-state actions correct" if mode == "oracle_step" else "valid completed rollout and correct verdict",
            "micro_action_accuracy": sum(r["correct_actions"] for r in selected) / sum(r["target_actions"] for r in selected),
            "failures": dict(Counter(r["failure"] for r in selected if r["failure"])),
            "model_calls": sum(r["model_calls"] for r in selected),
            "mean_controller_calls": sum(r["controller_calls"] for r in selected) / len(selected),
            "usage": {key: (sum(call.get("usage", {}).get(key, 0) or 0 for r in selected for call in r["calls"]) if usage_available else None)
                      for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "usage_available": usage_available,
            "seconds": sum(r["seconds"] for r in selected)}
    return summaries


def run_suite(instances, output_dir, *, adapter=None, reference=False, split="test", limit=None,
              modes=MODES, max_calls=4096, on_record=None):
    source = Path(instances).read_bytes()
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    rows = [r for r in rows if r["split"] == split]
    if limit is not None:
        if limit < 2 or limit % 2:
            raise ValueError("limit must be a positive even number to preserve contrastive pairs")
        # Round-robin whole pairs over family/depth cells, so a small pilot
        # does not silently consist only of the first family in the file.
        groups = defaultdict(list)
        pairs = defaultdict(list)
        for row in rows:
            pairs[row["pair_id"]].append(row)
        for pair in pairs.values():
            groups[(pair[0]["family"], pair[0]["depth_band"])].append(pair)
        selected = []
        while len(selected) < limit and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(selected) < limit:
                    selected.extend(groups[key].pop(0))
        rows = selected
    if not rows:
        raise ValueError("no evaluation rows selected")
    if any(count != 2 for count in Counter(row["pair_id"] for row in rows).values()):
        raise ValueError("evaluation selection must contain both cases of each contrastive pair")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing expensive run logs.
    records = []
    with (output / "records.jsonl").open("x") as handle:
        for row in rows:
            for mode in modes:
                record = run_case(row, mode, adapter, reference=reference, max_calls=max_calls)
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                if on_record is not None:
                    on_record(record, output)
    metadata = {"run_kind": "reference_control_NOT_model_results" if reference else "model_evaluation",
        "model": "handwritten_DFS" if reference else getattr(adapter, "model", type(adapter).__name__),
        "temperature": getattr(adapter, "temperature", None), "max_completion_tokens_per_call": getattr(adapter, "max_tokens", None),
        "instances_sha256": hashlib.sha256(source).hexdigest(), "split": split,
        "selected_task_ids": [r["task_id"] for r in rows], "max_calls_per_case": max_calls,
        "python": platform.python_version(), "platform": platform.platform(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("experiment.py", "controller.py", "benchmark.py", "cfg.py", "trace.py", "oracle.py")},
        "protocol": "strict canonical DFS; no correction, retries, or repair; malformed/unfinished cases count as failures",
        "comparison_limits": "same tasks/actions; full_trace gets initial state once; oracle_step gets every teacher state; external_state gets exact live state after each valid action. Context and call budgets differ; compare quality and total tokens, not accuracy alone.",
        "metrics": summarize(records),
        "by_family": {family: summarize([r for r in records if r["family"] == family]) for family in sorted({r["family"] for r in records})},
        "by_depth_band": {band: summarize([r for r in records if r["depth_band"] == band]) for band in sorted({r["depth_band"] for r in records})},
        "nested_balanced_counts": summarize([r for r in records if r["difficulty"].get("balanced_pair_counts")]),
        "by_depth": {str(depth): summarize([r for r in records if r["difficulty"]["derivation_depth"] == depth]) for depth in sorted({r["difficulty"]["derivation_depth"] for r in records})}}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata
