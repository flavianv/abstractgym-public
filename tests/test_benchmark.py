import json
from collections import Counter, defaultdict

import pytest

from abstractgym.benchmark import build_suite, write_suite
from abstractgym.cfg import Grammar
from abstractgym.controller import ControllerEnv, teacher_trajectory
from abstractgym.dataset import generate_grammar, make_instance
from abstractgym.experiment import MODES, ChatCompletionsAdapter, run_case, run_suite
from abstractgym.oracle import accepts
from abstractgym.schema import InstanceComplexityConfig, SearchConfig
from abstractgym.trace import generate_trace


@pytest.fixture(scope="module")
def suite():
    return build_suite(structures_per_split=1, pairs_per_structure=2)


def test_structural_splits_and_contrastive_pairs(suite):
    rows, manifest, quarantine = suite
    assert len(rows) == manifest["requested_rows"] == 32
    assert not quarantine
    assert build_suite(structures_per_split=1, pairs_per_structure=2)[0] == rows
    split_structures = defaultdict(set)
    pairs = defaultdict(list)
    for row in rows:
        split_structures[row["split"]].add(row["structure_id"])
        pairs[row["pair_id"]].append(row)
        assert accepts(Grammar.from_dict(row["grammar"]), tuple(row["input"])) == (row["target_answer"] == "accept")
    assert not (split_structures["train"] & split_structures["test"])
    assert not (split_structures["validation"] & split_structures["test"])
    assert not (split_structures["train"] & split_structures["validation"])
    for structure in split_structures["test"]:
        assert {r["depth_band"] for r in rows if r["structure_id"] == structure} == {"shallow", "deep"}
    for pair in pairs.values():
        assert len(pair) == 2
        assert pair[0]["input"] == pair[1]["input"]
        assert pair[0]["grammar_id"] != pair[1]["grammar_id"]
        assert {r["target_answer"] for r in pair} == {"accept", "reject"}
        if pair[0]["family"] == "nested":
            # Both grammars have the same base word. Verdict requires checking
            # the changed opening/closing correspondence, not the center token.
            bases = []
            for row in pair:
                grammar = Grammar.from_dict(row["grammar"])
                bases.append([p.rhs for p in grammar.productions_for("S") if all(s in grammar.terminals for s in p.rhs)])
            assert bases[0] == bases[1]


def test_legacy_identity_is_content_not_seed():
    a, b = generate_grammar(seed=1), generate_grammar(seed=2)
    assert a.grammar == b.grammar
    assert a.grammar_id == b.grammar_id


def test_bounded_rejection_is_quarantined_even_if_oracle_rejects():
    instance, record = make_instance(generate_grammar(),
        InstanceComplexityConfig(derivation_depth=3, validity="rejected", invalidity_type="wrong_suffix"),
        search_config=SearchConfig(max_steps=1))
    assert instance is None
    assert record.reason == "search_budget_exhausted"
    rows, manifest, quarantine = build_suite(structures_per_split=1, pairs_per_structure=1, max_steps=1)
    assert not rows and quarantine
    assert manifest["retained_rows"] == 0
    assert manifest["quarantined_rows"] == manifest["requested_rows"]


def test_step_machine_matches_recursive_reference(suite):
    mapping = {"try_production": "TRY", "backtrack": "BT", "prune": "PRN", "reject_node": "DONE", "accept": "ACC", "reject_leaf": "REJ"}
    for row in suite[0]:
        steps = teacher_trajectory(row)
        recursive = generate_trace(Grammar.from_dict(row["grammar"]), tuple(row["input"]), **row["search_config"])
        assert [s["action"]["op"] for s in steps] == [mapping[s["op"]] for s in recursive.trace]
        assert [s["action"].get("production") for s in steps if s["action"]["op"] == "TRY"] == [s["production_id"] for s in recursive.trace if s["op"] == "try_production"]
        assert all("target_answer" not in step["observation"] for step in steps)
        assert all("target_trace_compact" not in step["observation"] for step in steps)


def test_environment_does_not_correct_invalid_action(suite):
    env = ControllerEnv(suite[0][0])
    before = env.observation()
    with pytest.raises(ValueError, match="invalid_action"):
        env.step({"op": "ACC"})
    assert env.observation() == before
    changed = env.observation()
    changed["stack"].clear()
    assert env.stack  # observations cannot mutate stored state


def test_environment_budget_is_unknown(suite):
    row = {**suite[0][0], "search_config": {"max_steps": 1, "max_depth": 64}}
    env = ControllerEnv(row)
    env.step(env.reference_action())
    assert env.outcome == "unknown"


@pytest.mark.parametrize("mode", MODES)
def test_reference_and_failed_model_have_separate_results(suite, mode):
    row = suite[0][0]
    result = run_case(row, mode, reference=True)
    assert result["success"] and result["model_calls"] == 0
    prompts = []
    def broken(prompt):
        prompts.append(prompt)
        return {"text": "not JSON", "usage": {}}
    failure = run_case(row, mode, broken, max_calls=2)
    assert not failure["success"] and failure["action_accuracy"] == 0
    for prompt in prompts:
        assert "target_answer" not in prompt and "target_trace" not in prompt
    if mode == "oracle_step":
        assert failure["controller_calls"] == 2
    else:
        assert failure["controller_calls"] == 1


def test_full_trace_trailing_action_fails(suite):
    row = suite[0][0]
    actions = [step["action"] for step in teacher_trajectory(row)]
    result = run_case(row, "full_trace", lambda _: {"text": json.dumps(actions + [{"op": "ACC"}])})
    assert not result["success"]
    assert result["failure"] == "invalid_or_trailing_action"


def test_batched_teacher_queries_preserve_scoring_and_budget(suite):
    row = suite[0][0]
    gold = teacher_trajectory(row)
    replies = [{"text": json.dumps(gold[0]["action"])}, {"text": "bad JSON"}]
    class Batched:
        def generate_batch(self, prompts):
            assert len(prompts) == 2
            assert all("target_answer" not in p and "target_trace" not in p for p in prompts)
            return replies
        def __call__(self, prompt):
            raise AssertionError("teacher queries must use the batch")
    result = run_case(row, "oracle_step", Batched(), max_calls=2)
    serial = iter(replies)
    baseline = run_case(row, "oracle_step", lambda p: next(serial), max_calls=2)
    for key in ("correct_actions", "first_error", "failure", "success", "model_calls"):
        assert result[key] == baseline[key]


def test_demonstrations_use_only_training_and_replay(suite):
    from abstractgym.prompting import build_demonstrations, demonstration_prefix
    rows = suite[0]
    bundle = build_demonstrations(rows)
    assert build_demonstrations([r for r in rows if r["split"] == "train"]) == bundle
    lookup = {r["task_id"]: r for r in rows}
    for example in bundle["single_actions"]:
        row = lookup[example["task_id"]]
        assert row["split"] == "train"
        step = teacher_trajectory(row)[example["step"]]
        assert step["observation"] == example["observation"]
        assert step["action"] == example["action"]
    trace = bundle["full_trace"]
    env = ControllerEnv(lookup[trace["task_id"]])
    for action in trace["actions"]:
        env.step(action)
    assert env.outcome == lookup[trace["task_id"]]["target_answer"]
    assert "Correct complete action array" in demonstration_prefix(bundle)


def test_reasoning_boundary_does_not_repair_answer_json():
    from abstractgym.prompting import split_completion
    class Tokenizer:
        def convert_tokens_to_ids(self, token):
            assert token == "</think>"
            return 2
        def decode(self, ids, skip_special_tokens=True):
            return ''.join({1: 'reason', 2: '</think>', 3: '```json\n{}\n```', 4: '{}'}[i] for i in ids)
    tokenizer = Tokenizer()
    incomplete = split_completion([1], tokenizer, True)
    assert incomplete['text'] == '' and not incomplete['reasoning_closed']
    fenced = split_completion([1, 2, 3], tokenizer, True)
    assert fenced['text'].startswith('```')
    assert fenced['reasoning_tokens'] == 2 and fenced['answer_tokens'] == 1
    assert split_completion([1, 2, 4], tokenizer, True)['text'] == '{}'
    assert split_completion([4], tokenizer, False)['reasoning_tokens'] == 0


def test_reference_run_audit_and_no_overwrite(tmp_path, suite):
    write_suite(tmp_path / "data", structures_per_split=1, pairs_per_structure=1)
    snapshots = []
    def persist_callback(record, directory):
        saved = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
        assert saved[-1]["task_id"] == record["task_id"]
        assert saved[-1]["mode"] == record["mode"]
        snapshots.append(len(saved))
    result = run_suite(tmp_path / "data/instances.jsonl", tmp_path / "run", reference=True, on_record=persist_callback)
    assert snapshots == list(range(1, len(snapshots) + 1))
    assert result["run_kind"] == "reference_control_NOT_model_results"
    assert {"regular", "nested"} == set(result["by_family"])
    assert all(m["all_case_success_rate"] == 1 for m in result["metrics"].values())
    assert all(m["model_calls"] == 0 for m in result["metrics"].values())
    with pytest.raises(FileExistsError):
        run_suite(tmp_path / "data/instances.jsonl", tmp_path / "run", reference=True)
    with pytest.raises(ValueError, match="even"):
        run_suite(tmp_path / "data/instances.jsonl", tmp_path / "other", reference=True, limit=1)


def test_api_adapter_sends_only_prompt_and_records_usage(monkeypatch):
    import io
    import abstractgym.experiment as experiment
    def fake_urlopen(request, timeout):
        payload = json.loads(request.data)
        assert payload == {"model": "test-model", "messages": [{"role": "user", "content": "task"}], "temperature": 0, "max_tokens": 20}
        assert request.full_url == "http://localhost:8000/v1/chat/completions"
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": '{"op":"BT"}'}, "finish_reason": "stop"}], "usage": {"total_tokens": 30}}).encode())
    monkeypatch.setattr(experiment, "urlopen", fake_urlopen)
    adapter = ChatCompletionsAdapter(base_url="http://localhost:8000/v1", model="test-model", max_tokens=20)
    assert adapter("task")["usage"]["total_tokens"] == 30


def test_lucky_reject_with_unfinished_trace_is_not_valid(suite):
    from abstractgym.evaluate import score_prediction
    row = next(r for r in suite[0] if r["target_answer"] == "reject")
    row = {**row, "search_config": {"max_steps": 1, "max_depth": 64}}
    trace = generate_trace(Grammar.from_dict(row["grammar"]), tuple(row["input"]), **row["search_config"])
    row["target_trace_compact"] = trace.target_trace_compact
    score = score_prediction(row, {"output": f"<cot>{trace.target_trace_compact}</cot><answer>reject</answer>"})
    assert score.verdict_correct
    assert not score.trace_valid and not score.verdict_trace_consistent
    assert score.trace_failure == "budget_exhausted"


def test_model_adapter_path_uses_live_observations(suite):
    row = suite[0][0]
    def scripted_adapter(prompt):
        obs = json.loads(prompt.split("Observation:\n", 1)[1])
        env = ControllerEnv({"grammar": obs["grammar"], "input": obs["input"],
                             "search_config": {"max_steps": obs["max_steps"], "max_depth": obs["max_depth"]}})
        env.stack, env.steps, env.cut = obs["stack"], obs["steps"], obs["cut_seen"]
        return {"text": json.dumps(env.reference_action()), "usage": {"total_tokens": 1}}
    result = run_case(row, "external_state", scripted_adapter)
    assert result["success"]
    assert result["model_calls"] == result["target_actions"]
