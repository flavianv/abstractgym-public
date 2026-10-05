"""Auditable training-only demonstrations and explicit reasoning separation."""
from __future__ import annotations
import json
from abstractgym.controller import teacher_trajectory


def build_demonstrations(rows):
    training = [r for r in rows if r["split"] == "train"]
    if not training:
        raise ValueError("demonstrations require training rows")
    trajectories = [(r, teacher_trajectory(r)) for r in training]
    examples = []
    for op in ("TRY", "PRN", "BT", "DONE", "ACC"):
        candidates = [(len(json.dumps(s["observation"])), r["task_id"], i, r, s)
                      for r, steps in trajectories for i, s in enumerate(steps)
                      if s["action"]["op"] == op]
        _, _, i, row, step = min(candidates, key=lambda x: x[:3])
        examples.append({"task_id": row["task_id"], "structure_id": row["structure_id"],
                         "split": "train", "step": i, **step})
    row, steps = min(trajectories, key=lambda pair: (len(pair[1]), pair[0]["task_id"]))
    return {"selection": "shortest serialized training state per action; shortest training trace",
            "single_actions": examples,
            "full_trace": {"task_id": row["task_id"], "structure_id": row["structure_id"],
                           "split": "train", "observation": steps[0]["observation"],
                           "actions": [s["action"] for s in steps]}}


def demonstration_prefix(bundle):
    lines = ["Worked examples from other grammars. The stack is ordered root to current node; "
             "the last frame is the current node. Use that frame's phase and tried list."]
    for example in bundle["single_actions"]:
        lines += ["Single-action example observation:", json.dumps(example["observation"], sort_keys=True),
                  "Correct next-action JSON:", json.dumps(example["action"], sort_keys=True)]
    example = bundle["full_trace"]
    lines += ["Complete-trace example initial observation:", json.dumps(example["observation"], sort_keys=True),
              "Correct complete action array:", json.dumps(example["actions"], sort_keys=True),
              "End of examples. Now execute the new task below using its grammar and state."]
    return "\n".join(lines) + "\n\n"


def split_completion(token_ids, tokenizer, thinking):
    raw = tokenizer.decode(token_ids, skip_special_tokens=True)
    if not thinking:
        return {"text": raw, "raw_text": raw, "reasoning_tokens": 0,
                "answer_tokens": len(token_ids), "reasoning_closed": True}
    closing = tokenizer.convert_tokens_to_ids("</think>")
    if closing not in token_ids:
        return {"text": "", "raw_text": raw, "reasoning_content": raw,
                "reasoning_tokens": len(token_ids), "answer_tokens": 0, "reasoning_closed": False}
    boundary = token_ids.index(closing) + 1
    return {"text": tokenizer.decode(token_ids[boundary:], skip_special_tokens=True).strip(),
            "raw_text": raw, "reasoning_content": tokenizer.decode(token_ids[:boundary], skip_special_tokens=True),
            "reasoning_tokens": boundary, "answer_tokens": len(token_ids) - boundary,
            "reasoning_closed": True}
