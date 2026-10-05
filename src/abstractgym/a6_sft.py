"""Development-only text examples for A6 E2; preserves the frozen stack prompt."""
import json
from abstractgym.a6 import dev_data, digest
from abstractgym.brackets import expected, advance, stack_prompt


def training_examples():
    rows, _, _ = dev_data()
    examples = []
    for row in rows:
        pos, mode, stack = 0, 'opening', []
        while True:
            action = expected(row, pos, mode, stack)
            examples.append(dict(trajectory=row['id'], position=pos,
                prompt=stack_prompt(row,pos,mode,stack),
                answer=json.dumps(action,separators=(',',':')),
                action=action, stack_depth=len(stack)))
            if action['op'] in ('ACCEPT','REJECT'):break
            pos,mode,stack=advance(action,pos,mode,stack)
    assert len(examples)==2305 and max(r['stack_depth'] for r in examples)<=4
    return examples


def tokenize_example(tokenizer, example):
    prefix=tokenizer.apply_chat_template([{'role':'user','content':example['prompt']}],
        tokenize=True,add_generation_prompt=True,enable_thinking=False)
    answer=tokenizer.encode(example['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
    return dict(input_ids=prefix+answer,labels=[-100]*len(prefix)+answer,
                prompt_tokens=len(prefix),answer_tokens=len(answer))
