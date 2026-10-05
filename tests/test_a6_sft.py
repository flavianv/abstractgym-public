import json
from abstractgym.a6_sft import training_examples,tokenize_example
from abstractgym.brackets import STACK_PROTOCOL,expected


def test_sft_is_dev_only_and_retains_full_stack_prompt():
    rows=training_examples()
    assert len(rows)==2305 and len({r['prompt'] for r in rows})==148
    for r in rows:
        assert r['prompt'].startswith(STACK_PROTOCOL)
        obs=json.loads(r['prompt'][len(STACK_PROTOCOL):])
        assert obs['pairs']=={'u':'w','v':'x'} and obs['center']=='y'
        assert len(obs['stack'])<=4
        dummy=dict(obs,input=[] if obs['end'] else [obs['token']])
        assert json.loads(r['answer'])==expected(dummy,0,obs['mode'],obs['stack'])
        assert set(obs['stack'])<=set('wx')


def test_response_only_loss_and_no_thinking_prefix():
    class Tokenizer:
        eos_token_id=99
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs==dict(tokenize=True,add_generation_prompt=True,enable_thinking=False)
            assert messages==[{'role':'user','content':'unchanged prompt'}]
            return [1,2,3]
        def encode(self,text,add_special_tokens):
            assert text=='{"op":"POP"}' and not add_special_tokens
            return [4,5]
    result=tokenize_example(Tokenizer(),dict(prompt='unchanged prompt',answer='{"op":"POP"}'))
    assert result['input_ids']==[1,2,3,4,5,99]
    assert result['labels']==[-100,-100,-100,4,5,99]
    assert result['prompt_tokens']==3 and result['answer_tokens']==3
