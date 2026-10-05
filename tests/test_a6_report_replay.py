"""Shared replay helpers; CPU only, no inference or model dependencies."""
from abstractgym.experiment import run_case


def replay_complexity(rows,records):
    lookup={r['task_id']:r for r in rows}
    for r in records:
        class Adapter:
            def __init__(self):self.index=0
            def __call__(self,prompt):
                c=r['calls'][self.index];self.index+=1
                assert c['model_prompt']==prompt
                return {k:v for k,v in c.items() if k not in ('prompt','seconds')}
            def generate_batch(self,prompts):return [self(p) for p in prompts]
        adapter=Adapter();replayed=run_case(lookup[r['task_id']],r['mode'],adapter,max_calls=512)
        assert adapter.index==len(r['calls'])
        for key in ('success','failure','first_error','correct_actions','target_actions','outcome','controller_calls','model_calls'):
            assert replayed[key]==r[key],(r['task_id'],r['mode'],key)
