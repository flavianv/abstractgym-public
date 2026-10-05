"""Training-only examples for the X7/X8 objectives."""
import json
from abstractgym.a6 import dev_data,grammar_for
from abstractgym.benchmark import build_suite
from abstractgym.brackets import direct_prompt
from abstractgym.controller import teacher_trajectory
from abstractgym.experiment import prompt_for


def dfs_examples():
    rows,_,_=build_suite();result=[]
    for row in rows:
        if row['split']!='train':continue
        for i,state in enumerate(teacher_trajectory(row)):
            action=state['action']
            result.append(dict(trajectory=row['task_id'],position=i,action=action,
                prompt=prompt_for(state['observation'],'oracle_step'),answer=json.dumps(action,separators=(',',':'))))
    return result


def membership_examples():
    rows,_,_=dev_data()
    return [dict(trajectory=r['id'],position=0,prompt=direct_prompt(dict(r,grammar=grammar_for(r).to_dict())),
                 action=dict(accept=r['accept']),answer=json.dumps(dict(accept=r['accept']),separators=(',',':'))) for r in rows]
