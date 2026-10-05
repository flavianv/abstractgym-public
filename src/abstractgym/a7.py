"""Frozen, training-only evidence ablation; no evidence enters test prompts."""
from copy import deepcopy
import hashlib
import json
from abstractgym.benchmark import build_suite
from abstractgym.controller import ControllerEnv, teacher_trajectory
from abstractgym.experiment import prompt_for
from abstractgym.complexity import build_complexity_suite
from abstractgym.cfg import Grammar, Production
from abstractgym.oracle import accepts
from abstractgym.trace import _apply_production, _leftmost_nonterminal

MODEL = 'Qwen/Qwen3-14B'
REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'
VARIANTS = ('action', 'successor', 'branches')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def compact(value):
    return json.dumps(value, separators=(',', ':'))

def restore(obs):
    env = ControllerEnv({'grammar': obs['grammar'], 'input': obs['input'],
        'search_config': {k: obs[k] for k in ('max_steps', 'max_depth')}})
    env.stack = deepcopy(obs['stack']); env.steps = obs['steps']
    env.outcome = obs['outcome']; env.cut = obs['cut_seen']
    return env

def consequences(obs):
    """Grammar-legal counterfactual expansions, not canonical-policy actions."""
    env = restore(obs); action = env.reference_action(); results = []
    if action['op'] != 'TRY':
        env.step(action)
        return {'branches': [], 'successor': env.observation()}
    form = tuple(env.stack[-1]['form']); index = _leftmost_nonterminal(env.grammar, form)
    rules = [r for r in env.grammar.productions_for(form[index])
             if env.grammar.production_id(r) not in env.stack[-1]['tried']][:2]
    for rule in rules:
        branch = deepcopy(env); pid = branch.grammar.production_id(rule)
        child = list(_apply_production(form, index, rule)); parent_depth = len(branch.stack)
        branch.stack[-1]['tried'].append(pid)
        branch.stack.append({'form': child, 'tried': [], 'phase': 'entered'})
        branch.steps += 1
        if branch.steps >= branch.max_steps: branch.outcome = 'unknown'
        continuation = []
        for _ in range(3):
            if branch.outcome is not None or len(branch.stack) <= parent_depth: break
            nxt = branch.reference_action(); continuation.append(nxt); branch.step(nxt)
        status = (branch.outcome if branch.outcome else
                  'failed' if len(branch.stack) <= parent_depth else 'unresolved')
        results.append({'production': pid, 'child_form': child, 'continuation': continuation,
                        'status': status, 'state': branch.observation()})
    return {'branches': results}

AUX = {
 'successor': 'Training auxiliary task: return a JSON object with action (the canonical next action) and successor (the complete observation after exactly that action).',
 'branches': 'Training auxiliary task: return a JSON object with action and evidence. For TRY, evidence has branches: the first at most two untried applicable productions in order. For each, hypothetically expand it (even if not canonical first), then execute at most three canonical actions within that child, stopping on child failure or global completion. Each branch has production, child_form, continuation, status (accept/reject/unknown/failed/unresolved), state (complete resulting observation). Otherwise evidence has branches=[] and successor (complete observation after action).'
}

def training_pool():
    rows, _, _ = build_suite(); pool = []
    for row in rows:
        if row['split'] != 'train': continue
        trajectory = teacher_trajectory(row)
        chosen = {next(i for i,s in enumerate(trajectory) if s['action']['op'] == op)
                  for op in sorted({s['action']['op'] for s in trajectory})}
        candidates = sorted(range(len(trajectory)), key=lambda i: digest([row['task_id'], i, 17]))
        for i in candidates:
            if len(chosen) >= 8: break
            chosen.add(i)
        assert len(chosen) == 8
        for i in sorted(chosen):
            state = trajectory[i]; env = restore(state['observation'])
            successor = env.step(state['action'])
            pool.append({'trajectory': row['task_id'], 'position': i, **state,
                         'successor': successor, 'evidence': consequences(state['observation'])})
    assert len(pool) == 512
    return pool

def examples(pool, variant):
    assert variant in VARIANTS
    rows = []
    for s in pool:
        prompt = prompt_for(s['observation'], 'oracle_step')
        common = {'trajectory': s['trajectory'], 'position': s['position']}
        rows.append(dict(common, kind='action', prompt=prompt, answer=compact(s['action'])))
        if variant == 'action':
            rows.append(dict(common, kind='action_repeat', prompt=prompt, answer=compact(s['action'])))
        else:
            target = {'action': s['action'],
                      ('successor' if variant == 'successor' else 'evidence'): s[variant if variant == 'successor' else 'evidence']}
            # Replace only the output instruction, retaining the same protocol and observation.
            from abstractgym.controller import PROTOCOL
            aux = PROTOCOL + '\n' + AUX[variant] + '\nOutput only raw JSON.\nObservation:\n' + json.dumps(s['observation'], sort_keys=True)
            rows.append(dict(common, kind=variant, prompt=aux, answer=compact(target)))
    return rows

def shifted_suite():
    rows = build_complexity_suite(levels=(3, 6, 12, 24))
    for row in rows:
        old = Grammar.from_dict(row['grammar'])
        rename = dict(zip('abcd','wxyz'))
        rename.update({n: 'Root' if n == old.start else 'Node'+str(i)
                       for i,n in enumerate(sorted(old.nonterminals))})
        new = Grammar(rename[old.start], frozenset(rename[x] for x in old.nonterminals),
            frozenset(rename[x] for x in old.terminals),
            tuple(Production(rename[r.lhs], tuple(rename[x] for x in r.rhs)) for r in old.productions))
        row['grammar'] = new.to_dict(); row['input'] = [rename[x] for x in row['input']]
        row['task_id'] = 'a7_'+digest(row)[:20]; row['pair_id'] = 'a7_'+row['pair_id']
        assert accepts(new,tuple(row['input'])) == (row['target_answer']=='accept')
        teacher_trajectory(row)
    return rows

def forward_kl(student_logits, teacher_logits):
    """Full vocabulary KL(teacher || student), detached teacher; per-token mean."""
    import torch
    logq = torch.log_softmax(teacher_logits.detach().float(), dim=-1)
    logp = torch.log_softmax(student_logits.float(), dim=-1)
    return (logq.exp() * (logq-logp)).sum(-1).mean()
