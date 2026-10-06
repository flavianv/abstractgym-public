from abstractgym.complexity import build_complexity_suite,build_cuefree_complexity_suite,FAMILIES
from abstractgym.controller import ControllerEnv,teacher_trajectory
from abstractgym.cuefree import shortcut_scores
from abstractgym.cfg import Grammar
from abstractgym.oracle import accepts


def test_cuefree_preserves_yes_and_changes_only_no_grammar():
    old=build_complexity_suite();new=build_cuefree_complexity_suite()
    assert len(new)==240 and len({r['task_id'] for r in new})==240
    assert new[::2]==old[::2]
    for a,b in zip(old[1::2],new[1::2]):
        assert a['input']==b['input'] and a['pair_id']==b['pair_id']
        assert a['task_id']!=b['task_id'] and a['grammar']!=b['grammar']
        assert a['search_config']==b['search_config']


def test_cuefree_shortcuts_match_prototype():
    scores=shortcut_scores(build_cuefree_complexity_suite())
    for name in list(scores)[:5]:assert scores[name]['overall']==[120,240]
    for name in list(scores)[5:]:
        assert scores[name]['by_family']['terminal_length']==[40,40]
        assert scores[name]['by_family']['chain_depth']==[40,40]
    count_names=list(scores)[5:]
    assert [scores[n]['overall'][0] for n in count_names]==[160,172]
    assert [[scores[n]['by_family'][f][0] for f in FAMILIES] for n in count_names]==[
        [40,40,24,20,20,16],[40,40,28,20,24,20]]
    assert shortcut_scores(build_complexity_suite())['last rule, last terminal in string']['overall']==[240,240]


def test_cuefree_oracle_and_canonical_trajectories():
    for row in build_cuefree_complexity_suite():
        assert accepts(Grammar.from_dict(row['grammar']),tuple(row['input']))==(row['target_answer']=='accept')
        env=ControllerEnv(row)
        for step in teacher_trajectory(row):
            assert step['observation']==env.observation()
            env.step(step['action'])
        assert env.outcome==row['target_answer']
        assert env.steps<row['search_config']['max_steps']
