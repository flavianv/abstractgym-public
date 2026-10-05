from collections import defaultdict
from abstractgym.a6_replication import fresh_suite
from abstractgym.a6 import digest
from abstractgym.brackets import suite


def test_fresh_cases_are_unique_balanced_and_symbol_disjoint():
    rows=fresh_suite();assert len(rows)==312
    assert len({r['id'] for r in rows})==312
    assert len({(r['alphabet'],tuple(r['input'])) for r in rows})==312
    old=set(''.join(r['alphabet'] for r in suite()));groups=defaultdict(list)
    for r in rows:
        assert not set(r['alphabet'])&old
        groups[r['alphabet'],r['depth']].append(r['accept'])
    assert len(groups)==28
    assert all(sum(v)*2==len(v) for v in groups.values())
    assert digest(rows)==digest(fresh_suite())
