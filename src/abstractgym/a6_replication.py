"""X3's independently frozen maps/templates; no model outputs used."""
import itertools
import random
from abstractgym.a6 import grammar_for
from abstractgym.oracle import accepts


def fresh_suite():
    rng=random.Random(20261004);rows=[]
    for alphabet in ('klmno','56789','{<}>@','PQRST'):
        a,b,c,d,center=alphabet;mapping={a:c,b:d}
        base=dict(alphabet=alphabet,pairs=mapping,center=center)
        grammar=grammar_for(base)
        for depth in (0,1,2,3,4,8,16):
            if depth<=3:patterns=list(itertools.product(mapping,repeat=depth))
            else:
                patterns=set()
                while len(patterns)<8:patterns.add(tuple(rng.choice((a,b)) for _ in range(depth)))
                patterns=sorted(patterns)
            for k,opens in enumerate(patterns):
                good=[mapping[t] for t in reversed(opens)];bad=good.copy()
                if depth:
                    pos=rng.randrange(depth);bad[pos]=d if bad[pos]==c else c
                else:bad=[rng.choice((c,d))]
                for label,closes in [(True,good),(False,bad)]:
                    tokens=list(opens)+[center]+closes
                    assert accepts(grammar,tuple(tokens))==label
                    rows.append(dict(base,id=f'fresh-{alphabet}-{depth}-{k}-{label}',split='test',depth=depth,
                        input=tokens,accept=label,grammar=grammar.to_dict(),negative=None if label else 'random_close_mutation'))
    return rows
