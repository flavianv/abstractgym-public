"""Shortcut diagnostics from the supplied cue-free prototype; never model inputs."""
def shortcut_predictions(row):
    rules=row['grammar']['productions'];nts=set(row['grammar']['nonterminals'])
    terms=[s for r in rules for s in r['rhs'] if s not in nts];tokens=row['input']
    counts=[sum(s not in nts for s in r['rhs']) for r in rules]
    last=[s for s in rules[-1]['rhs'] if s not in nts][-1]
    return {
        'last rule, last terminal in string':last in tokens,
        'string symbols subset of grammar terminals':set(tokens)<=set(terms),
        'grammar terminals subset of string symbols':set(terms)<=set(tokens),
        'same symbol set':set(terms)==set(tokens),
        'always yes':True,
        'some rule has as many terminals as the string':len(tokens) in counts,
        'longest rule no longer than the string':max(counts)<=len(tokens),
    }


def shortcut_scores(rows):
    return {name:{'overall':[sum(shortcut_predictions(r)[name]==(r['target_answer']=='accept') for r in rows),len(rows)],
                  'by_family':{f:[sum(shortcut_predictions(r)[name]==(r['target_answer']=='accept') for r in rows if r['family']==f),sum(r['family']==f for r in rows)] for f in sorted({r['family'] for r in rows})}}
            for name in shortcut_predictions(rows[0])}
