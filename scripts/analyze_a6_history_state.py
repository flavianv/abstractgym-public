"""Post-hoc X5 diagnostic: original versus corrupted-stack-consistent suffixes.

This never modifies primary scores, outputs, or forced prefixes.
"""
import gzip
import json
from pathlib import Path
from abstractgym.brackets import expected,advance
from abstractgym.a6_trace import reference,injected_prefix


def corrupted_reference(row,k):
    forced=json.loads(injected_prefix(row,k)[:-1]+']');pos,mode,stack=0,'opening',[]
    for action in forced:pos,mode,stack=advance(action,pos,mode,stack)
    result=[]
    for _ in range(len(row['input'])+1):
        action=expected(row,pos,mode,stack);result.append(action)
        if action['op'] in ('ACCEPT','REJECT'):return result
        pos,mode,stack=advance(action,pos,mode,stack)
    raise AssertionError('nonterminating reference')


def summarize(audit):
    rows={r['id']:r for r in audit['instances']};result=[]
    for model,run in audit['runs'].items():
        for k,records in run['injection'].items():
            cells=[]
            for r in records:
                original=reference(rows[r['id']])[4:];corrupted=corrupted_reference(rows[r['id']],int(k))
                suffix=r['actions'][4:] if isinstance(r['actions'],list) else None
                cells.append(dict(id=r['id'],targets_differ=original!=corrupted,original_correct=suffix==original,corrupted_correct=suffix==corrupted))
            result.append(dict(model=model,k=int(k),total=len(cells),targets_differ=sum(c['targets_differ'] for c in cells),original_correct=sum(c['original_correct'] for c in cells),corrupted_correct=sum(c['corrupted_correct'] for c in cells),corrupted_only=[c['id'] for c in cells if c['corrupted_correct'] and not c['original_correct']],cases=cells))
    return result


def main():
    root=Path('docs/results/2026-10-04-x5');audit=json.loads(gzip.decompress((root/'audit.json.gz').read_bytes()))
    results=summarize(audit)
    (root/'posthoc_corrupted_state.json').write_text(json.dumps(dict(label='post-hoc diagnostic; primary X5 scores unchanged',results=results),indent=2)+'\n')
    for r in results:print(r['model'],r['k'],r['targets_differ'],r['original_correct'],r['corrupted_correct'],len(r['corrupted_only']))


if __name__=='__main__':main()
