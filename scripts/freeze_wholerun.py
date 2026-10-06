"""Freeze training examples, suites, and historical evidence before inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from abstractgym.a6 import digest
from abstractgym.a6_objectives import dfs_examples
from abstractgym.complexity import build_complexity_suite, build_cuefree_complexity_suite
from abstractgym.wholerun import ARMS, fulltrace_examples, training_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('experiments/wholerun'))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    full = fulltrace_examples()
    steps = dfs_examples()
    x7 = json.loads(Path('docs/results/2026-10-04-x7/metadata.json').read_text())['training']['training']
    assert len(full) == 64 and len(steps) == 11002
    assert digest(steps) == x7['training_hash']
    save('fulltrace.json', full)
    (out / 'single_step.json.gz').write_bytes(gzip.compress(json.dumps(steps, sort_keys=True).encode(), mtime=0))
    for arm in ARMS:
        assert {r['trajectory'] for r in training_examples(arm)} == {r['trajectory'] for r in full}
    original, cuefree = build_complexity_suite(), build_cuefree_complexity_suite()
    assert cuefree == json.loads(Path('experiments/cuefree/instances.json').read_text())
    assert original[::2] == cuefree[::2]
    save('original.json', original)
    save('cuefree.json', cuefree)
    sources = [Path('docs/results/2026-10-04-x7') / n for n in ('audit.json.gz', 'metadata.json', 'summary.json')]
    sources += [Path('docs/results/cuefree') / n for n in ('audit.json.gz', 'metadata.json', 'summary.json')]
    save('protocol.json', dict(model=x7['model'], revision=x7['revision'], seed=17,
        arms=list(ARMS), references=['dfs_lora', 'zeroshot'], training=x7,
        matched_by='supervised target tokens including EOS, stopping at first effective batch reaching budget',
        matched_target_tokens=x7['scheduled_supervised_tokens'], epochs=3,
        batch_policy='X7 microbatch 8 and accumulation 4; no truncation; use checkpointing',
        mixed_sampling='all 11002 single-step examples plus 64 full traces, shuffled jointly each epoch',
        evaluation=dict(suites=['original', 'cuefree'], cases_per_suite=240,
            modes=['direct', 'full_trace', 'external_state', 'oracle_step'],
            direct_tokens=128, trace_tokens=4096, step_tokens=128, max_calls=512,
            context=16384, thinking=False, temperature=0, top_p=1, top_k=-1,
            direct_batch=32, teacher_batch=32, execution_batch=1, examples=0, retries=0, repairs=0),
        historical_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        bracket='skipped: distinct action protocol; no new bracket training examples'))
    save('manifest.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'})
    print('frozen 64 full traces / 11002 steps / two 240-case suites')


if __name__ == '__main__':
    main()
