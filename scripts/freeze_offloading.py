"""Freeze the hint protocol only after baseline evidence passes complete replay."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_countpreserving import assemble
from run_ray_offloading import CONDITIONS, STUDY_SETTINGS, suite_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=Path('build/countpreserving-20261005'))
    args = parser.parse_args()
    instances, runs, _ = assemble(args.baseline)
    suites, hashes = suite_inputs()
    if instances != suites:
        raise ValueError('baseline and offloading suites differ')
    protocol = dict(conditions=CONDITIONS, settings=STUDY_SETTINGS, suite_hashes=hashes,
        baseline_gate=dict(replayed_cases=sum(len(r) for r in runs.values()), completed_suites=['A', 'B'],
            completion_sha256={s: hashlib.sha256((args.baseline / s / 'results/completion.json').read_bytes()).hexdigest()
                for s in ('A', 'B')}, baseline_source_commit='3826395'),
        scope='zero-shot only; H0 fresh within each suite job; guard is live-only; no filtering or grading changes')
    output = Path('experiments/offloading/protocol.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != protocol:
        raise ValueError('refusing to overwrite a different frozen protocol')
    output.write_text(json.dumps(protocol, sort_keys=True, indent=2) + '\n')
    print('offloading_protocol_frozen: all 3840 baseline cases replayed before hint jobs')


if __name__ == '__main__':
    main()
