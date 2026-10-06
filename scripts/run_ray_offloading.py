"""Fresh H0-H5, matched serving within each suite, with auditable raw calls."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from abstractgym.controller import teacher_trajectory
from abstractgym.experiment import prompt_for, run_case
def persist(path, destination):   # the reported runs also copied each file to durable cluster storage; the public copy keeps results local
    return None
from run_ray_countpreserving import MODEL, REVISION, SETTINGS, save, verify_inputs

CONDITIONS = {
    'H0': dict(hints=[], legality_guard=False),
    'H1': dict(hints=['untried_rules'], legality_guard=False),
    'H2': dict(hints=[], legality_guard=True),
    'H3': dict(hints=['untried_rules'], legality_guard=True),
    'H4': dict(hints=['untried_rules', 'prefix_matches', 'complete_match'], legality_guard=False),
    'H5': dict(hints=['untried_rules', 'prefix_matches', 'complete_match', 'next_untried'], legality_guard=False),
}
SUITES = ('original', 'cuefree', 'A', 'B')
STUDY_SETTINGS = dict(SETTINGS, retries='one bookkeeping re-ask in guarded live conditions only',
    modes=['external_state', 'oracle_step'], teacher_batching='within case, maximum 32',
    guard_note='pK was already tried here', guard_in_teacher_checks=False)


def suite_inputs():
    suites, count_manifest = verify_inputs()
    whole = Path('experiments/wholerun')
    manifest = json.loads((whole / 'manifest.json').read_text())
    hashes = {str(Path('experiments/countpreserving') / (s + '.json')): count_manifest[s + '.json']
              for s in ('A', 'B')}
    for s in ('original', 'cuefree'):
        path = whole / (s + '.json')
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[s + '.json']:
            raise ValueError('historical suite hash mismatch')
        suites[s] = json.loads(path.read_text())
        hashes[str(path)] = manifest[s + '.json']
    for rows in suites.values():
        if len(rows) != 240 or len({r['task_id'] for r in rows}) != 240:
            raise ValueError('incomplete or duplicate suite')
    return suites, hashes


def verify_study_inputs():
    suites, hashes = suite_inputs()
    protocol = json.loads(Path('experiments/offloading/protocol.json').read_text())
    if (protocol['conditions'] != CONDITIONS or protocol['settings'] != STUDY_SETTINGS
            or protocol['suite_hashes'] != hashes
            or protocol['baseline_gate']['replayed_cases'] != 3840
            or protocol['baseline_gate']['completed_suites'] != ['A', 'B']):
        raise ValueError('offloading prerequisites or protocol mismatch')
    return suites, protocol


def preflight(rows, tokenizer):
    cases = []
    for row in rows:
        maxima = {}
        for name in ('H0', 'H1', 'H4', 'H5'):
            trajectory = teacher_trajectory(row, hints=CONDITIONS[name]['hints'])
            sizes = []
            for step in trajectory:
                prompt = prompt_for(step['observation'], 'external_state')
                tried = step['observation']['stack'][-1]['tried']
                # Include every possible guard note at this valid pre-error state.
                variants = [prompt] + [prompt + '\n' + p + ' was already tried here' for p in tried]
                sizes.extend(len(tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)) for p in variants)
            maxima[name] = max(sizes)
        if max(maxima.values()) + 128 > 16384 or len(trajectory) > 512:
            raise ValueError('fixed context or canonical action budget exceeded')
        cases.append(dict(task_id=row['task_id'], max_prompt_tokens=maxima, canonical_actions=len(trajectory)))
    return dict(cases=cases, context_check_passed=True, policy='no filtering; all retries count within 512 calls')


def evaluate(root, rows, adapter):
    started = time.perf_counter()
    total = 0
    for condition, config in CONDITIONS.items():
        out = root / (condition + '.jsonl')
        count = 0
        begin = time.perf_counter()
        try:
            with out.open('x') as stream:
                for mode in ('external_state', 'oracle_step'):
                    mode_start = time.perf_counter()
                    for index, row in enumerate(rows, 1):
                        record = run_case(row, mode, adapter, max_calls=512, **config)
                        record.update(condition=condition, hints=config['hints'],
                            guard_requested=config['legality_guard'],
                            guard_effective=config['legality_guard'] and mode == 'external_state')
                        stream.write(json.dumps(record, sort_keys=True) + '\n')
                        stream.flush()
                        count += 1
                        total += 1
                        if index % 12 == 0:
                            persist(out, os.environ['RAY_CLI_OUTPUT_DIR'])
                            print('condition_progress ' + json.dumps(dict(condition=condition, mode=mode,
                                cases=index, records=total, seconds=time.perf_counter() - mode_start)), flush=True)
        finally:
            if out.exists():
                persist(out, os.environ['RAY_CLI_OUTPUT_DIR'])
        save(root, condition + '-completion.json', dict(completed=True, cases=count,
            seconds=time.perf_counter() - begin))
        print('condition_complete ' + condition, flush=True)
    save(root, 'completion.json', dict(completed=True, cases=total,
        seconds=time.perf_counter() - started))
    print('offloading_complete', flush=True)


def main():
    from run_ray_cfg import VLLMAdapter   # the vLLM serving adapter used for the reported runs (GPU + vLLM); not part of the public repo
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    suites, protocol = verify_study_inputs()
    if not os.environ.get('RAY_CLI_OUTPUT_DIR'):
        raise RuntimeError('durable output directory required')
    from transformers import AutoTokenizer
    root = Path(tempfile.mkdtemp(prefix='abstractgym-offloading-' + args.suite + '-'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    save(root, 'preflight.json', preflight(suites[args.suite], tokenizer))
    save(root, 'runtime.json', dict(suite=args.suite, protocol=protocol,
        packages={p: importlib.metadata.version(p) for p in ('torch', 'vllm', 'transformers', 'huggingface_hub')},
        hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
            '--format=csv,noheader'], text=True).strip(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src/abstractgym', 'scripts') for p in Path(folder).glob('*.py')}))
    print('preflight_passed ' + args.suite, flush=True)
    if args.preflight_only:
        return
    adapter = VLLMAdapter(MODEL, REVISION, 4096, 16384, 128, batch_size=32, enable_lora=False)
    evaluate(root, suites[args.suite], adapter)


if __name__ == '__main__':
    main()
