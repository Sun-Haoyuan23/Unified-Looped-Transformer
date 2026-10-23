"""v70 TRAIN-only composition. Frozen v49/v50 and v69 are never modified."""
import json
import math
import random
from pathlib import Path

import torch
from candidate_provenance import file_hash
from process_training_targets import ProcessCorpus
from runtime import tensor_hash
from state_space_revisit import capture_features

NATIVE_IDS = tuple(f'train_extra_{i:05d}_seed101_i35' for i in (6, 10, 54))


def check_quality(row, pair, baseline, canonical_pair):
    for key in ('state_id', 'split', 'prompt', 'prompt_id', 'seed', 'flow_index', 'true_t', 'state_fingerprint'):
        if row[key] != baseline[key]:
            raise ValueError('Native teacher identity mismatch: ' + key)
    if row['split'] != 'train' or pair['baseline_correct'] != 0 or pair['candidate_correct'] != 1:
        raise ValueError('Require an actual TRAIN wrong-to-correct teacher')
    result = {}
    for metric in ('hps', 'pickscore'):
        base = canonical_pair['baseline_metrics'][metric]
        value = pair['candidate_metrics'][metric]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (base, value)):
            raise ValueError('Nonfinite teacher quality')
        if pair['baseline_metrics'][metric] != base or value < base:
            raise ValueError('Quality lost against original canonical baseline')
        result[metric] = value - base
    return result


def checked_additive(original, delta, expected):
    if original.dtype != torch.float32 or delta.dtype != torch.float32:
        raise ValueError('Require original FP32 arithmetic')
    if original.shape != delta.shape or original.shape != expected.shape:
        raise ValueError('Physical shape mismatch')
    if not all(bool(torch.isfinite(q).all()) for q in (original, delta, expected)):
        raise ValueError('Nonfinite physical target')
    if not torch.equal(delta[:1], delta[1:]) or not torch.equal(original[:1], original[1:]):
        raise ValueError('Physical CFG branches disagree')
    actual = original + delta
    if not torch.equal(actual, expected):
        raise ValueError('Delta does not reproduce the scored additive input')
    return actual


def expanded_schedule(teachers, correct, seed=20261008):
    teachers, correct = sorted(teachers), sorted(correct)
    if len(teachers) != 6 or len(correct) != 35 or set(teachers) & set(correct):
        raise ValueError('Require six qualified teachers and 35 real correct states')
    rng = random.Random(seed)
    repairs = teachers * 35
    preserve = correct * 6
    rng.shuffle(repairs)
    rng.shuffle(preserve)
    return [(sid, keep) for a, b in zip(repairs, preserve) for sid, keep in ((a, False), (b, True))]


class ExpandedProcessCorpus:
    def __init__(self, root):
        self.root = Path(root)
        self.old = ProcessCorpus(self.root)
        self.correct = self.old.correct
        self.baselines = self.old.baselines
        self.teachers = dict(self.old.teachers)
        self.input_hashes = dict(self.old.input_hashes)
        native = self.root / 'native_teacher_additive_v69'
        proof_path = self.root / 'v69_portable_verification.json'
        proof = json.loads(proof_path.read_text())
        report_path = native / 'canonical_paired_report.json'
        if (not proof['passed'] or proof['qualified_state_ids'] != list(NATIVE_IDS)
                or proof['canonical_paired_report_sha256'] != file_hash(report_path)
                or json.loads((native / 'pipeline_status.json').read_text())['phase'] != 'complete'):
            raise ValueError('Native additive quality/provenance proof incomplete')
        report = json.loads(report_path.read_text())
        if not report['matched_zero_intervention_passed']:
            raise ValueError('Native matched-zero proof missing')
        self.native_process = {q['state_id']: q for q in json.loads((native / 'process_manifest.json').read_text())}
        records = [json.loads(q) for q in (native / 'all_actions.jsonl').read_text().splitlines()]
        native_teachers = {q['state_id']: q for q in records if q['action'] == 'native_additive_teacher'}
        pairs = {q['state_id']: q for q in report['pairs'] if q['action'] == 'native_additive_teacher'}
        canonical = json.loads((self.root / 'repair_train_targets_v36/paired_report.json').read_text())
        canonical_pairs = {q['state_id']: q for q in canonical['pairs'] if q['action'] == 'training_target'}
        if set(native_teachers) != set(NATIVE_IDS) or set(self.native_process) != set(NATIVE_IDS) or set(pairs) != set(NATIVE_IDS):
            raise ValueError('Native teacher set changed')
        self.quality_deltas = dict(self.old.original_baseline_deltas)
        for sid, row in native_teachers.items():
            self.quality_deltas[sid] = check_quality(row, pairs[sid], self.baselines[sid], canonical_pairs[sid])
            for field in ('image_path', 'final_latent'):
                if file_hash(row[field]) != row[field + '_sha256']:
                    raise ValueError('Native scored artifact changed')
            artifact = self.native_process[sid]
            if file_hash(artifact['artifact_path']) != artifact['artifact_sha256']:
                raise ValueError('Native process artifact changed')
        self.teachers.update(native_teachers)
        if len(self.teachers) != 6 or len(self.correct) != 35 or set(self.teachers) & set(self.correct):
            raise ValueError('Expanded cohort overlap or size changed')
        for path in (proof_path, report_path, native / 'all_actions.jsonl', native / 'process_manifest.json', native / 'config.json', native / 'scores.jsonl'):
            self.input_hashes[str(path.relative_to(self.root))] = file_hash(path)

    def state(self, sid):
        if sid not in self.teachers and sid not in self.correct:
            raise ValueError('Not a qualified TRAIN member')
        if sid not in NATIVE_IDS:
            return self.old.state(sid)
        path = self.root / 'states/repair_train_targets_v36' / (sid + '.pt')
        state = torch.load(path, map_location='cpu', weights_only=False)
        row = self.baselines[sid]
        if (state['state_id'] != sid or state['split'] != 'train' or state['flow_index'] != 35
                or not state['zero_intervention_test']['passed']
                or tensor_hash(state['latent']) != state['state_fingerprint']
                or state['state_fingerprint'] != row['state_fingerprint']):
            raise ValueError('Native TRAIN state corrupted')
        for key in ('prompt', 'prompt_id', 'seed', 'true_t'):
            if state[key] != row[key]:
                raise ValueError('Native state identity changed')
        if file_hash(path) != self.native_process[sid]['source_state_sha256']:
            raise ValueError('Native source state bytes differ')
        return state

    def tensors(self, state):
        sid = state['state_id']
        original = state['latent'].float().cpu()
        if sid in self.correct:
            return dict(delta=torch.zeros_like(original), input=original, final=state['baseline_final_latent'].float().cpu(), preservation=True)
        if sid not in self.teachers:
            raise ValueError('Teacher absent')
        native = sid in NATIVE_IDS
        artifact = self.native_process[sid] if native else self.old.process[sid]
        if file_hash(artifact['artifact_path']) != artifact['artifact_sha256']:
            raise ValueError('Process file changed')
        payload = torch.load(artifact['artifact_path'], map_location='cpu', weights_only=False)
        if (payload['state_id'] != sid or payload['split'] != 'train'
                or payload['source_state_fingerprint'] != state['state_fingerprint']):
            raise ValueError('Process identity mismatch')
        delta = payload['physical_delta']
        target_input = original + delta
        if native:
            if not torch.equal(original, payload['original_input']):
                raise ValueError('Native original tensor differs')
            target_input = checked_additive(original, delta, payload['target_input'])
            final = payload['target_final']
            expected_hashes = artifact['tensor_sha256']
            for key, tensor in dict(original=original, physical_delta=delta, target_input=target_input, target_final=final).items():
                if tensor_hash(tensor) != expected_hashes[key]:
                    raise ValueError('Native tensor hash differs: ' + key)
        else:
            # v49 direct teacher features are not the v50 additive features.
            final = payload['additive_final']
            checked_additive(original, delta, target_input)
        row = self.teachers[sid]
        scored = torch.load(row['final_latent'], map_location='cpu', weights_only=False)
        if (scored['state_id'] != sid or scored['action'] != row['action']
                or not torch.equal(scored['latent'], final) or not bool(torch.isfinite(final).all())):
            raise ValueError('Target final is not the scored additive endpoint')
        return dict(delta=delta.detach(), input=target_input.detach(), final=final.float().detach(), preservation=False)

    def target_features(self, runtime, state, operator):
        target = self.tensors(state)
        target_state = dict(state, latent=target['input'], state_fingerprint=tensor_hash(target['input']))
        with torch.no_grad():
            features, options = capture_features(runtime, target_state, operator)
        return target, features, options

    def ledger(self):
        rows = []
        for sid in sorted(set(self.teachers) | set(self.correct)):
            state = self.state(sid)
            target = self.tensors(state)
            rows.append(dict(state_id=sid, split='train', prompt=state['prompt'], prompt_id=state['prompt_id'], seed=state['seed'],
                flow_index=35, preservation=target['preservation'], state_fingerprint=state['state_fingerprint'],
                state_file_sha256=file_hash(self.root / 'states/repair_train_targets_v36' / (sid + '.pt')),
                tensor_sha256={key: tensor_hash(target[key]) for key in ('delta', 'input', 'final')},
                quality_deltas=None if target['preservation'] else self.quality_deltas[sid],
                target_source='baseline' if target['preservation'] else ('v69_additive' if sid in NATIVE_IDS else 'v50_additive')))
        return rows
