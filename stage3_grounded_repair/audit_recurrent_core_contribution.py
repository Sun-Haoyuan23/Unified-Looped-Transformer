"""Read-only v46 fixed-checkpoint audit: BF16 weights and core-learning ablation.

Replay existing TRAIN outputs exactly before interpreting a counterfactual.
LoRA-off retains the same recurrent blocks, evidence and trained readout. This
isolates dependence on the learned low-rank update at inference; it is not a
matched retraining experiment or evidence of generation quality improvement.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from candidate_provenance import file_hash, freeze_config
from evidence_encoder import condition_tensor
from recurrent_training_protocol import STUDIES, build_operator
from runtime import Runtime, rng_restore, tensor_hash, write_json
from state_space_revisit import capture_features, full_model_context
from trainable_recurrent_state import replay_delta
from training_pairs import selected_pairs, validate_training_pairs


def rms(value):
    return float(value.float().square().mean().sqrt())


def alignment(displacement, target):
    a, b = displacement.float().flatten(), target.float().flatten()
    dot = torch.dot(a, b)
    bb = torch.dot(b, b).clamp_min(1e-20)
    projection = dot / bb
    return dict(cosine=float(dot / (a.norm() * b.norm()).clamp_min(1e-20)),
                target_projection=float(projection),
                orthogonal_rms=rms(a - projection * b),
                displacement_rms=rms(a), target_rms=rms(b))


def weight_audit(op):
    rows = []
    for branch_name, branch in (('outer', op.outer), ('inner', op.inner)):
        if branch is None:
            continue
        for index, core in enumerate(branch.shared_core):
            weights = dict(core.named_parameters())
            for name, key in branch.targets[index].items():
                weight = weights[name].detach()
                low_rank = branch.low_rank[key]
                raw = low_rank.delta().detach()
                effective = weight + raw.to(weight.dtype)
                actual = effective.float() - weight.float()
                rows.append(dict(branch=branch_name, core_index=index, weight=name,
                    elements=weight.numel(), dtype=str(weight.dtype),
                    A_rms=rms(low_rank.A), B_rms=rms(low_rank.B),
                    raw_delta_rms=rms(raw), base_rms=rms(weight),
                    effective_delta_rms=rms(actual),
                    effective_to_raw_rms=rms(actual) / max(rms(raw), 1e-20),
                    effective_to_base_rms=rms(actual) / max(rms(weight), 1e-20),
                    changed_elements=int((effective != weight).sum()),
                    changed_fraction=float((effective != weight).float().mean()),
                    raw_delta_sha256=tensor_hash(raw),
                    effective_weight_sha256=tensor_hash(effective),
                    base_weight_sha256=tensor_hash(weight)))
    return rows


def main():
    p = argparse.ArgumentParser()
    for name in ('work', 'states', 'teacher-root', 'cohort', 'checkpoint', 'reference', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--study', choices=STUDIES, required=True)
    args = p.parse_args()
    args.output.mkdir()
    try:
        cohort = json.loads(args.cohort.read_text())
        teacher_report = json.loads((args.teacher_root / 'paired_report.json').read_text())
        records = [json.loads(s) for s in (args.teacher_root / 'all_actions.jsonl').read_text().splitlines()]
        gate = validate_training_pairs(teacher_report, records)
        pairs = selected_pairs(teacher_report, records)
        if (gate['selected_teacher_states'] != 3 or gate['preservation_states'] != 35 or
                cohort['split'] != 'train' or len(cohort['state_ids']) != 6 or
                cohort['teacher_state_ids'] != sorted(pairs) or
                cohort['teacher_report_sha256'] != file_hash(args.teacher_root / 'paired_report.json')):
            raise ValueError('Frozen TRAIN gate/cohort changed')
        baselines = {r['state_id']: r for r in records if r['action'] == 'baseline'}
        if len(baselines) != 112 or any(r['split'] != 'train' or r['flow_index'] != 35 for r in baselines.values()):
            raise ValueError('Canonical TRAIN source changed')
        reference = [json.loads(s) for s in (args.reference / 'all_actions.jsonl').read_text().splitlines()]
        reference = {r['state_id']: r for r in reference if r['action'] == 'conditional_revisit'}
        if set(reference) != set(cohort['state_ids']):
            raise ValueError('Replay reference cohort mismatch')
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        config = payload['metadata']
        if (payload['step'] != 100 or payload['level'] != 2 or payload['width'] != 64 or
                config['operator_kind'] != 'recurrent_' + args.study or config['window'] != 1 or
                config['paired_report_sha256'] != cohort['teacher_report_sha256']):
            raise ValueError('Require frozen final checkpoint and TRAIN source')
        source = Path(__file__).parent
        for name, digest in config['training_code_sha256'].items():
            if file_hash(source / name) != digest:
                raise ValueError('Frozen training interface changed: ' + name)
        for row in reference.values():
            if row['checkpoint'] != str(args.checkpoint) or row['study'] != args.study:
                raise ValueError('Wrong checkpoint reference')
        freeze_config(args.output / 'config.json', dict(study=args.study, optimizer_updates=0,
            cohort_sha256=file_hash(args.cohort), checkpoint_sha256=file_hash(args.checkpoint),
            reference_records_sha256=file_hash(args.reference / 'all_actions.jsonl'),
            source_sha256={n:file_hash(source / n) for n in (
                'audit_recurrent_core_contribution.py', 'trainable_recurrent_state.py',
                'recurrent_training_protocol.py', 'runtime.py', 'state_space_revisit.py',
                'evidence_encoder.py', 'training_pairs.py', 'candidate_provenance.py')},
            actions=['learned_core_replay', 'trained_readout_frozen_core'], loops=2,
            teacher_read_only_after_both_predictions=True, inference_external_masks=False,
            inference_pixel_teacher=False, output_images=False, score_claim=False,
            note='Inference counterfactual, not matched retraining or independent validation'))
        runtime = Runtime(args.work, vae=False)
        torch.manual_seed(20261005)
        torch.cuda.manual_seed_all(20261005)
        op = build_operator(runtime, args.study)
        expected = {k for k in op.state_dict() if '.shared_core.' not in k}
        if set(payload['operator']) != expected:
            raise ValueError('Checkpoint tensors incompatible')
        loaded = op.load_state_dict(payload['operator'], strict=False)
        if loaded.unexpected_keys or set(loaded.missing_keys) != set(op.state_dict()) - expected:
            raise ValueError('Checkpoint load mismatch')
        op.loop_count = 2
        base_hash = op.core_hashes()
        adapter_hash = {k:tensor_hash(v) for k,v in op.adapter_state().items()}
        with torch.no_grad():
            write_json(args.output / 'effective_weight_audit.json', weight_audit(op))
        results = []
        with (args.output / 'audit.jsonl').open('w') as log:
            for sid in cohort['state_ids']:
                path = args.states / (sid + '.pt')
                if file_hash(path) != cohort['state_sha256'][sid]:
                    raise ValueError('Frozen state bytes changed')
                state = torch.load(path, map_location='cpu', weights_only=False)
                if (state['state_id'] != sid or state['split'] != 'train' or state['flow_index'] != 35 or
                        not state['zero_intervention_test']['passed'] or
                        tensor_hash(state['latent']) != state['state_fingerprint'] or
                        state['state_fingerprint'] != baselines[sid]['state_fingerprint']):
                    raise ValueError('TRAIN capture provenance changed')
                op.condition = condition_tensor(state['evidence'], state['true_t'], runtime.device)
                rng_restore(state['rng'])
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad(), full_model_context(runtime):
                    features, options = capture_features(runtime, state, op)
                    original = state['latent'].to(runtime.device)
                    predictions, deltas, costs = {}, {}, {}
                    for action, enabled in (('learned_core_replay', True), ('trained_readout_frozen_core', False)):
                        op.set_core_learning(enabled)
                        t0 = time.perf_counter()
                        delta = op.latent_delta(runtime, features, options['modality_positions'], original)
                        final = replay_delta(runtime, state, delta)
                        torch.cuda.synchronize()
                        predictions[action] = final
                        deltas[action] = delta
                        costs[action] = dict(post_update_full_velocity_calls=14,
                            extra_outer_shared_blocks=0 if op.outer is None else 4,
                            extra_inner_shared_blocks=0 if op.inner is None else 4,
                            wall_time_s=time.perf_counter() - t0, FLOPs=None)
                    op.set_core_learning(True)
                row = reference[sid]
                if file_hash(row['final_latent']) != row['final_latent_sha256']:
                    raise ValueError('Reference terminal bytes changed')
                previous = torch.load(row['final_latent'], map_location=runtime.device, weights_only=False)
                if previous['state_id'] != sid or previous['action'] != 'conditional_revisit':
                    raise ValueError('Reference terminal identity changed')
                if not torch.equal(predictions['learned_core_replay'], previous['latent']):
                    raise ValueError('Learned-core replay not strictly identical to v47: ' + sid)
                # Targets are read only now, after both inference predictions.
                baseline = state['baseline_final_latent'].to(runtime.device).float()
                target = baseline
                if sid in pairs:
                    row = pairs[sid]
                    if file_hash(row['final_latent']) != row['final_latent_sha256']:
                        raise ValueError('Teacher file changed')
                    teacher = torch.load(row['final_latent'], map_location=runtime.device, weights_only=False)
                    if teacher['state_id'] != sid or teacher['action'] != row['action']:
                        raise ValueError('Teacher identity mismatch')
                    target = teacher['latent'].float()
                normalizer = ((target - baseline).square().mean() if sid in pairs else baseline.square().mean()).clamp_min(1e-5)
                teacher_rows = {}
                for action in predictions:
                    final = predictions[action]
                    saved = args.output / (sid + '__' + action + '.pt')
                    torch.save(dict(latent=final.cpu(), physical_delta=deltas[action].cpu(),
                        state_id=sid, action=action, study=args.study), saved)
                    teacher_rows[action] = dict(normalized_target_mse=float((final-target).square().mean()/normalizer),
                        physical_delta_rms=rms(deltas[action]),
                        terminal_alignment=alignment(final-baseline, target-baseline),
                        latent_path=str(saved), latent_sha256=file_hash(saved), cost=costs[action])
                on, off = predictions['learned_core_replay'], predictions['trained_readout_frozen_core']
                result = dict(state_id=sid, study=args.study, split='train',
                    cohort='teacher' if sid in pairs else 'preservation', flow_index=35, loops=2,
                    replay_exact=True, condition_rms=rms(op.condition),
                    latent_core_contribution_rms=rms(deltas['learned_core_replay']-deltas['trained_readout_frozen_core']),
                    latent_core_contribution_relative_to_learned=rms(deltas['learned_core_replay']-deltas['trained_readout_frozen_core'])/max(rms(deltas['learned_core_replay']),1e-20),
                    terminal_core_contribution_rms=rms(on-off),
                    terminal_changed_fraction=float((on!=off).float().mean()),
                    measurements=teacher_rows, wall_time_s=time.perf_counter()-started,
                    original_feature_capture_full_velocity_calls=1, optimizer_updates=0,
                    new_image_scores=0, old_KV_or_inner_prefix=False)
                log.write(json.dumps(result)+'\n')
                log.flush()
                results.append(result)
                print(json.dumps(dict(study=args.study,state_id=sid,replay_exact=True)),flush=True)
        if op.core_hashes() != base_hash or {k:tensor_hash(v) for k,v in op.adapter_state().items()} != adapter_hash:
            raise ValueError('Read-only audit changed a base/adapter tensor')
        if any(q.grad is not None for q in runtime.model.parameters()) or any(q.grad is not None for q in op.trainable_parameters()):
            raise ValueError('Unexpected gradient in read-only audit')
        write_json(args.output / 'audit_complete.json', dict(success=True, strict_replays=len(results),
            counterfactual_rollouts=len(results), optimizer_updates=0, new_scored_images=0,
            core_hash_unchanged=True, adapter_hash_unchanged=True, base_gradients=False,
            independent_validation=False, controller_started=False))
    except Exception as error:
        write_json(args.output / 'audit_failed.json', dict(error=str(error), optimizer_updates=0))
        raise


if __name__ == '__main__':
    main()
