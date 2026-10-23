"""Frozen analytical-head endpoint test; no teacher/label/mask reads at inference."""
import argparse
import json
from pathlib import Path
import time
import torch
from analytic_readout_operator import load_adapter, recurrence_mode
from candidate_provenance import file_hash, freeze_config, shuffled_source
from evidence_encoder import condition_tensor
from recurrent_training_protocol import build_operator
from runtime import Runtime, rng_restore, tensor_hash, write_json
from state_space_revisit import full_model_context, replay_state


def main():
    parser = argparse.ArgumentParser()
    for name in ('work', 'experiment', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mode', choices=('loop2', 'no_loop'), required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = json.loads((args.experiment/'preparation_manifest.json').read_text())
    protocol = json.loads((args.experiment/'config.json').read_text())['configuration']
    cohort = json.loads((args.experiment/'cohort.json').read_text())
    validation = json.loads((args.experiment/'validation_status.json').read_text())
    if validation['phase'] != 'complete' or cohort['split'] != 'train' or len(cohort['state_ids']) != 6:
        raise ValueError('Require prevalidated fixed six TRAIN protocol')
    root = args.work/'stage3_grounded_repair'
    for name, digest in manifest['source_sha256'].items():
        if file_hash(root/name) != digest:
            raise ValueError('Frozen v56 source changed: '+name)
    if file_hash(args.experiment/'cohort.json') != manifest['cohort_sha256']:
        raise ValueError('Cohort bytes changed')
    files = {path.stem: path for path in (root/'states/repair_train_targets_v36').glob('*_i35.pt')}
    if set(files) != set(manifest['evidence_state_sha256']):
        raise ValueError('TRAIN evidence set changed')
    for sid, path in files.items():
        if file_hash(path) != manifest['evidence_state_sha256'][sid]:
            raise ValueError('TRAIN evidence bytes changed')
    row = manifest['derivations'][args.mode]
    if file_hash(row['checkpoint']) != row['checkpoint_sha256']:
        raise ValueError('Derived checkpoint changed')
    payload = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
    if payload['derivation']['mode'] != args.mode or payload['derivation']['optimizer_updates_after_source'] != 0:
        raise ValueError('Wrong fixed analytical checkpoint')
    runtime = Runtime(args.work)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261005); torch.cuda.manual_seed_all(20261005)
    op = build_operator(runtime, 'joint_ilsd')
    load_adapter(op, payload['operator'])
    torch.manual_seed(20261005); torch.cuda.manual_seed_all(20261005)
    zero = build_operator(runtime, 'joint_ilsd')
    core_hashes = op.core_hashes()
    actions = protocol['actions'][args.mode]
    frozen = freeze_config(args.output/'config_shard0.json', dict(
        preparation_config_sha256=file_hash(args.experiment/'config.json'),
        preparation_manifest_sha256=file_hash(args.experiment/'preparation_manifest.json'),
        validation_sha256=file_hash(args.experiment/'validation_status.json'),
        checkpoint_sha256=row['checkpoint_sha256'], mode=args.mode, actions=actions,
        cohort_sha256=manifest['cohort_sha256'], split='train', optimizer_updates=0,
        analytical_fit_after_source_step=100, inference_teacher_reads=False,
        inference_pixel_teacher=False, inference_external_masks=False, controller=False, FLOPs=None))
    metadata = manifest['state_metadata']
    with (args.output/'actions_shard0.jsonl').open('w') as stream:
        for sid in cohort['state_ids']:
            state = torch.load(files[sid], map_location='cpu', weights_only=False)
            if (state['state_id'] != sid or state['split'] != 'train' or state['flow_index'] != 35 or
                    not state['zero_intervention_test']['passed'] or
                    tensor_hash(state['latent']) != state['state_fingerprint'] or
                    state['state_fingerprint'] != metadata[sid]['state_fingerprint']):
                raise ValueError('Captured TRAIN state identity/fingerprint mismatch')
            baseline = state['baseline_final_latent'].to(runtime.device).float()
            for action in actions:
                current = zero if action in ('baseline', 'conditional_zero') else op
                evidence = state['evidence']; source_id = None
                if action == 'shuffled_evidence':
                    source_id = shuffled_source(metadata, sid)
                    other = torch.load(files[source_id], map_location='cpu', weights_only=False)
                    if (other['split'] != 'train' or other['flow_index'] != 35 or
                            tensor_hash(other['latent']) != other['state_fingerprint'] or
                            other['state_fingerprint'] != metadata[source_id]['state_fingerprint']):
                        raise ValueError('Shuffled evidence provenance changed')
                    evidence = other['evidence']
                elif action == 'opposite_evidence':
                    evidence = dict(evidence)
                    for a, b in (('target_margin', 'opposite_margin'), ('target_hidden', 'opposite_hidden')):
                        evidence[a], evidence[b] = evidence[b], evidence[a]
                    evidence['evidence'] = -evidence['evidence']
                current.condition = condition_tensor(evidence, state['true_t'], runtime.device)
                current.loop_count = 1 if action == 'conditional_budget1' else 2
                rng_restore(state['rng']); torch.cuda.synchronize(); start = time.perf_counter()
                with torch.no_grad(), full_model_context(runtime), recurrence_mode(current, args.mode) as counts:
                    final, cost = replay_state(runtime, state, current, apply_update=action != 'baseline')
                expected = 0 if action == 'baseline' or args.mode == 'no_loop' else 2*current.loop_count
                if counts != dict(outer=expected, inner=expected):
                    raise ValueError('Actual shared core call count mismatch')
                if action in ('baseline', 'conditional_zero') and not torch.equal(final, baseline):
                    raise ValueError('Full14 baseline/fresh zero is not strictly identical')
                image = runtime.decode(final); torch.cuda.synchronize()
                cost.update(wall_time_s=time.perf_counter()-start,
                    outer_shared_revisit_calls=counts['outer'], inner_shared_revisit_calls=counts['inner'],
                    recurrent_loop_budget=0 if expected == 0 else current.loop_count)
                full_calls = 49 if action == 'baseline' else 50
                cost.update(full_network_velocity_calls_nominal=full_calls,
                    full_outer_backbone_layer_calls_nominal=full_calls*28+counts['outer'],
                    full_flow_head_layer_calls_nominal=full_calls*10+counts['inner'],
                    full_semantic_encoder_layer_calls_nominal=full_calls*26)
                directory = args.output/action/sid
                (directory/'samples').mkdir(parents=True)
                image_path = directory/'samples/0000.png'; image.save(image_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                final_path = directory/'final_latent.pt'
                torch.save(dict(latent=final.cpu(), state_id=sid, action=action), final_path)
                record = {key: state[key] for key in ('state_id', 'prompt_id', 'prompt', 'seed', 'split',
                    'flow_index', 'true_t', 'state_fingerprint', 'zero_intervention_test', 'runtime')}
                record.update(action=action, configuration_sha256=frozen,
                    image_path=str(image_path), final_latent=str(final_path),
                    image_path_sha256=file_hash(image_path), final_latent_sha256=file_hash(final_path),
                    cost=cost, checkpoint=row['checkpoint'], arm=args.mode, study='analytical_readout',
                    shuffled_source_state=source_id,
                    added_trainable_parameters=0 if action == 'baseline' else sum(p.numel() for p in current.trainable_parameters()),
                    stored_adapter_parameters=sum(p.numel() for p in current.trainable_parameters()),
                    analytical_fit_after_optimizer_step=100, optimizer_updates_this_experiment=0,
                    cost_notes=dict(scope='remaining14 velocities, feature capture and decode; not end-to-end',
                        old_inner_prefix_reused=False, old_KV_reused=False,
                        MMU_checks_if_deployed=0 if action == 'baseline' else 2,
                        evidence_precomputed=True, external_detector_used=False, FLOPs_measured=False,
                        no_loop_low_rank_core_parameters_stored_but_not_executed=args.mode == 'no_loop'), FLOPs=None)
                stream.write(json.dumps(record)+'\n'); stream.flush()
                print(json.dumps(dict(mode=args.mode, state_id=sid, action=action, status='complete')), flush=True)
    if op.core_hashes() != core_hashes or any(p.grad is not None for p in runtime.model.parameters()):
        raise ValueError('Base core/gradient boundary changed during inference')
    if {key: tensor_hash(v) for key, v in op.adapter_state().items()} != row['adapter_tensor_hashes']:
        raise ValueError('Frozen derived adapter changed')
    write_json(args.output/'generation_status.json', dict(success=True, expected_images=6*len(actions),
        optimizer_steps=0, split='train', arm=args.mode, val_images=0, controller_started=False,
        strict_baseline_and_fresh_zero_checks=12, inference_teacher_reads=False))


if __name__ == '__main__':
    main()
