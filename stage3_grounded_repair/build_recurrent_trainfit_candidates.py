"""Fixed TRAIN capacity diagnostic; inference reads only captured MMU/state data."""
import argparse
import json
import time
from pathlib import Path
import torch
from runtime import Runtime, rng_restore, tensor_hash, write_json
from evidence_encoder import condition_tensor
from state_space_revisit import full_model_context, replay_state
from recurrent_training_protocol import build_operator,STUDIES
from candidate_provenance import file_hash, freeze_config, shuffled_source
from training_pairs import selected_pairs, validate_training_pairs


def main():
    p = argparse.ArgumentParser()
    for name in ('work', 'states', 'teacher-root', 'cohort', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--study', choices=STUDIES, required=True)
    args = p.parse_args()
    args.arm=args.study.split('_',1)[0]
    source = Path(__file__).parent
    args.output.mkdir()
    cohort = json.loads(args.cohort.read_text())
    report = json.loads((args.teacher_root/'paired_report.json').read_text())
    records = [json.loads(l) for l in (args.teacher_root/'all_actions.jsonl').read_text().splitlines()]
    gate = validate_training_pairs(report, records)
    pairs = selected_pairs(report, records)
    if gate['selected_teacher_states'] != 3 or gate['preservation_states'] != 35:
        raise ValueError('Strict TRAIN corpus changed')
    if (cohort['split'] != 'train' or len(cohort['state_ids']) != 6 or
            cohort['teacher_state_ids'] != sorted(pairs) or
            cohort['teacher_report_sha256'] != file_hash(args.teacher_root/'paired_report.json')):
        raise ValueError('Diagnostic cohort provenance changed')
    baselines = {row['state_id']:row for row in records if row['action'] == 'baseline'}
    if len(baselines) != 112 or any(row['split'] != 'train' or row['flow_index'] != 35 for row in baselines.values()):
        raise ValueError('Evidence must use the frozen 112 TRAIN states at index35')
    files = {q.stem:q for q in args.states.glob('*_i35.pt')}
    if set(files) != set(baselines):
        raise ValueError('Evidence cohort incomplete or contains an extra split')
    metadata = {sid:{k:row[k] for k in ('split', 'flow_index', 'prompt_id')} for sid,row in baselines.items()}
    for sid in cohort['state_ids']:
        if file_hash(files[sid]) != cohort['state_sha256'][sid]:
            raise ValueError('Frozen diagnostic state bytes changed')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = payload['metadata']
    if (payload['step'] != 100 or payload['level'] != 2 or payload['width'] != 64 or
            config['operator_kind'] != 'recurrent_' + args.study or
            config['paired_report_sha256'] != cohort['teacher_report_sha256'] or
            config['window'] != 1):
        raise ValueError('Require fixed final step100, matching architecture and TRAIN source')
    for name,digest in config['training_code_sha256'].items():
        if file_hash(source/name) != digest:
            raise ValueError('Training/inference interface changed: ' + name)
    runtime = Runtime(args.work)
    torch.manual_seed(20261005); torch.cuda.manual_seed_all(20261005)
    op = build_operator(runtime, args.study)
    zero = build_operator(runtime, args.study)
    expected = {k for k in op.state_dict() if '.shared_core.' not in k}
    if set(payload['operator']) != expected:
        raise ValueError('Incomplete or incompatible trained adapter')
    loaded = op.load_state_dict(payload['operator'], strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) != set(op.state_dict()) - expected:
        raise ValueError('Unexpected checkpoint loading mismatch')
    core_hash = op.core_hashes()
    actions = ('baseline', 'conditional_zero', 'conditional_revisit', 'shuffled_evidence', 'opposite_evidence', 'conditional_budget1')
    frozen = freeze_config(args.output/'config_shard0.json', dict(
        actions=actions, split='train', arm=args.arm, window=1,
        checkpoint_sha256=file_hash(args.checkpoint), cohort_sha256=file_hash(args.cohort),
        teacher_report_sha256=cohort['teacher_report_sha256'],
        state_sha256=cohort['state_sha256'], evidence_state_sha256={sid:file_hash(q) for sid,q in files.items()},
        source_sha256={n:file_hash(source/n) for n in ('build_recurrent_trainfit_candidates.py','trainable_recurrent_state.py','recurrent_training_protocol.py','state_space_revisit.py','runtime.py','candidate_provenance.py','evidence_encoder.py','training_pairs.py')},
        checkpoint_selection='fixed final step100; TRAIN capacity diagnostic, not independent validation',
        inference_external_masks=False, inference_pixel_teacher=False, teacher_used_only_for_offline_distance=True))
    with (args.output/'actions_shard0.jsonl').open('w') as stream:
        for sid in cohort['state_ids']:
            state = torch.load(files[sid], map_location='cpu', weights_only=False)
            if (state['state_id'] != sid or state['split'] != 'train' or state['flow_index'] != 35 or
                    not state['zero_intervention_test']['passed'] or
                    tensor_hash(state['latent']) != state['state_fingerprint'] or
                    state['state_fingerprint'] != baselines[sid]['state_fingerprint']):
                raise ValueError('Captured TRAIN state identity/fingerprint mismatch')
            baseline = state['baseline_final_latent'].to(runtime.device).float()
            # Targets measure fitting after inference; never decide an inference action.
            target = baseline
            if sid in pairs:
                row = pairs[sid]
                if file_hash(row['final_latent']) != row['final_latent_sha256']:
                    raise ValueError('Qualified teacher bytes changed')
                teacher = torch.load(row['final_latent'], map_location=runtime.device, weights_only=False)
                if teacher['state_id'] != sid or teacher['action'] != row['action']:
                    raise ValueError('Wrong terminal teacher identity')
                target = teacher['latent'].float()
            for action in actions:
                current = zero if action in ('baseline', 'conditional_zero') else op
                evidence = state['evidence']; source_id = None
                if action == 'shuffled_evidence':
                    source_id = shuffled_source(metadata, sid)
                    other = torch.load(files[source_id], map_location='cpu', weights_only=False)
                    if (other['split'] != 'train' or other['flow_index'] != 35 or
                            tensor_hash(other['latent']) != other['state_fingerprint'] or
                            other['state_fingerprint'] != baselines[source_id]['state_fingerprint']):
                        raise ValueError('Shuffled evidence provenance changed')
                    evidence = other['evidence']
                elif action == 'opposite_evidence':
                    evidence = dict(evidence)
                    for a,b in (('target_margin', 'opposite_margin'), ('target_hidden', 'opposite_hidden')):
                        evidence[a], evidence[b] = evidence[b], evidence[a]
                    evidence['evidence'] = -evidence['evidence']
                current.condition = condition_tensor(evidence, state['true_t'], runtime.device)
                current.loop_count=1 if action=='conditional_budget1' else 2
                rng_restore(state['rng']); torch.cuda.synchronize(); started = time.perf_counter()
                with torch.no_grad(), full_model_context(runtime):
                    final,cost = replay_state(runtime, state, current, apply_update=action!='baseline')
                if action in ('baseline', 'conditional_zero') and not torch.equal(final, baseline):
                    raise ValueError('Full recomputation baseline/fresh untrained zero is not strictly identical')
                image = runtime.decode(final); torch.cuda.synchronize()
                cost['wall_time_s'] = time.perf_counter() - started
                cost['outer_shared_revisit_calls']=0 if action=='baseline' or current.outer is None else 2*current.loop_count
                cost['inner_shared_revisit_calls']=0 if action=='baseline' or current.inner is None else 2*current.loop_count
                cost['recurrent_loop_budget']=0 if action=='baseline' else current.loop_count
                # Full nominal deployed path includes the original index35 check velocity.
                full_calls = 49 if action == 'baseline' else 50
                cost.update(full_network_velocity_calls_nominal=full_calls,
                    full_outer_backbone_layer_calls_nominal=full_calls*28 + cost['outer_shared_revisit_calls'],
                    full_flow_head_layer_calls_nominal=full_calls*10 + cost['inner_shared_revisit_calls'],
                    full_semantic_encoder_layer_calls_nominal=full_calls*26)
                directory = args.output/action/sid; (directory/'samples').mkdir(parents=True)
                image_path = directory/'samples/0000.png'; image.save(image_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                final_path = directory/'final_latent.pt'
                torch.save(dict(latent=final.cpu(), state_id=sid, action=action), final_path)
                scale = baseline.square().mean().clamp_min(1e-5)
                displacement = (target-baseline).square().mean().clamp_min(1e-5)
                row = {k:state[k] for k in ('state_id', 'prompt_id', 'prompt', 'seed', 'split',
                    'flow_index', 'true_t', 'state_fingerprint', 'zero_intervention_test', 'runtime')}
                row.update(action=action, configuration_sha256=frozen,
                    image_path=str(image_path), final_latent=str(final_path),
                    image_path_sha256=file_hash(image_path), final_latent_sha256=file_hash(final_path), cost=cost,
                    cost_notes=dict(scope='remaining14 full velocities and decode, not measured end-to-end',
                        old_inner_prefix_reused=False, old_KV_reused=False,
                        MMU_checks_if_deployed=0 if action == 'baseline' else 2, external_detector_used=False,
                        evidence_precomputed=True, extra_check_velocity_nominal=0 if action == 'baseline' else 1,
                        FLOPs_measured=False),
                    trainfit_teacher_distance=float((final-target).square().mean()/scale),
                    trainfit_baseline_teacher_distance=float((baseline-target).square().mean()/scale),
                    trainfit_relative_displacement_error=None if sid not in pairs else float((final-target).square().mean()/displacement),
                    shuffled_source_state=source_id, checkpoint=str(args.checkpoint), arm=args.arm,study=args.study,
                    added_trainable_parameters=0 if action == 'baseline' else sum(q.numel() for q in current.trainable_parameters()),
                    FLOPs=None)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                print(json.dumps(dict(state_id=sid, action=action, arm=args.arm, status='complete')), flush=True)
    if op.core_hashes() != core_hash:
        raise ValueError('Frozen shared core changed during inference')
    write_json(args.output/'generation_status.json', dict(success=True, expected_images=36,
        optimizer_steps=0, split='train', arm=args.arm, val_images=0, controller_started=False))


if __name__ == '__main__':
    main()
