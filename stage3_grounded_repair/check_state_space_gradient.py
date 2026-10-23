"""Real-model zero, latent-update and full-terminal-gradient proof; no optimizer."""
from pathlib import Path
import shutil
import time
import json
import torch
from runtime import Runtime, rng_restore, tensor_hash, write_json
from state_space_revisit import make_operator, full_model_context, capture_features, replay_state
from evidence_encoder import condition_tensor
from training_pairs import selected_pairs, validate_training_pairs
from candidate_provenance import file_hash

w = Path('/home/shy_23/iterative_visual_stage1_20261003')
r = w/'stage3_grounded_repair'; out = r/'state_space_gradient_v44'; out.mkdir()


def update(phase, **kw):
    write_json(out/'pipeline_status.json', dict(phase=phase, time=time.time(), optimizer_updates=0, **kw))


try:
    names = ('check_state_space_gradient.py', 'state_space_revisit.py', 'runtime.py', 'evidence_encoder.py', 'training_pairs.py')
    (out/'source').mkdir()
    for name in names: shutil.copy2(r/name, out/'source'/name)
    teacher = r/'repair_train_targets_v36'
    report = json.loads((teacher/'paired_report.json').read_text())
    records = list(map(json.loads, (teacher/'all_actions.jsonl').read_text().splitlines()))
    gate = validate_training_pairs(report, records); pairs = selected_pairs(report, records)
    if gate['selected_teacher_states'] != 3 or gate['preservation_states'] != 35:
        raise ValueError('Strict TRAIN gate changed')
    sid = 'train_extra_00021_seed101_i35'
    path = r/'states/repair_train_targets_v36'/f'{sid}.pt'
    update('loading'); rt = Runtime(w, vae=False); base = list(rt.model.parameters())
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['split'] != 'train' or not state['zero_intervention_test']['passed']:
        raise ValueError('Invalid TRAIN state')
    target_row = pairs[sid]
    if file_hash(target_row['final_latent']) != target_row['final_latent_sha256']:
        raise ValueError('Teacher provenance changed')
    target = torch.load(target_row['final_latent'], map_location=rt.device, weights_only=False)['latent'].float()
    baseline = state['baseline_final_latent'].to(rt.device).float()
    config = dict(source_sha256={n:file_hash(r/n) for n in names}, state_sha256=file_hash(path),
                  training_gate=gate, action='one state update at fixed index35 then all14 fresh original velocities',
                  probe_readout_std=.001, probe_optimizer_updates=0)
    write_json(out/'config.json', config)
    zero_arms, parameter_counts = [], {}
    for arm in ('head_only', 'outer', 'inner', 'joint'):
        torch.manual_seed(20261005); op = make_operator(rt, arm)
        op.condition = condition_tensor(state['evidence'], state['true_t'], rt.device)
        hashes = op.core_hashes(); update('zero_check', arm=arm)
        with full_model_context(rt), torch.no_grad():
            features, options = capture_features(rt, state, op)
            delta = op.latent_delta(rt, features, options['modality_positions'], state['latent'].to(rt.device))
            if not torch.equal(delta, torch.zeros_like(delta)):
                raise ValueError('Fresh physical delta is not exactly zero')
            final, cost = replay_state(rt, state, op, features=features)
        if not torch.equal(final, baseline) or hashes != op.core_hashes():
            raise ValueError('Zero endpoint/core mismatch for ' + arm)
        parameter_counts[arm] = sum(p.numel() for p in op.trainable_parameters()); zero_arms.append(arm)
        if arm != 'joint': del op, features; torch.cuda.empty_cache()
    with full_model_context(rt):
        update('zero_head_training_gradient')
        final, cost = replay_state(rt, state, op, features, checkpoint_rollout=True)
        (final-target).square().mean().backward()
        initial_head = sum(float(b.readout.weight.grad.abs().sum()) for b in (op.outer, op.inner))
        if initial_head <= 0: raise ValueError('Zero-initialized readout cannot start training')
        op.zero_grad(set_to_none=True)
        with torch.no_grad():
            for branch in (op.outer, op.inner): branch.readout.weight.normal_(std=.001)
        update('terminal_gradient'); start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        final, cost = replay_state(rt, state, op, features, checkpoint_rollout=True)
        (final-target).square().mean().backward()
        memory = sum(float(p.grad.abs().sum()) for p in op.encoder.parameters() if p.grad is not None)
        feedback = {name:float(branch.feedback.weight.grad.abs().sum()) for name,branch in (('outer',op.outer),('inner',op.inner))}
        grads = [p.grad for p in op.trainable_parameters() if p.grad is not None]
        if memory <= 0 or min(feedback.values()) <= 0 or any(not bool(torch.isfinite(g).all()) for g in grads):
            raise ValueError('Missing/nonfinite evidence or shared-revisit feedback gradient')
        if any(p.grad is not None for p in base): raise ValueError('Frozen base gradient')
    if hashes != op.core_hashes(): raise ValueError('Shared core changed')
    update('passed', zero_intervention_latent_exact=True, zero_arms=zero_arms, shared_cores_exact=True,
           base_parameter_gradients=False, shared_evidence_gradient_abs_sum=memory, feedback_gradient_abs_sum=feedback,
           initial_readout_gradient_abs_sum=initial_head, trainable_parameters=parameter_counts,
           adapter_gradient_abs_sum=sum(float(g.abs().sum()) for g in grads), time_s=time.perf_counter()-start,
           peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3, cost=cost,
           interpretation='Trainability/identity only; no optimizer or semantic/quality gain')
except Exception as error:
    update('failed', error=str(error)); raise
