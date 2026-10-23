"""Read-only same-head zero-loop counterfactuals; no new image scores or fitting."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import torch
from analytic_readout_operator import load_adapter, recurrence_mode
from candidate_provenance import file_hash, freeze_config
from evidence_encoder import condition_tensor
from launch_grounding_diagnostic import GPUS
from recurrent_training_protocol import build_operator
from runtime import Runtime, tensor_hash, write_json
from state_space_revisit import capture_features, full_model_context, replay_state

WORK = Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT = WORK/'stage3_grounded_repair'
SOURCE = ROOT/'analytic_readout_trainfit_v56'
OUT = ROOT/'analytic_core_contribution_v57'


def main():
    OUT.mkdir(exist_ok=False)
    def status(phase, **kw):
        write_json(OUT/'pipeline_status.json', dict(phase=phase, time=time.time(), optimizer_updates=0,
            new_scoring_images=0, analytical_fits=0, controller_started=False, **kw))
    try:
        check = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if check[0].strip() != GPUS[0] or int(check[1].strip()) >= 2000 or os.environ['CUDA_VISIBLE_DEVICES'] != GPUS[0]:
            raise ValueError('Require available GPU0 and original UUID; no other task interrupted')
        if json.loads((SOURCE/'pipeline_status.json').read_text())['phase'] != 'complete':
            raise ValueError('Require completed fixed v56 endpoints')
        prep = json.loads((SOURCE/'preparation_manifest.json').read_text())
        cohort = json.loads((SOURCE/'cohort.json').read_text())
        targets = {row['state_id']: row for row in map(json.loads,
            (ROOT/'teacher_additive_reachability_v50/all_actions.jsonl').read_text().splitlines())
            if row['action'] == 'additive_teacher'}
        names = ('audit_analytic_core_contribution.py', 'analytic_readout_operator.py',
            'trainable_recurrent_state.py', 'recurrent_training_protocol.py', 'state_space_revisit.py',
            'runtime.py', 'evidence_encoder.py', 'candidate_provenance.py')
        config = dict(source_sha256={n: file_hash(ROOT/n) for n in names},
            derived_checkpoint_sha256={k: v['checkpoint_sha256'] for k, v in prep['derivations'].items()},
            cohort_sha256=file_hash(SOURCE/'cohort.json'), state_sha256=cohort['state_sha256'],
            fixed_correct_evidence=True, same_head_counterfactual='Only actual shared-core route changes',
            budgets='original fixed loop2 versus actual no shared calls',
            original_endpoints=12, counterfactual_endpoints=12, new_scoring_images=0,
            caution='Dependence of a fitted head on its input distribution is not superiority over a refitted no-loop head',
            FLOPs=None)
        freeze_config(OUT/'config.json', config)
        (OUT/'source').mkdir()
        for name in names:
            shutil.copy2(ROOT/name, OUT/'source'/name)
        status('loading')
        runtime = Runtime(WORK, vae=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        operators = {}; reference_rows = {}
        for mode, row in prep['derivations'].items():
            if file_hash(row['checkpoint']) != row['checkpoint_sha256']:
                raise ValueError('Frozen v56 derived checkpoint changed')
            torch.manual_seed(20261005); torch.cuda.manual_seed_all(20261005)
            op = build_operator(runtime, 'joint_ilsd')
            payload = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
            load_adapter(op, payload['operator']); operators[mode] = op
            reference_rows[mode] = {row['state_id']: row for row in map(json.loads,
                (SOURCE/mode/'all_actions.jsonl').read_text().splitlines()) if row['action']=='conditional_revisit'}
        core_hashes = operators['loop2'].core_hashes()
        results = []
        with torch.no_grad(), full_model_context(runtime):
            for sid in cohort['state_ids']:
                status('same_head_counterfactual', state_id=sid)
                path = ROOT/'states/repair_train_targets_v36'/(sid+'.pt')
                if file_hash(path) != cohort['state_sha256'][sid]:
                    raise ValueError('Captured TRAIN state bytes changed')
                state = torch.load(path, map_location='cpu', weights_only=False)
                if state['split'] != 'train' or state['flow_index'] != 35:
                    raise ValueError('Expected TRAIN at index35')
                original = state['latent'].to(runtime.device)
                baseline = state['baseline_final_latent'].to(runtime.device).float()
                features, options = capture_features(runtime, state, operators['loop2'])
                outputs = []
                for head, op in operators.items():
                    op.condition = condition_tensor(state['evidence'], state['true_t'], runtime.device)
                    op.loop_count = 2
                    own = head; counter = 'no_loop' if own == 'loop2' else 'loop2'
                    values = {}
                    for route in (own, counter):
                        with recurrence_mode(op, route) as calls:
                            delta = op.latent_delta(runtime, features, options['modality_positions'], original)
                        with recurrence_mode(op, route) as rollout_calls:
                            final, cost = replay_state(runtime, state, op, features=features)
                        expected = 4 if route == 'loop2' else 0
                        if calls != dict(outer=expected, inner=expected) or rollout_calls != calls:
                            raise ValueError('Actual core route changed unexpectedly')
                        if route == own:
                            reference = reference_rows[head][sid]
                            if file_hash(reference['final_latent']) != reference['final_latent_sha256']:
                                raise ValueError('v56 reference bytes changed')
                            previous = torch.load(reference['final_latent'], map_location=runtime.device, weights_only=False)['latent']
                            if not torch.equal(final, previous):
                                raise ValueError('v56 exact terminal reproduction failed')
                        values[route] = dict(delta=delta, final=final)
                    # Targets are post-inference analysis only, after both routes are fixed/executed.
                    preserve = sid not in cohort['teacher_state_ids']
                    target = baseline
                    if not preserve:
                        teacher = targets[sid]
                        if file_hash(teacher['final_latent']) != teacher['final_latent_sha256']:
                            raise ValueError('Posthoc TRAIN target bytes changed')
                        target = torch.load(teacher['final_latent'], map_location=runtime.device, weights_only=False)['latent'].float()
                    scale = (baseline.square().mean() if preserve else (target-baseline).square().mean()).clamp_min(1e-5)
                    a, b = values[own], values[counter]
                    saved = OUT/(head+'__'+sid+'.pt')
                    torch.save(dict(state_id=sid, head=head, source_route=own, counterfactual_route=counter,
                        delta_original=a['delta'].cpu(), delta_counterfactual=b['delta'].cpu(),
                        final_original=a['final'].cpu(), final_counterfactual=b['final'].cpu()), saved)
                    da, db = a['delta'].float().flatten(), b['delta'].float().flatten()
                    outputs.append(dict(state_id=sid, head=head, preservation=preserve,
                        original_route=own, counterfactual_route=counter, original_endpoint_strictly_reproduced=True,
                        original_endpoint_target_relative_mse=float((a['final']-target).square().mean()/scale),
                        counterfactual_endpoint_target_relative_mse=float((b['final']-target).square().mean()/scale),
                        delta_counterfactual_change_relative_rms=float((db-da).square().mean().sqrt()/da.square().mean().sqrt().clamp_min(1e-12)),
                        delta_direction_cosine=float(torch.nn.functional.cosine_similarity(da, db, dim=0)),
                        original_shared_calls_per_branch=4 if own=='loop2' else 0,
                        counterfactual_shared_calls_per_branch=4 if counter=='loop2' else 0,
                        counterfactual_all14_velocities_recomputed=True,
                        tensor_path=str(saved), tensor_sha256=file_hash(saved)))
                results.extend(outputs)
                write_json(OUT/'results.json', results)
        for head, op in operators.items():
            if op.core_hashes() != core_hashes:
                raise ValueError('Base shared cores changed')
            if {k: tensor_hash(v) for k, v in op.adapter_state().items()} != prep['derivations'][head]['adapter_tensor_hashes']:
                raise ValueError('Read-only derived head changed')
        if any(p.grad is not None for p in runtime.model.parameters()):
            raise ValueError('Unexpected base gradients')
        status('complete', strict_reference_endpoints=12, counterfactual_endpoints=12,
            base_core_and_adapter_hashes_unchanged=True, base_and_adapter_no_grad=True,
            no_new_semantic_or_quality_measurement=True)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
