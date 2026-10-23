"""Pre-generation real-model FP32 conversion, routing and zero endpoint checks."""
import argparse
import json
from pathlib import Path
import time
import torch
from analytic_readout_operator import load_adapter, recurrence_mode, compare_analytical_prediction
from candidate_provenance import file_hash
from evidence_encoder import condition_tensor
from recurrent_training_protocol import build_operator
from runtime import Runtime, tensor_hash, write_json
from state_space_revisit import capture_features, full_model_context, replay_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    out = args.experiment
    def status(phase, **kw):
        write_json(out/'validation_status.json', dict(phase=phase, time=time.time(), optimizer_updates=0, **kw))
    try:
        manifest = json.loads((out/'preparation_manifest.json').read_text())
        root = args.work/'stage3_grounded_repair'
        for name, digest in manifest['source_sha256'].items():
            if file_hash(root/name) != digest:
                raise ValueError('Frozen v56 source changed: '+name)
        cohort = json.loads((out/'cohort.json').read_text())
        status('loading')
        runtime = Runtime(args.work, vae=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        base_hashes = {name: tensor_hash(p) for name, p in runtime.model.named_parameters()}
        operators, coefficients = {}, {}
        for mode, row in manifest['derivations'].items():
            torch.manual_seed(20261005); torch.cuda.manual_seed_all(20261005)
            op = build_operator(runtime, 'joint_ilsd')
            if file_hash(row['checkpoint']) != row['checkpoint_sha256'] or file_hash(row['source_probe']) != row['source_probe_sha256']:
                raise ValueError('Derived checkpoint or fixed coefficient bytes changed')
            payload = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
            load_adapter(op, payload['operator'])
            operators[mode] = op
            coefficients[mode] = torch.load(row['source_probe'], map_location='cpu', weights_only=False)['coefficient']
        results = []
        states = root/'states/repair_train_targets_v36'
        with torch.no_grad(), full_model_context(runtime):
            for sid in cohort['state_ids']:
                status('checking_actual_readout', state_id=sid)
                path = states/(sid+'.pt')
                if file_hash(path) != cohort['state_sha256'][sid]:
                    raise ValueError('Frozen diagnostic state changed')
                state = torch.load(path, map_location='cpu', weights_only=False)
                if state['split'] != 'train' or state['flow_index'] != 35:
                    raise ValueError('Expected TRAIN at index35')
                features, options = capture_features(runtime, state, operators['loop2'])
                for mode, op in operators.items():
                    op.condition = condition_tensor(state['evidence'], state['true_t'], runtime.device)
                    op.loop_count = 2
                    row = compare_analytical_prediction(op, runtime, features, options['modality_positions'],
                        state['latent'].to(runtime.device), coefficients[mode], mode)
                    expected = 4 if mode == 'loop2' else 0
                    if row['actual_shared_calls'] != dict(outer=expected, inner=expected):
                        raise ValueError('Actual recurrence route does not match protocol')
                    results.append(dict(state_id=sid, mode=mode, **row))
                write_json(out/'fp32_conversion_checks.json', results)
            # Dedicated zero operators; no trained/analytical readout reused as zero.
            sid = cohort['state_ids'][0]
            state = torch.load(states/(sid+'.pt'), map_location='cpu', weights_only=False)
            torch.manual_seed(20261005); torch.cuda.manual_seed_all(20261005)
            zero = build_operator(runtime, 'joint_ilsd')
            zero.condition = condition_tensor(state['evidence'], state['true_t'], runtime.device)
            baseline = state['baseline_final_latent'].to(runtime.device).float()
            status('checking_full14_zero_endpoints', state_id=sid)
            original, original_cost = replay_state(runtime, state, zero, apply_update=False)
            if not torch.equal(original, baseline):
                raise ValueError('Full14 baseline is not tensor-identical')
            zero_checks = []
            for mode in ('loop2', 'no_loop'):
                with recurrence_mode(zero, mode) as counts:
                    final, cost = replay_state(runtime, state, zero)
                if not torch.equal(final, baseline):
                    raise ValueError('Fresh zero full14 endpoint is not tensor-identical')
                zero_checks.append(dict(mode=mode, state_id=sid, strict_endpoint_equal=True,
                    actual_shared_calls=dict(counts), remaining_velocity_calls=cost['post_update_velocity_calls']))
        if base_hashes != {name: tensor_hash(p) for name, p in runtime.model.named_parameters()}:
            raise ValueError('Base weights changed')
        for mode, op in operators.items():
            if {key: tensor_hash(v) for key, v in op.adapter_state().items()} != manifest['derivations'][mode]['adapter_tensor_hashes']:
                raise ValueError('Derived adapter changed during validation')
        if any(p.grad is not None for p in runtime.model.parameters()):
            raise ValueError('Unexpected base gradient')
        write_json(out/'zero_endpoint_checks.json', zero_checks)
        status('complete', actual_fp32_checks=len(results), zero_endpoint_checks=zero_checks,
            full_base_tensor_hashes_unchanged=True, base_gradients_absent=True,
            inference_teacher_or_mask_read=False, new_generation_images=0)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
