"""v68 six frozen TRAIN pairs, native controls with complete fresh flow replay."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
import torch
from candidate_provenance import file_hash
from native_teacher_interface import ACTIONS, physical, validate_identity, inferred_noise, rebridge, additive_parent
from runtime import Runtime, rng_restore, tensor_hash, write_json
from state_space_revisit import full_model_context


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def load_pair(pair, baselines):
    values = []
    for role, id_key in (('source','source_state_id'), ('parent','native_parent_state_id')):
        item = pair[role+'_state']; sid = pair[id_key]
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Original state file changed: '+sid)
        state = torch.load(item['path'], map_location='cpu', weights_only=False)
        final_path = pair[role+'_final_latent_path']
        if file_hash(final_path) != pair[role+'_final_latent_sha256']:
            raise ValueError('Canonical final latent changed: '+sid)
        payload = torch.load(final_path, map_location='cpu', weights_only=False)
        if payload['state_id'] != sid or payload['action'] != 'baseline':
            raise ValueError('Canonical final identity differs')
        final = physical(payload['latent'])
        if not torch.equal(final, state['baseline_final_latent']):
            raise ValueError('Capture and canonical baseline endpoints differ')
        if tensor_hash(state['latent']) != state['state_fingerprint']:
            raise ValueError('Original latent fingerprint differs')
        old = baselines[sid]
        if old['split'] != 'train' or old['prompt'] != state['prompt']:
            raise ValueError('Canonical TRAIN prompt differs')
        if file_hash(old['image_path']) != pair[role+'_baseline_image_sha256']:
            raise ValueError('Canonical original image differs')
        values.append((state, final))
    validate_identity(values[0][0], values[1][0], pair)
    return values


def replay_input(runtime, state, initial):
    """Direct physical input: avoid silently recomposing direct with FP32 delta."""
    physical(initial)
    grid = state['time_grid'].to(runtime.device)
    if not torch.equal(grid, runtime.grid) or state['CFG'] != runtime.cfg:
        raise ValueError('Runtime solver grid/CFG mismatch')
    x = initial.to(runtime.device).clone()
    options = runtime.inputs(state['prompt'])
    rng_restore(state['rng'])
    with full_model_context(runtime):
        for i in range(state['flow_index'], len(grid)-1):
            v = runtime.velocity(x, grid[i], options)
            x = x + (grid[i+1]-grid[i])*v
    return physical(x)


def pixel_difference(image, old_path):
    a, b = np.asarray(image).astype(np.int16), np.asarray(Image.open(old_path).convert('RGB')).astype(np.int16)
    if a.shape != b.shape: raise ValueError('Decoded baseline image shape differs')
    return int(np.abs(a-b).max())


def model_hashes(runtime):
    return {n: tensor_hash(v) for n,v in runtime.model.state_dict().items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--num-shards', type=int, required=True)
    args = p.parse_args()
    root = args.work/'stage3_grounded_repair'; out = args.experiment/('shard'+str(args.shard))
    out.mkdir()
    config = json.loads((args.experiment/'config.json').read_text())
    for name, digest in config['configuration']['source_sha256'].items():
        if file_hash(root/name) != digest: raise ValueError('Frozen source changed: '+name)
    if file_hash(root/'repair_train_targets_v36/all_actions.jsonl') != config['configuration']['canonical_actions_sha256']:
        raise ValueError('Canonical provenance changed')
    pairs = json.loads((args.experiment/'pairs.json').read_text())[args.shard::args.num_shards]
    baselines = {r['state_id']:r for r in read_rows(root/'repair_train_targets_v36/all_actions.jsonl') if r['action']=='baseline'}
    data = [(pair, load_pair(pair, baselines)) for pair in pairs]
    start = time.perf_counter(); runtime = Runtime(args.work)
    if runtime.config.transport.path_type != 'Linear' or runtime.config.transport.prediction != 'velocity':
        raise ValueError('Require official Linear velocity parameterization')
    before = model_hashes(runtime)
    write_json(out/'model_before_hashes.json', before)
    write_json(out/'generation_progress.json', dict(phase='running',completed_states=0,expected_states=len(pairs),model_load_s=time.perf_counter()-start))
    audits = []
    with (out/'actions_shard0.jsonl').open('w') as log, torch.no_grad():
        for pair, ((source, source_final), (parent, parent_final)) in data:
            started = time.perf_counter(); sid = source['state_id']
            original = source['latent'].to(runtime.device)
            parent_input = parent['latent'].to(runtime.device)
            # Strict interface checks happen BEFORE edited continuation/scoring.
            zero_final = replay_input(runtime, source, original)
            if not torch.equal(zero_final.cpu(), source_final):
                raise ValueError('Fresh zero endpoint not strictly equal: '+sid)
            zero_image = runtime.decode(zero_final)
            zero_pixel_diff = pixel_difference(zero_image, baselines[sid]['image_path'])
            if zero_pixel_diff != 0: raise ValueError('Fresh zero pixels differ: '+sid)
            direct_final = replay_input(runtime, parent, parent_input)
            if not torch.equal(direct_final.cpu(), parent_final):
                raise ValueError('Fresh parent direct endpoint not strictly equal: '+sid)
            direct_image = runtime.decode(direct_final)
            parent_pixel_diff = pixel_difference(direct_image, baselines[parent['state_id']]['image_path'])
            if parent_pixel_diff != 0: raise ValueError('Parent direct pixels differ: '+sid)
            # One fresh source velocity shared ONLY for constructing the two bridges.
            rng_restore(source['rng'])
            with full_model_context(runtime):
                velocity = runtime.velocity(original, runtime.grid[35], runtime.inputs(source['prompt']))
            noise = inferred_noise(original, velocity, runtime.grid[35])
            source_bridge = rebridge(noise, source_final.to(runtime.device), runtime.grid[35])
            native_bridge = rebridge(noise, parent_final.to(runtime.device), runtime.grid[35])
            additive, delta = additive_parent(original, parent_input)
            input_values = dict(source=original, source_velocity=velocity, inferred_noise=noise,
                parent_direct=parent_input, parent_delta=delta, parent_additive=additive,
                source_final_rebridge=source_bridge, native_final_rebridge=native_bridge,
                source_final=source_final.to(runtime.device),parent_final=parent_final.to(runtime.device))
            artifact_path = out/(sid+'_inputs.pt')
            torch.save(dict(state_id=sid,parent_state_id=parent['state_id'],split='train',flow_index=35,
                time_grid=source['time_grid'],t=source['time_grid'][35],
                **{k:v.cpu() for k,v in input_values.items()}),artifact_path)
            audit = dict(state_id=sid,parent_state_id=parent['state_id'],prompt_id=source['prompt_id'],
                split='train',source_state_sha256=pair['source_state']['sha256'],parent_state_sha256=pair['parent_state']['sha256'],
                source_final_sha256=pair['source_final_latent_sha256'],parent_final_sha256=pair['parent_final_latent_sha256'],
                source_mmu_evidence=float(source['evidence']['evidence']),
                source_mmu_detected_error=float(source['evidence']['evidence'])<0,
                source_target_margin=float(source['evidence']['target_margin']),
                source_opposite_margin=float(source['evidence']['opposite_margin']),
                zero_endpoint_exact=True,zero_pixel_max_abs_diff=zero_pixel_diff,
                parent_direct_endpoint_exact=True,parent_direct_pixel_max_abs_diff=parent_pixel_diff,
                source_fresh_velocity_equals_saved=torch.equal(velocity.cpu(),source['baseline_prediction']),
                source_fresh_velocity_max_abs_diff=float((velocity.cpu()-source['baseline_prediction']).abs().max()),
                parent_additive_input_exact=torch.equal(additive,parent_input),
                parent_additive_input_max_abs_diff=float((additive-parent_input).abs().max()),
                parent_additive_input_relative_l2=float((additive-parent_input).norm()/parent_input.norm()),
                artifact_path=str(artifact_path),artifact_sha256=file_hash(artifact_path),
                input_tensor_sha256={k:tensor_hash(v) for k,v in input_values.items()},
                shared_bridge_construction_velocity_calls=1,source_check_evidence_reused=True,
                optimizer_updates=0,learned_loop_success=False)
            zero_test=dict(passed=True,latent_exact=True,pixel_max_abs_diff=0,complete_fresh_velocity_calls=14)
            for action in ACTIONS:
                now=time.perf_counter(); directory=out/action/sid
                (directory/'samples').mkdir(parents=True)
                image_path=directory/'samples/0000.png'; final_path=directory/'final_latent.pt'
                if action=='baseline':
                    old=baselines[sid];shutil.copy2(old['image_path'],image_path);shutil.copy2(old['final_latent'],final_path)
                    final=source_final.to(runtime.device)
                else:
                    if action=='checked_zero': final,image=zero_final,zero_image
                    elif action=='native_parent_direct': final,image=direct_final,direct_image
                    else:
                        initial={'source_final_rebridge':source_bridge,'native_parent_additive':additive,
                                 'native_final_rebridge':native_bridge}[action]
                        final=replay_input(runtime,source,initial);image=runtime.decode(final)
                    image.save(image_path)
                    torch.save(dict(latent=final.cpu(),state_id=sid,action=action),final_path)
                if action=='native_parent_additive':
                    audit.update(parent_additive_endpoint_exact=torch.equal(final,direct_final),
                        parent_additive_endpoint_relative_l2=float((final-direct_final).norm()/direct_final.norm()))
                (directory/'metadata.jsonl').write_text(json.dumps(source['geneval_metadata'])+'\n')
                calls=0 if action=='baseline' else 14
                bridge=int(action in ('source_final_rebridge','native_final_rebridge'))
                row={k:source[k] for k in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t','state_fingerprint')}
                row.update(action=action,configuration_sha256=config['configuration_sha256'],
                    zero_intervention_test=zero_test,runtime=runtime.metadata(),
                    image_path=str(image_path),image_path_sha256=file_hash(image_path),
                    final_latent=str(final_path),final_latent_sha256=file_hash(final_path),
                    source_pair=pair,input_artifact=str(artifact_path),input_artifact_sha256=file_hash(artifact_path),
                    cost=dict(actual_remaining_velocity_calls=calls,actual_remaining_semantic_blocks=26*calls,
                        actual_remaining_qwen_blocks=28*calls,actual_remaining_flow_blocks=10*calls,
                        bridge_source_velocity_calls_attributed=bridge,bridge_source_velocity_calls_unique_per_state=0,
                        nominal_full_trajectory_velocity_calls=49+bridge,shared_revisit_calls=0,
                        actual_new_decodes=int(action!='baseline'),mmu_checks=0,FLOPs=None,
                        action_materialization_wall_s=time.perf_counter()-now),
                    cost_notes=dict(baseline_byte_alias=action=='baseline',old_KV_used=False,old_inner_prefix_used=False,
                        bridge_velocity_shared_between_two_bridges=True,unique_construction_calls_in_interface_audit=1,
                        prior_source_prefix_steps=35,prior_mmu_checks=2,
                        precomputed_zero_and_direct_wall_in_state_audit=True),
                    purpose='Offline TRAIN native teacher interface; not learned recurrence',
                    optimizer_updates=0,learned_recurrence=False)
                log.write(json.dumps(row)+'\n');log.flush()
            audit['complete_state_wall_s']=time.perf_counter()-started
            audits.append(audit);write_json(out/'interface_audit.json',audits)
            write_json(out/'generation_progress.json',dict(phase='running',completed_states=len(audits),expected_states=len(pairs)))
    after=model_hashes(runtime)
    if after!=before or any(p.grad is not None or p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Base parameters changed or have gradients')
    write_json(out/'generation_status.json',dict(success=True,expected_images=6*len(pairs),
        actual_new_images=5*len(pairs),baseline_aliases=len(pairs),
        base_hashes_exact=True,base_has_no_gradients=True,optimizer_updates=0,
        unique_new_velocity_calls=71*len(pairs),fresh_remaining_velocity_calls=70*len(pairs),
        reused_prefix_or_KV=False,wall_time_s=time.perf_counter()-start))


if __name__=='__main__':main()
