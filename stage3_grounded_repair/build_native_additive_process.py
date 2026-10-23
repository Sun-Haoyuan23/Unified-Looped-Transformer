"""v69 audit actual FP32 residual reachability of new native TRAIN teachers."""
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import torch
from candidate_provenance import file_hash
from build_native_teacher_candidates import load_pair,read_rows,replay_input,model_hashes
from native_teacher_interface import additive_parent
from runtime import Runtime,tensor_hash,rng_restore,write_json
from state_space_revisit import capture_features,full_model_context

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT=WORK/'stage3_grounded_repair';OUT=ROOT/'native_teacher_additive_v69'


def trajectory(runtime,state,initial):
    options=runtime.inputs(state['prompt']);rng_restore(state['rng']);x=initial.clone()
    states=[x.cpu()];velocities=[]
    with full_model_context(runtime):
        for i in range(35,len(runtime.grid)-1):
            v=runtime.velocity(x,runtime.grid[i],options);velocities.append(v.cpu())
            x=x+(runtime.grid[i+1]-runtime.grid[i])*v;states.append(x.cpu())
    return x,states,velocities


def main():
    generated=OUT/'generated';generated.mkdir()
    config=json.loads((OUT/'config.json').read_text())
    for n,h in config['configuration']['source_sha256'].items():
        if file_hash(ROOT/n)!=h:raise ValueError('Frozen v69 source changed: '+n)
    v68=ROOT/'native_teacher_interface_v68'
    candidates=json.loads((v68/'qualified_native_teachers.json').read_text())['candidates']
    candidates=sorted(candidates,key=lambda r:r['state_id'])
    baselines={r['state_id']:r for r in read_rows(ROOT/'repair_train_targets_v36/all_actions.jsonl') if r['action']=='baseline'}
    runtime=Runtime(WORK);before=model_hashes(runtime)
    write_json(generated/'model_before_hashes.json',before)
    marker=SimpleNamespace(outer=True,inner=True,arm='joint')
    audits=[];start=time.perf_counter()
    with torch.no_grad(),(generated/'actions_shard0.jsonl').open('w') as log:
        for candidate in candidates:
            now=time.perf_counter();sid=candidate['state_id'];pair=candidate['source_pair']
            state=load_pair(pair,baselines)[0][0]
            row=candidate['action_record']
            if file_hash(row['input_artifact'])!=row['input_artifact_sha256'] or file_hash(row['final_latent'])!=row['final_latent_sha256']:
                raise ValueError('Qualified v68 teacher artifact changed')
            payload=torch.load(row['input_artifact'],map_location='cpu',weights_only=False)
            if payload['state_id']!=sid or payload['split']!='train' or not torch.equal(payload['source'],state['latent']):
                raise ValueError('Teacher input identity differs')
            teacher_input=payload['native_final_rebridge'].to(runtime.device)
            if tensor_hash(teacher_input)!=candidate['interface_audit']['input_tensor_sha256']['native_final_rebridge']:
                raise ValueError('Native bridge input fingerprint changed')
            target_final=torch.load(row['final_latent'],map_location=runtime.device,weights_only=False)['latent']
            direct_final=replay_input(runtime,state,teacher_input)
            if not torch.equal(direct_final,target_final):raise ValueError('v68 teacher full endpoint failed strict reproduction')
            original=state['latent'].to(runtime.device)
            additive,delta=additive_parent(original,teacher_input)
            original_features,options=capture_features(runtime,state,marker)
            target_state=dict(state,latent=additive.cpu(),state_fingerprint=tensor_hash(additive))
            target_features,_=capture_features(runtime,target_state,marker)
            additive_final,states,velocities=trajectory(runtime,state,additive)
            artifact=generated/(sid+'_process.pt')
            torch.save(dict(state_id=sid,split='train',flow_index=35,source_state_fingerprint=state['state_fingerprint'],
                original_input=original.cpu(),direct_native_input=teacher_input.cpu(),target_input=additive.cpu(),
                physical_delta=delta.cpu(),original_features={n:original_features[n][0].cpu() for n in ('outer','inner')},
                target_features={n:target_features[n][0].cpu() for n in ('outer','inner')},
                modality_positions=options['modality_positions'].cpu(),time_grid=state['time_grid'],
                trajectory_states=states,trajectory_velocities=velocities,target_final=additive_final.cpu(),
                original_native_final=target_final.cpu(),source_pair=pair,
                configuration_sha256=config['configuration_sha256']),artifact)
            audit=dict(state_id=sid,split='train',flow_index=35,direct_native_endpoint_exact=True,
                additive_input_exact=torch.equal(additive,teacher_input),
                additive_input_max_abs_diff=float((additive-teacher_input).abs().max()),
                additive_endpoint_exact=torch.equal(additive_final,target_final),
                additive_endpoint_relative_l2=float((additive_final-target_final).norm()/target_final.norm()),
                physical_delta_rms=float(delta.square().mean().sqrt()),
                physical_delta_relative_rms=float(delta.norm()/original.norm()),
                feature_target_relative_rms={n:float((target_features[n][0].float()-original_features[n][0].float()).norm()/original_features[n][0].float().norm()) for n in ('outer','inner')},
                source_mmu_evidence=float(state['evidence']['evidence']),
                artifact_path=str(artifact),artifact_sha256=file_hash(artifact),
                source_state_sha256=pair['source_state']['sha256'],
                tensor_sha256=dict(original=tensor_hash(original),target_input=tensor_hash(additive),
                    physical_delta=tensor_hash(delta),target_final=tensor_hash(additive_final)),
                direct_reproduction_velocity_calls=14,additive_remaining_velocity_calls=14,feature_capture_velocity_calls=2,
                complete_state_wall_s=time.perf_counter()-now,optimizer_updates=0,FLOPs=None)
            for action in ('baseline','native_additive_teacher'):
                directory=generated/action/sid;(directory/'samples').mkdir(parents=True)
                image=directory/'samples/0000.png';final=directory/'final_latent.pt'
                old=baselines[sid]
                if action=='baseline':shutil.copy2(old['image_path'],image);shutil.copy2(old['final_latent'],final)
                else:
                    runtime.decode(additive_final).save(image)
                    torch.save(dict(latent=additive_final.cpu(),state_id=sid,action=action),final)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                result={k:state[k] for k in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t','state_fingerprint')}
                result.update(action=action,configuration_sha256=config['configuration_sha256'],
                    zero_intervention_test=dict(passed=True,latent_exact=True,pixel_max_abs_diff=0,
                        proof='v68 complete fresh zero with identical frozen runtime/source state',
                        proof_manifest_sha256=config['configuration']['v68_manifest_sha256']),
                    image_path=str(image),image_path_sha256=file_hash(image),
                    final_latent=str(final),final_latent_sha256=file_hash(final),
                    input_artifact=str(artifact),input_artifact_sha256=file_hash(artifact),
                    cost=dict(actual_remaining_velocity_calls=0 if action=='baseline' else 14,
                        actual_new_decodes=int(action!='baseline'),FLOPs=None),
                    cost_notes=dict(offline_teacher=True,base_trajectory_full_recomputed=True,
                        old_KV_used=False,old_inner_prefix_used=False,diagnostic_and_feature_calls_in_audit=True),
                    optimizer_updates=0,learned_recurrence=False)
                log.write(json.dumps(result)+'\n');log.flush()
            audits.append(audit);write_json(OUT/'process_manifest.json',audits)
    if before!=model_hashes(runtime) or any(p.grad is not None or p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Base changed or has gradients')
    write_json(OUT/'generation_status.json',dict(success=True,expected_images=6,actual_new_images=3,baseline_aliases=3,
        optimizer_updates=0,base_hashes_exact=True,base_has_no_gradients=True,unique_new_velocity_calls=90,
        wall_time_s=time.perf_counter()-start))


if __name__=='__main__':main()
