"""Reconstruct qualified TRAIN teacher states, verify exact endpoints, export process.

This is a read-only supervision-interface audit, not learned-loop generation.
No terminal scores select inference actions. The corpus/cohort are frozen TRAIN.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from candidate_provenance import file_hash, freeze_config
from correlated_scene_mirror import reflected_input
from fm_rebridge import linear_rebridge
from recurrent_training_protocol import build_operator
from runtime import Runtime, rng_restore, tensor_hash, write_json
from state_space_revisit import capture_features, full_model_context
from training_pairs import selected_pairs, validate_training_pairs


def cpu_features(features):
    return {name: features[name][0].detach().cpu() for name in ('outer', 'inner')}


def trajectory(runtime, state, initial):
    options = runtime.inputs(state['prompt'])
    rng_restore(state['rng'])
    x = initial.clone()
    states, velocities = [x.detach().cpu()], []
    for index in range(state['flow_index'], len(runtime.grid)-1):
        v = runtime.velocity(x, runtime.grid[index], options)
        velocities.append(v.detach().cpu())
        x = x + (runtime.grid[index+1]-runtime.grid[index])*v
        states.append(x.detach().cpu())
    return x, states, velocities


def main():
    p = argparse.ArgumentParser()
    for name in ('work','states','teacher-root','cohort','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir()
    def status(phase,**kw):
        write_json(args.output/'pipeline_status.json',dict(phase=phase, time=time.time(),
            optimizer_updates=0,new_scored_images=0,controller_started=False,**kw))
    try:
        root = args.work/'stage3_grounded_repair'
        records = [json.loads(s) for s in (args.teacher_root/'all_actions.jsonl').read_text().splitlines()]
        report = json.loads((args.teacher_root/'paired_report.json').read_text())
        gate = validate_training_pairs(report, records)
        teachers = selected_pairs(report, records)
        cohort = json.loads(args.cohort.read_text())
        if (gate['selected_teacher_states']!=3 or gate['preservation_states']!=35 or
                cohort['split']!='train' or len(cohort['state_ids'])!=6 or
                cohort['teacher_state_ids']!=sorted(teachers) or
                cohort['teacher_report_sha256']!=file_hash(args.teacher_root/'paired_report.json')):
            raise ValueError('Frozen TRAIN gate/cohort changed')
        baselines = {q['state_id']:q for q in records if q['action']=='baseline'}
        if len(baselines)!=112 or any(q['split']!='train' or q['flow_index']!=35 for q in baselines.values()):
            raise ValueError('Canonical TRAIN source changed')
        # These are the two original successful interfaces, never native-latent
        # reflection or a new geometry/strength/time/seed search.
        interfaces = {
            'checked_inferred_correlated_target': ('scene_mirror_geometry_train10_v18','reflection'),
            'checked_inferred_target': ('minimal_geometry_expansion_v34','minimum'),
        }
        geometry = {}
        for sid,row in teachers.items():
            directory,kind = interfaces[row['original_source_action']]
            manifest_path = root/directory/'teacher_manifest.json'
            choices = [q for q in json.loads(manifest_path.read_text())['rows'] if q['state_id']==sid]
            if len(choices)!=1 or choices[0]['teacher_sha256']!=row['teacher_sha256']:
                raise ValueError('Original geometry teacher identity changed')
            item = choices[0]
            if file_hash(item['teacher_path'])!=item['teacher_sha256'] or item['state_fingerprint']!=row['state_fingerprint']:
                raise ValueError('Geometry bytes or source fingerprint changed')
            geometry[sid]=dict(item,kind=kind,manifest=str(manifest_path),manifest_sha256=file_hash(manifest_path))
        source=Path(__file__).parent
        names=('export_teacher_process.py','runtime.py','state_space_revisit.py','trainable_recurrent_state.py',
            'recurrent_training_protocol.py','fm_rebridge.py','correlated_scene_mirror.py',
            'candidate_provenance.py','training_pairs.py','evidence_encoder.py')
        frozen=freeze_config(args.output/'config.json',dict(
            source_sha256={n:file_hash(source/n) for n in names},
            state_sha256=cohort['state_sha256'],cohort_sha256=file_hash(args.cohort),
            teacher_report_sha256=file_hash(args.teacher_root/'paired_report.json'),
            teacher_records_sha256=file_hash(args.teacher_root/'all_actions.jsonl'),
            geometry=geometry,training_gate=gate,split='train',index=35,
            states=cohort['state_ids'],feature_layers=dict(outer=27,inner=9),
            arithmetic='original float(t) reflection; original grid tensor t minimum bridge',
            original_source_interface_only=True,optimizer_updates=0,scored_images=0,
            purpose='Process targets and additive-interface reachability; not shared-loop success'))
        (args.output/'source').mkdir()
        for name in names:
            (args.output/'source'/name).write_bytes((source/name).read_bytes())
        status('loading')
        runtime=Runtime(args.work,vae=False)
        torch.manual_seed(20261005)
        op=build_operator(runtime,'joint_ilsd')
        core_hash=op.core_hashes()
        rows=[]
        with torch.no_grad(),full_model_context(runtime), (args.output/'process_manifest.jsonl').open('w') as log:
            for sid in cohort['state_ids']:
                path=args.states/(sid+'.pt')
                if file_hash(path)!=cohort['state_sha256'][sid]:
                    raise ValueError('Frozen capture bytes changed')
                state=torch.load(path,map_location='cpu',weights_only=False)
                if (state['split']!='train' or state['flow_index']!=35 or state['state_id']!=sid or
                        tensor_hash(state['latent'])!=state['state_fingerprint'] or
                        state['state_fingerprint']!=baselines[sid]['state_fingerprint'] or
                        not state['zero_intervention_test']['passed']):
                    raise ValueError('TRAIN state identity changed')
                status('exporting',state_id=sid,completed=len(rows))
                original=state['latent'].to(runtime.device).float()
                target_input=original
                bridge_exact=True
                if sid in teachers:
                    row=teachers[sid]
                    item=geometry[sid]
                    geometry_payload=torch.load(item['teacher_path'],map_location='cpu',weights_only=False)
                    if geometry_payload['state_fingerprint']!=state['state_fingerprint']:
                        raise ValueError('Geometry payload from another state')
                    desired=geometry_payload['latents']['target'].to(runtime.device)
                    v=state['baseline_prediction'].to(runtime.device).float()
                    t=runtime.grid[state['flow_index']]
                    if float(t)!=state['true_t'] or not torch.equal(state['time_grid'].to(runtime.device),runtime.grid):
                        raise ValueError('Actual original solver time changed')
                    if item['kind']=='reflection':
                        target_input=reflected_input(original,v,runtime.noise(state['seed']).float(),desired,
                            float(t),'inferred_correlated_target')
                        expected=row['bridge']['reflected_input_sha256']
                    else:
                        target_input=linear_rebridge(original-t*v,desired,t)
                        expected=row['bridge']['rebridged_input_sha256']
                    bridge_exact=tensor_hash(target_input)==expected
                    if not bridge_exact:
                        raise ValueError('Reconstructed original bridge input not hash-identical: '+sid)
                original_features,options=capture_features(runtime,state,op)
                target_state=dict(state,latent=target_input.cpu(),state_fingerprint=tensor_hash(target_input))
                target_features,_=capture_features(runtime,target_state,op) if sid in teachers else (original_features,options)
                final,states,velocities=trajectory(runtime,state,target_input)
                # Canonical successful final is a TRAIN supervision target; it
                # never feeds the reconstruction or a deployment action.
                expected_final=state['baseline_final_latent'].to(runtime.device)
                if sid in teachers:
                    row=teachers[sid]
                    if file_hash(row['final_latent'])!=row['final_latent_sha256']:
                        raise ValueError('Canonical teacher bytes changed')
                    target_payload=torch.load(row['final_latent'],map_location=runtime.device,weights_only=False)
                    if target_payload['state_id']!=sid or target_payload['action']!=row['action']:
                        raise ValueError('Canonical teacher identity changed')
                    expected_final=target_payload['latent']
                if not torch.equal(final,expected_final):
                    raise ValueError('Reconstructed process endpoint not strictly identical: '+sid)
                delta=target_input-original
                additive=original+delta
                additive_input_exact=torch.equal(additive,target_input)
                additive_final=final
                additive_velocity_calls=0
                if not additive_input_exact:
                    additive_final,_,_=trajectory(runtime,state,additive)
                    additive_velocity_calls=14
                artifact=args.output/(sid+'.pt')
                torch.save(dict(state_id=sid,split='train',flow_index=35,
                    source_state_fingerprint=state['state_fingerprint'],
                    target_input=target_input.cpu(),physical_delta=delta.cpu(),
                    original_features=cpu_features(original_features),target_features=cpu_features(target_features),
                    modality_positions=options['modality_positions'].cpu(),
                    time_grid=runtime.grid.cpu(),trajectory_states=states,trajectory_velocities=velocities,
                    target_final=final.cpu(),additive_final=additive_final.cpu(),
                    teacher=sid in teachers,configuration_sha256=frozen),artifact)
                result=dict(state_id=sid,split='train',flow_index=35,
                    cohort='teacher' if sid in teachers else 'preservation',
                    source_state_fingerprint=state['state_fingerprint'],
                    bridge_input_hash_exact=bridge_exact,target_input_sha256=tensor_hash(target_input),
                    terminal_exact=True,target_terminal_sha256=tensor_hash(final),
                    additive_input_exact=additive_input_exact,
                    additive_input_max_abs_diff=float((additive-target_input).abs().max()),
                    additive_terminal_exact=torch.equal(additive_final,final),
                    additive_terminal_max_abs_diff=float((additive_final-final).abs().max()),
                    additive_terminal_relative_l2=float((additive_final-final).norm()/final.norm().clamp_min(1e-20)),
                    physical_delta_rms=float(delta.square().mean().sqrt()),
                    feature_target_relative_rms={name:float((target_features[name][0].float()-original_features[name][0].float()).square().mean().sqrt()/original_features[name][0].float().square().mean().sqrt().clamp_min(1e-20)) for name in ('outer','inner')},
                    artifact_path=str(artifact),artifact_sha256=file_hash(artifact),
                    original_feature_capture_velocity_calls=1,target_feature_capture_velocity_calls=int(sid in teachers),
                    direct_remaining_full_velocity_calls=14,additive_remaining_full_velocity_calls=additive_velocity_calls,
                    optimizer_updates=0,new_scored_images=0,FLOPs=None)
                log.write(json.dumps(result)+'\n');log.flush();rows.append(result)
                print(json.dumps(result),flush=True)
        if core_hash!=op.core_hashes() or any(q.grad is not None for q in runtime.model.parameters()):
            raise ValueError('Read-only process export changed frozen cores or created base gradients')
        status('complete',states=len(rows),teacher_states=3,preservation_states=3,
            all_original_bridge_inputs_exact=True,all_direct_endpoints_exact=True,
            additive_exact_states=sum(q['additive_terminal_exact'] for q in rows),
            core_hash_unchanged=True,base_gradients=False)
    except Exception as error:
        status('failed',error=str(error))
        raise


if __name__=='__main__':main()
