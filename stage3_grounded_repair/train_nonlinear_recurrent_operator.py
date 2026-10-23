"""Fixed210 nonlinear recurrent pilot; no automatic quality/val expansion."""
import argparse,json,shutil,time
import torch
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from nonlinear_recurrent_repair import ARMS,balanced_schedule,normalized_losses
from nonlinear_recurrent_protocol import ROOT,WORK,frozen_inputs,make_repair,training_target,tensor_fingerprints,gradient_groups
from runtime import Runtime,rng_restore,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta

NAMES=('train_nonlinear_recurrent_operator.py','run_nonlinear_recurrent_training.py',
       'nonlinear_recurrent_repair.py','nonlinear_recurrent_protocol.py','nonlinear_token_readout.py',
       'process_training_targets.py','recurrent_training_protocol.py','trainable_recurrent_state.py',
       'state_space_revisit.py','runtime.py','candidate_provenance.py','evidence_encoder.py',
       'training_pairs.py','analytic_readout_operator.py')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True)
    args=parser.parse_args();out=ROOT/'nonlinear_recurrent_train_v64'/args.arm;out.mkdir(exist_ok=False)
    def status(phase,**kw):write_json(out/'worker_status.json',dict(phase=phase,arm=args.arm,time=time.time(),**kw))
    try:
        proof_dir=ROOT/'nonlinear_recurrent_gradient_v63';proof=json.loads((proof_dir/'pipeline_status.json').read_text())
        if proof['phase']!='passed' or proof['optimizer_updates']!=0 or proof['base_gradients'] or not proof['base_hash_unchanged']:
            raise ValueError('Require real nonlinear recurrent endpoint gradient/zero proof')
        for name,digest in json.loads((proof_dir/'config.json').read_text())['configuration']['source_sha256'].items():
            if file_hash(ROOT/name)!=digest:raise ValueError('Proven interface source changed: '+name)
        corpus,checkpoint,heads,labels,sources=frozen_inputs()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('loading',optimizer_updates=0);runtime=Runtime(WORK,vae=False)
        repair=make_repair(runtime,args.arm,checkpoint,heads)
        initial=list(tensor_fingerprints(repair))
        if initial!=proof['initial_tensor_hashes'][args.arm]:raise ValueError('Training differs from proven fixed initialization')
        schedule=balanced_schedule(corpus.teachers,corpus.correct)
        config=dict(arm=args.arm,steps=210,optimizer='AdamW',lr=1e-4,weight_decay=0.,gradient_clip=1.,
            schedule=schedule,sampling_seed=20261008,inputs=sources,initial_tensor_hashes=initial,
            loss_weights=dict(physical=.5,terminal=.5),full_remaining_velocities_per_training_forward=14,
            shared_calls_per_repair=dict(outer=0,inner=0) if args.arm=='no_loop' else dict(outer=4,inner=4),
            source_sha256={n:file_hash(ROOT/n) for n in NAMES},gradient_proof_sha256=file_hash(proof_dir/'pipeline_status.json'),
            trainable_parameters=sum(p.numel() for p in repair.trainable_parameters()),
            unchanged_base_parameters=sum(p.numel() for p in runtime.model.parameters()),
            head_parameters=sum(p.numel() for p in repair.head.parameters()),input_normalizer='fixed v62 head buffer',
            capacity_gate=dict(each_teacher_mse_less_than=.01,max35_preservation_mse_less_than=.001),
            checkpoint_selection='fixed final step210; no best loss/image/val selection',
            no_automatic_endpoint_generation_or_val_or_controller=True,FLOPs=None)
        freeze_config(out/'config.json',config);(out/'source').mkdir()
        for name in NAMES:shutil.copy2(ROOT/name,out/'source'/name)
        shutil.copy2(ROOT/'STAGE3_NONLINEAR_RECURRENT_PROTOCOL_20261008.md',out/'frozen_protocol.md')
        base_hash={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}
        core_hash=repair.operator.core_hashes();normalizer=tensor_hash(repair.head.input_rms)
        optimizer=torch.optim.AdamW(repair.trainable_parameters(),lr=1e-4,weight_decay=0)
        features_cache={};targets={};states={};options_cache={}
        with full_model_context(runtime),(out/'train.jsonl').open('w') as log:
            def prepare(sid):
                if sid not in features_cache:
                    state=corpus.state(sid);states[sid]=state
                    features_cache[sid],options_cache[sid]=capture_features(runtime,state,repair.operator)
                    targets[sid]=training_target(runtime,corpus,state,features_cache[sid],repair,labels)
                return states[sid],features_cache[sid],options_cache[sid],targets[sid]
            for step,(sid,preserve) in enumerate(schedule,1):
                state,features,options,target=prepare(sid)
                if target['preservation']!=preserve:raise ValueError('TRAIN schedule identity changed')
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                rng_restore(state['rng']);optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                original=state['latent'].to(runtime.device).float();baseline=state['baseline_final_latent'].to(runtime.device).float()
                delta=repair.latent_delta(runtime,state,features,options['modality_positions'])
                final=replay_delta(runtime,state,delta,checkpoint_rollout=True)
                loss,physical,terminal=normalized_losses(delta,target['delta'],original,final,target['final'],baseline,preserve)
                loss.backward();groups=gradient_groups(repair)
                if any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base receives gradient')
                if args.arm!='trainable_loop' and (groups['outer_core'] or groups['inner_core']):raise ValueError('Frozen core gradient')
                grad=torch.nn.utils.clip_grad_norm_(repair.trainable_parameters(),1.)
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad)):raise ValueError('Nonfinite loss/gradient')
                optimizer.step();torch.cuda.synchronize()
                row=dict(step=step,state_id=sid,preservation_step=preserve,optimizer_update=True,
                    loss=float(loss.detach()),physical_loss=float(physical.detach()),terminal_loss=float(terminal.detach()),
                    gradient_norm_before_clip=float(grad),gradient_abs_sum=groups,actual_shared_calls=repair.last_shared_calls,
                    full_remaining_velocity_forward_calls=14,rollout_checkpoint_recomputation_in_backward=True,
                    post_update_prefix_or_KV_used=False,walltime_seconds=time.perf_counter()-started,
                    peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,FLOPs=None)
                log.write(json.dumps(row)+'\n');log.flush()
                status('training',optimizer_updates=step,state_id=sid,loss=row['loss'])
                del delta,final,loss,physical,terminal
            # Same fixed final checkpoint; all-state audit never chooses a checkpoint.
            status('auditing_fixed_step210',optimizer_updates=210);rows=[]
            with torch.no_grad():
                for sid in sorted(set(corpus.teachers)|set(corpus.correct)):
                    state,features,options,target=prepare(sid)
                    repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                    delta=repair.latent_delta(runtime,state,features,options['modality_positions'])
                    original=state['latent'].to(runtime.device).float()
                    scale=(original.square().mean() if target['preservation'] else target['delta'].square().mean()).clamp_min(1e-5)
                    value=float((delta-target['delta']).square().mean()/scale)
                    rows.append(dict(state_id=sid,preservation=target['preservation'],physical_relative_target_mse=value))
            teacher=[q['physical_relative_target_mse'] for q in rows if not q['preservation']]
            correct=[q['physical_relative_target_mse'] for q in rows if q['preservation']]
            passed=all(q<.01 for q in teacher) and max(correct)<.001
            audit=dict(states=rows,teacher_mean=sum(teacher)/3,teacher_max=max(teacher),
                preservation_mean=sum(correct)/35,preservation_max=max(correct),physical_capacity_gate_passed=passed,
                no_new_endpoint_or_quality_measurement=True)
            write_json(out/'capacity_report.json',audit)
        if repair.operator.core_hashes()!=core_hash or base_hash!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}:
            raise ValueError('Frozen base/shared weights changed')
        if tensor_hash(repair.head.input_rms)!=normalizer:raise ValueError('Fixed normalization buffer changed')
        if frozen_inputs()[-1]!=sources:raise ValueError('Source inputs changed')
        final_hashes=list(tensor_fingerprints(repair))
        frozen_keys=[k for k in initial[0] if '.readout.' in k or ('.low_rank.' in k and args.arm!='trainable_loop')]
        if any(initial[0][k]!=final_hashes[0][k] for k in frozen_keys):raise ValueError('Unused/frozen adapter changed')
        path=out/'repair_step210.pt'
        torch.save(dict(**repair.portable_state(),step=210,metadata=config,optimizer_updates_this_run=210,
                        warm_start='v52 step100 operator + v62 final accepted nonlinear head'),path)
        write_json(out/'training_complete.json',dict(optimizer_updates=210,checkpoint_sha256=file_hash(path),
            final_tensor_hashes=final_hashes,base_hash_unchanged=True,base_gradients=False,source_inputs_unchanged=True,
            physical_capacity_gate_passed=passed,training_states_seen=len(states),new_scoring_images=0,controller_started=False))
        status('complete',optimizer_updates=210,physical_capacity_gate_passed=passed,checkpoint_sha256=file_hash(path))
    except Exception as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
