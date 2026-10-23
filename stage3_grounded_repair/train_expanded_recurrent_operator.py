"""v72 fixed420 matched expanded TRAIN pilot, no automatic quality expansion."""
import argparse,json,shutil,time
import torch
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_process_corpus import expanded_schedule
from expanded_recurrent_protocol import ROOT,WORK,NAMES,proof_inputs,make_repair,target_on_device
from nonlinear_recurrent_protocol import tensor_fingerprints,gradient_groups
from nonlinear_recurrent_repair import ARMS,normalized_losses
from runtime import Runtime,rng_restore,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True);args=parser.parse_args()
    out=ROOT/'expanded_recurrent_train_v72'/args.arm;out.mkdir(exist_ok=False)
    def status(phase,**kw):write_json(out/'worker_status.json',dict(phase=phase,arm=args.arm,time=time.time(),**kw))
    try:
        corpus,checkpoint,rms,proof=proof_inputs()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('loading',optimizer_updates=0);runtime=Runtime(WORK,vae=False)
        repair=make_repair(runtime,args.arm,checkpoint,rms)
        initial=list(tensor_fingerprints(repair))
        if initial!=proof['initial_tensor_hashes'][args.arm]:raise ValueError('Proven initialization differs')
        schedule=expanded_schedule(corpus.teachers,corpus.correct)
        config=dict(arm=args.arm,steps=420,schedule=schedule,sampling_seed=20261008,optimizer='AdamW',lr=1e-4,weight_decay=0.,gradient_clip=1.,
            teacher_count=6,preservation_count=35,teacher_updates=210,preservation_updates=210,
            each_teacher_seen=35,each_preservation_seen=6,initial_tensor_hashes=initial,
            operator_start='v52 joint_process step100',head_start='fresh identical zero output head all3 arms',
            common_input_rms_tensor_sha256=tensor_hash(rms),common_normalizer_all_three_arms=True,
            loss_weights=dict(physical=.5,terminal=.5),hidden_auxiliary_loss=0.,
            target_source='v50/v69 scored additive endpoints with original FP32 delta',
            source_sha256={n:file_hash(ROOT/n) for n in NAMES},corpus_config_sha256=file_hash(ROOT/'expanded_process_corpus_v70/config.json'),
            real_gradient_proof_sha256=file_hash(ROOT/'expanded_recurrent_gradient_v71/pipeline_status.json'),
            protocol_sha256=file_hash(ROOT/'STAGE3_EXPANDED_RECURRENT_PROTOCOL_20261008.md'),
            trainable_parameters=sum(p.numel() for p in repair.trainable_parameters()),
            stored_adapter_and_head_parameters=sum(p.numel() for n,p in repair.named_parameters() if '.shared_core.' not in n),
            base_parameters=sum(p.numel() for p in runtime.model.parameters()),
            shared_calls_per_repair=dict(outer=0,inner=0) if args.arm=='no_loop' else dict(outer=4,inner=4),
            full_remaining_velocity_calls=14,velocity_layers=dict(semantic=26,Qwen=28,flow=10),
            checkpoint_selection='fixed step420 only',capacity_gate=dict(each_teacher_mse_less_than=.01,max35_preservation_mse_less_than=.001),
            no_automatic_endpoint_generation_or_val_or_controller=True,FLOPs=None)
        freeze_config(out/'config.json',config);(out/'source').mkdir()
        for name in NAMES:shutil.copy2(ROOT/name,out/'source'/name)
        shutil.copy2(ROOT/'STAGE3_EXPANDED_RECURRENT_PROTOCOL_20261008.md',out/'frozen_protocol.md')
        base_hash={k:tensor_hash(v) for k,v in runtime.model.named_parameters()};core_hash=repair.operator.core_hashes()
        optimizer=torch.optim.AdamW(repair.trainable_parameters(),lr=1e-4,weight_decay=0.)
        states={};features_cache={};options_cache={};targets={}
        def prepare(sid):
            if sid not in states:
                state=corpus.state(sid);states[sid]=state
                features_cache[sid],options_cache[sid]=capture_features(runtime,state,repair.operator)
                targets[sid]=target_on_device(corpus,state,runtime.device)
            return states[sid],features_cache[sid],options_cache[sid],targets[sid]
        with full_model_context(runtime),(out/'train.jsonl').open('w') as log:
            for step,(sid,preserve) in enumerate(schedule,1):
                state,features,options,target=prepare(sid)
                if target['preservation']!=preserve:raise ValueError('Schedule split identity differs')
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                rng_restore(state['rng']);optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                original=state['latent'].to(runtime.device).float();baseline=state['baseline_final_latent'].to(runtime.device).float()
                delta=repair.latent_delta(runtime,state,features,options['modality_positions'])
                final=replay_delta(runtime,state,delta,checkpoint_rollout=True)
                loss,physical,terminal=normalized_losses(delta,target['delta'],original,final,target['final'],baseline,preserve)
                loss.backward();groups=gradient_groups(repair)
                if any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base gradient')
                if args.arm!='trainable_loop' and (groups['outer_core'] or groups['inner_core']):raise ValueError('Frozen/skipped core gradient')
                grad=torch.nn.utils.clip_grad_norm_(repair.trainable_parameters(),1.)
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad)):raise ValueError('Nonfinite loss/gradient')
                optimizer.step();torch.cuda.synchronize()
                row=dict(step=step,state_id=sid,preservation_step=preserve,optimizer_update=True,
                    loss=float(loss.detach()),physical_loss=float(physical.detach()),terminal_loss=float(terminal.detach()),
                    gradient_norm_before_clip=float(grad),gradient_abs_sum=groups,actual_shared_calls=repair.last_shared_calls,
                    full_remaining_velocity_forward_calls=14,rollout_checkpoint_recomputation_in_backward=True,
                    post_update_prefix_or_KV_used=False,walltime_seconds=time.perf_counter()-started,
                    peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,FLOPs=None)
                log.write(json.dumps(row)+'\n');log.flush();status('training',optimizer_updates=step,state_id=sid,loss=row['loss'])
                del delta,final,loss,physical,terminal
            status('auditing_fixed_step420',optimizer_updates=420);rows=[]
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
            write_json(out/'capacity_report.json',dict(states=rows,teacher_mean=sum(teacher)/6,teacher_max=max(teacher),
                preservation_mean=sum(correct)/35,preservation_max=max(correct),physical_capacity_gate_passed=passed,
                no_new_endpoint_or_quality_measurement=True))
        if repair.operator.core_hashes()!=core_hash or base_hash!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}:
            raise ValueError('Base/shared tensors changed')
        if tensor_hash(repair.head.input_rms)!=tensor_hash(rms):raise ValueError('Common normalizer changed')
        final_hashes=list(tensor_fingerprints(repair))
        frozen_keys=[k for k in initial[0] if '.readout.' in k or ('.low_rank.' in k and args.arm!='trainable_loop')]
        if any(initial[0][k]!=final_hashes[0][k] for k in frozen_keys):raise ValueError('Unused/frozen adapter changed')
        # Re-read all41 identities/bytes, quality origins and proven source hashes.
        proof_inputs()
        path=out/'repair_step420.pt';torch.save(dict(**repair.portable_state(),step=420,metadata=config,optimizer_updates_this_run=420),path)
        write_json(out/'training_complete.json',dict(optimizer_updates=420,checkpoint_sha256=file_hash(path),final_tensor_hashes=final_hashes,
            base_hash_unchanged=True,base_gradients=False,source_inputs_unchanged=True,physical_capacity_gate_passed=passed,
            training_states_seen=len(states),new_scoring_images=0,controller_started=False))
        status('complete',optimizer_updates=420,physical_capacity_gate_passed=passed,checkpoint_sha256=file_hash(path))
    except Exception as error:status('failed',error=str(error));raise

if __name__=='__main__':main()
