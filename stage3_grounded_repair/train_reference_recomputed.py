"""Fixed420 v81 connected-reference recomputation, two matched new arms."""
import argparse,json,shutil,time
import torch
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_process_corpus import expanded_schedule
from expanded_recurrent_protocol import target_on_device
from nonlinear_recurrent_protocol import tensor_fingerprints
from process_four_loop_protocol import ROOT,WORK,CHECK,NAMES as OLD_NAMES,verified_training_inputs,aligned_hidden
from reference_recomputed_repair import OLD_ARMS,build_reference as build
from check_process_alignment import count_shared
TRAIN=ROOT/'reference_recomputed_train_v81'
PROTOCOL='STAGE3_REFERENCE_RECOMPUTE_TRAINING_PROTOCOL_20261009.md'
NAMES=tuple(dict.fromkeys(OLD_NAMES+('process_alignment_diagnostics.py','check_process_alignment.py',
    'reference_recomputed_repair.py','test_reference_recomputed_repair.py','check_reference_recomputed.py',
    'run_reference_recomputed.py','train_reference_recomputed.py','run_reference_recomputed_training.py')))
ARMS=('recomputed_trainable4','recomputed_frozen4')
def check_v80():
    proof=json.loads((ROOT/'v80_portable_verification.json').read_text())
    if not proof['passed']:raise ValueError('Complete local v80 proof required')
    c=json.loads((ROOT/'reference_recomputed_interface_v80/config.json').read_text())['configuration']
    for n,h in c['source_sha256'].items():
        if file_hash(ROOT/n)!=h:raise ValueError('Frozen v80 source changed')
    return proof
from process_four_loop_repair import process_loss,hidden_loss,process_gradient_groups
from runtime import Runtime,rng_restore,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta


def effective_weights(operator):
    result={}
    with torch.no_grad():
        for name in ('outer','inner'):
            branch=getattr(operator,name)
            for i,targets in enumerate(branch.targets):
                weights=dict(branch.shared_core[i].named_parameters())
                for param,key in targets.items():
                    result[name+'/'+key]=(weights[param]+branch.low_rank[key].delta().to(weights[param].dtype)).detach().cpu()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True);arm=parser.parse_args().arm
    out=TRAIN/arm;out.mkdir(exist_ok=False)
    def status(phase,**kw):write_json(out/'worker_status.json',dict(phase=phase,arm=arm,time=time.time(),**kw))
    try:
        corpus,op_path,rms,warm,proof=verified_training_inputs();v80=check_v80()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('loading',optimizer_updates=0);runtime=Runtime(WORK,vae=False);repair=build(runtime,arm,op_path,rms,warm)
        initial=list(tensor_fingerprints(repair))
        if initial!=proof['initial_tensor_hashes'][OLD_ARMS[arm]]:raise ValueError('v77 actual initialization changed')
        actual_v80=json.loads((ROOT/'reference_recomputed_interface_v80/initial_tensor_hashes.json').read_text())
        if initial!=actual_v80[arm]:raise ValueError('v80 actual initialization changed')
        schedule=expanded_schedule(corpus.teachers,corpus.correct);ledger={q['state_id']:q for q in json.loads((CHECK/'target_manifest.json').read_text())}
        config=dict(arm=arm,steps=420,schedule=schedule,sampling_seed=20261008,optimizer='AdamW',lr=1e-4,weight_decay=0.,gradient_clip=1.,
            teacher_count=6,preservation_count=35,teacher_updates=210,preservation_updates=210,each_teacher_seen=35,each_preservation_seen=6,
            initial_tensor_hashes=initial,operator_start='v74 final accepted trainable operator then zero feedback',head_start='fresh identical zero output',
            common_input_rms_tensor_sha256=tensor_hash(rms),common_normalizer_frozen_v77_four_route_RMS=True,reference_recomputed_per_iteration=True,reference_connected=True,
            loss_weights=dict(physical=.5,terminal=.5,hidden=1.),hidden_target_normalization='FP32 nonaffine layernorm eps1e-5, fixed v77 scale',
            target_manifest_sha256=file_hash(CHECK/'target_manifest.json'),warm_checkpoint_sha256=file_hash(warm),target_source='v50/v69 scored additive input/final and v71 actual additive hidden',
            source_sha256={n:file_hash(ROOT/n) for n in NAMES},protocol_sha256=file_hash(ROOT/PROTOCOL),
            real_gradient_proof_sha256=file_hash(ROOT/'reference_recomputed_interface_v80/pipeline_status.json'),portable_proof_sha256=file_hash(ROOT/'v80_portable_verification.json'),
            v80_initialization_sha256=file_hash(ROOT/'reference_recomputed_interface_v80/initial_tensor_hashes.json'),
            reused_v78_control_checkpoints={a:file_hash(ROOT/'process_four_loop_train_v78'/a/'repair_step420.pt') for a in ('anchor_trainable4','no_loop')},
            reused_controls='completed v78 cached trainable and no_loop, not new training or equal walltime/calls',
            corpus_config_sha256=file_hash(ROOT/'expanded_process_corpus_v70/config.json'),
            trainable_parameters=sum(p.numel() for p in repair.trainable_parameters()),stored_adapter_and_head_parameters=sum(p.numel() for n,p in repair.named_parameters() if '.shared_core.' not in n),
            base_parameters=sum(p.numel() for p in runtime.model.parameters()),shared_calls_per_repair=dict(outer=16,inner=16),
            full_remaining_velocity_calls=14,velocity_layers=dict(semantic=26,Qwen=28,flow=10),checkpoint_selection='fixed step420',
            capacity_gate=dict(each_teacher_mse_less_than=.01,max35_preservation_mse_less_than=.001),no_automatic_endpoint_generation_or_val_or_controller=True,FLOPs=None)
        freeze_config(out/'config.json',config);(out/'source').mkdir()
        for n in NAMES:shutil.copy2(ROOT/n,out/'source'/n)
        shutil.copy2(ROOT/PROTOCOL,out/'frozen_protocol.md')
        base={n:tensor_hash(p) for n,p in runtime.model.named_parameters()};core=repair.operator.core_hashes();effective_initial=effective_weights(repair.operator)
        optimizer=torch.optim.AdamW(repair.trainable_parameters(),lr=1e-4,weight_decay=0.)
        cache={}
        def prepare(sid):
            if sid not in cache:
                state=corpus.state(sid);features,options=capture_features(runtime,state,repair.operator)
                target=target_on_device(corpus,state,runtime.device)
                truth,scales,row=aligned_hidden(corpus,state,target,features,options['modality_positions'],repair.operator)
                if row!=ledger[sid]:raise ValueError('Pre-frozen hidden target/scale/source changed')
                cache[sid]=(state,features,options,target,truth,scales)
            return cache[sid]
        with full_model_context(runtime),(out/'train.jsonl').open('w') as log:
            for step,(sid,preserve) in enumerate(schedule,1):
                state,features,options,target,truth,scales=prepare(sid)
                if preserve!=target['preservation']:raise ValueError('Schedule target role changed')
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                rng_restore(state['rng']);optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                original=state['latent'].to(runtime.device).float();baseline=state['baseline_final_latent'].to(runtime.device).float()
                actual=dict(outer=0,inner=0)
                with count_shared(repair.operator,actual):delta,h=repair.forward_state(runtime,state,features,options['modality_positions'])
                if actual!=config['shared_calls_per_repair']:raise ValueError('Actual shared call count mismatch')
                final=replay_delta(runtime,state,delta,True)
                loss,p,t,q=process_loss(delta,target,original,final,baseline,h,truth,scales);loss.backward();groups=process_gradient_groups(repair)
                if any(v.grad is not None for v in runtime.model.parameters()):raise ValueError('Base gradient')
                if arm=='recomputed_frozen4' and (groups['outer_core'] or groups['inner_core']):raise ValueError('Frozen/skipped core gradient')
                grad=torch.nn.utils.clip_grad_norm_(repair.trainable_parameters(),1.)
                if not torch.isfinite(loss) or not torch.isfinite(grad):raise ValueError('Nonfinite loss/gradient')
                optimizer.step();torch.cuda.synchronize()
                row=dict(step=step,state_id=sid,preservation_step=preserve,optimizer_update=True,loss=float(loss.detach()),
                    physical_loss=float(p.detach()),terminal_loss=float(t.detach()),hidden_loss=float(q.detach()),gradient_norm_before_clip=float(grad),gradient_abs_sum=groups,
                    actual_shared_calls=actual,full_remaining_velocity_forward_calls=14,rollout_checkpoint_recomputation_in_backward=True,
                    post_update_prefix_or_KV_used=False,walltime_seconds=time.perf_counter()-started,peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,FLOPs=None)
                log.write(json.dumps(row)+'\n');log.flush();status('training',optimizer_updates=step,state_id=sid,loss=row['loss'])
                del delta,h,final,loss,p,t,q
            status('auditing_fixed_step420',optimizer_updates=420);rows=[];capacity_calls=dict(outer=0,inner=0)
            with torch.no_grad():
                for sid in sorted(ledger):
                    state,features,options,target,truth,scales=prepare(sid)
                    repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                    with count_shared(repair.operator,capacity_calls):delta,h=repair.forward_state(runtime,state,features,options['modality_positions'])
                    original=state['latent'].to(runtime.device).float()
                    scale=(original.square().mean() if target['preservation'] else target['delta'].square().mean()).clamp_min(1e-5)
                    rows.append(dict(state_id=sid,preservation=target['preservation'],physical_relative_target_mse=float((delta-target['delta']).square().mean()/scale),normalized_hidden_loss=float(hidden_loss(h,truth,scales))))
            teacher=[r['physical_relative_target_mse'] for r in rows if not r['preservation']];correct=[r['physical_relative_target_mse'] for r in rows if r['preservation']]
            passed=all(v<.01 for v in teacher) and max(correct)<.001
            write_json(out/'capacity_report.json',dict(states=rows,teacher_mean=sum(teacher)/6,teacher_max=max(teacher),preservation_mean=sum(correct)/35,preservation_max=max(correct),physical_capacity_gate_passed=passed,no_new_endpoint_or_quality_measurement=True,actual_shared_calls=capacity_calls))
        optimizer.zero_grad(set_to_none=True)
        final_hashes=list(tensor_fingerprints(repair));frozen=[n for n in initial[0] if '.readout.' in n or ('.low_rank.' in n and arm=='recomputed_frozen4')]
        if any(initial[0][n]!=final_hashes[0][n] for n in frozen):raise ValueError('Frozen adapter parameter changed')
        if repair.operator.core_hashes()!=core or base!={n:tensor_hash(p) for n,p in runtime.model.named_parameters()} or any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base/core mutation or gradient')
        if tensor_hash(repair.head.input_rms)!=tensor_hash(rms):raise ValueError('Common RMS changed')
        effective_final=effective_weights(repair.operator);changes=[]
        for n,b in effective_initial.items():
            a=effective_final[n];changes.append(dict(projection=n,elements=b.numel(),effective_bf16_changed_elements=int(torch.count_nonzero(a!=b)),initial_sha256=tensor_hash(b),final_sha256=tensor_hash(a)))
        write_json(out/'effective_core_changes.json',changes);verified_training_inputs();check_v80()
        for n,h in config['source_sha256'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('Frozen v81 source changed')
        if file_hash(ROOT/PROTOCOL)!=config['protocol_sha256']:raise ValueError('Frozen v81 protocol changed')
        for a,h in config['reused_v78_control_checkpoints'].items():
            if file_hash(ROOT/'process_four_loop_train_v78'/a/'repair_step420.pt')!=h:raise ValueError('Reused v78 control changed')
        path=out/'repair_step420.pt';torch.save(dict(**repair.portable_state(),step=420,metadata=config,optimizer_updates_this_run=420),path)
        write_json(out/'training_complete.json',dict(optimizer_updates=420,checkpoint_sha256=file_hash(path),final_tensor_hashes=final_hashes,base_hash_unchanged=True,base_gradients=False,
            source_inputs_unchanged=True,physical_capacity_gate_passed=passed,training_states_seen=len(cache),new_scoring_images=0,controller_started=False))
        status('complete',optimizer_updates=420,physical_capacity_gate_passed=passed,checkpoint_sha256=file_hash(path))
    except Exception as e:status('failed',error=str(e));raise
if __name__=='__main__':main()
