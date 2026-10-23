"""Matched warm-start full-batch physical optimization, no terminal scoring."""
import argparse,inspect,json,math,shutil,time
import torch
from analytic_readout_operator import load_adapter
from bounded_lbfgs import BoundedLBFGS
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_recurrent_protocol import ROOT,WORK,NAMES as OLD_NAMES,proof_inputs,make_repair,target_on_device
from nonlinear_recurrent_protocol import tensor_fingerprints,gradient_groups
from nonlinear_recurrent_repair import ARMS
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context

OPTIONS=dict(lr=1,max_iter=100,max_eval=125,history_size=20,tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn='strong_wolfe')
PROTOCOL='STAGE3_FULLBATCH_CAPACITY_PROTOCOL_20261009.md'
NAMES=tuple(dict.fromkeys(OLD_NAMES+('bounded_lbfgs.py','test_bounded_lbfgs.py','train_expanded_fullbatch.py','run_expanded_fullbatch.py','test_expanded_fullbatch.py')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True);arm=parser.parse_args().arm
    out=ROOT/'expanded_fullbatch_capacity_v74'/arm;out.mkdir(exist_ok=False)
    def status(phase,**kw):write_json(out/'worker_status.json',dict(phase=phase,arm=arm,time=time.time(),new_scoring_images=0,controller_started=False,**kw))
    try:
        corpus,start_op,rms,proof=proof_inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct))
        v73=json.loads((ROOT/'v73_portable_verification.json').read_text())
        if not v73['passed'] or v73['physical_reproductions']!=123:raise ValueError('v73 complete proof required')
        warm=ROOT/'expanded_recurrent_train_v72/trainable_loop/repair_step420.pt'
        done=json.loads((warm.parent/'training_complete.json').read_text())
        if file_hash(warm)!=done['checkpoint_sha256']:raise ValueError('Fixed warm checkpoint changed')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('loading');runtime=Runtime(WORK,vae=False);repair=make_repair(runtime,arm,start_op,rms)
        payload=torch.load(warm,map_location=runtime.device,weights_only=False)
        load_adapter(repair.operator,payload['operator']);repair.head.load_state_dict(payload['head'])
        initial=list(tensor_fingerprints(repair))
        if initial!=done['final_tensor_hashes']:raise ValueError('Common warm tensors mismatch')
        base={k:tensor_hash(v) for k,v in runtime.model.named_parameters()};core=repair.operator.core_hashes()
        config=dict(arm=arm,state_ids=ids,teachers=sorted(corpus.teachers),correct=sorted(corpus.correct),options=OPTIONS,
            warm_checkpoint_sha256=file_hash(warm),initial_tensor_hashes=initial,source_sha256={n:file_hash(ROOT/n) for n in NAMES},
            protocol_sha256=file_hash(ROOT/PROTOCOL),corpus_config_sha256=file_hash(ROOT/'expanded_process_corpus_v70/config.json'),
            v73_proof_sha256=file_hash(ROOT/'v73_portable_verification.json'),torch_version=torch.__version__,
            objective='.5 mean6 physical +.5 mean35 preservation physical; terminal weight0 capacity diagnostic',
            original_physical_normalization=True,per_state_clipping=False,common_warm_head_trained_with_loops=True,
            initial_no_loop_fit_may_differ=True,hard_closure_limit=125,optimizer_step_calls=1,
            fixed_final_accepted_iterate=True,no_automatic_endpoint_generation_or_val_or_controller=True,
            shared_calls_per_closure=dict(outer=0,inner=0) if arm=='no_loop' else dict(outer=164,inner=164),FLOPs=None)
        (out/'source').mkdir();torch_source=out/'source/torch_lbfgs_runtime.py'
        torch_source.write_text(inspect.getsource(torch.optim.LBFGS),encoding='utf-8');config['torch_lbfgs_source_sha256']=file_hash(torch_source)
        freeze_config(out/'config.json',config)
        for n in NAMES:shutil.copy2(ROOT/n,out/'source'/n)
        shutil.copy2(ROOT/PROTOCOL,out/'frozen_protocol.md')
        cache={};reference=[]
        with full_model_context(runtime):
            for sid in ids:
                status('preparing_original_features',state_id=sid)
                state=corpus.state(sid);target=target_on_device(corpus,state,runtime.device)
                features,options=capture_features(runtime,state,repair.operator)
                original=state['latent'].to(runtime.device).float()
                scale=(original.square().mean() if target['preservation'] else target['delta'].square().mean()).clamp_min(1e-5).detach()
                cache[sid]=(state,target,features,options,scale)
            def evaluate(backward=False):
                rows=[];total=0.;calls=dict(outer=0,inner=0)
                for sid in ids:
                    state,target,features,options,scale=cache[sid]
                    repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                    delta=repair.latent_delta(runtime,state,features,options['modality_positions'])
                    loss=(delta-target['delta']).square().mean()/scale;weight=.5/(35 if target['preservation'] else 6)
                    if backward:(weight*loss).backward()
                    value=float(loss.detach());total+=weight*value
                    rows.append(dict(state_id=sid,preservation=target['preservation'],physical_relative_target_mse=value))
                    for level in calls:calls[level]+=repair.last_shared_calls[level]
                    del delta,loss
                return total,rows,calls
            with torch.no_grad():before,reference,calls=evaluate()
            if arm!='no_loop':
                refs={q['state_id']:q for q in json.loads((warm.parent/'capacity_report.json').read_text())['states']}
                for q in reference:
                    expected=refs[q['state_id']]['physical_relative_target_mse']
                    if not math.isclose(q['physical_relative_target_mse'],expected,rel_tol=1e-4,abs_tol=1e-6):raise ValueError('Warm starting MSE mismatch')
            write_json(out/'initial_capacity.json',dict(loss=before,states=reference,actual_shared_calls=calls))
            with (out/'evaluations.jsonl').open('w') as log,(out/'accepted_updates.jsonl').open('w') as accepted:
                def on_accept(row):accepted.write(json.dumps(row)+'\n');accepted.flush()
                optimizer=BoundedLBFGS(repair.trainable_parameters(),on_accept=on_accept,**OPTIONS)
                def closure():
                    optimizer.zero_grad(set_to_none=True);began=time.perf_counter();loss,rows,calls=evaluate(backward=True)
                    grad=float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in repair.trainable_parameters() if p.grad is not None])))
                    if not math.isfinite(loss) or not math.isfinite(grad):raise ValueError('Nonfinite full-batch gradient')
                    if any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base gradient')
                    groups=gradient_groups(repair)
                    if arm!='trainable_loop' and (groups['outer_core'] or groups['inner_core']):raise ValueError('Frozen/skipped core gradient')
                    teacher=[q['physical_relative_target_mse'] for q in rows if not q['preservation']];correct=[q['physical_relative_target_mse'] for q in rows if q['preservation']]
                    row=dict(evaluation=optimizer.audit['closure_evaluations'],loss=loss,gradient_norm=grad,gradient_abs_sum=groups,
                        teacher_mean=sum(teacher)/6,preservation_mean=sum(correct)/35,actual_shared_calls=calls,
                        trial=optimizer._in_trial,walltime_seconds=time.perf_counter()-began)
                    log.write(json.dumps(row)+'\n');log.flush();status('optimizing',optimizer_counters=optimizer.audit.copy(),loss=loss)
                    return torch.tensor(loss,device=runtime.device,dtype=torch.float64)
                torch.cuda.reset_peak_memory_stats();started=time.perf_counter();optimizer.step(closure);seconds=time.perf_counter()-started
            optimizer.zero_grad(set_to_none=True);final_loss,rows,calls=evaluate(backward=True)
            final_grad=float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in repair.trainable_parameters() if p.grad is not None])))
            if not math.isfinite(final_grad) or final_loss>before+1e-7+1e-5*abs(before):raise ValueError('Final accepted objective invalid')
            teacher=[q['physical_relative_target_mse'] for q in rows if not q['preservation']];correct=[q['physical_relative_target_mse'] for q in rows if q['preservation']]
            passed=all(q<.01 for q in teacher) and max(correct)<.001
            write_json(out/'capacity_report.json',dict(states=rows,teacher_mean=sum(teacher)/6,teacher_max=max(teacher),
                preservation_mean=sum(correct)/35,preservation_max=max(correct),physical_capacity_gate_passed=passed))
        if base!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()} or repair.operator.core_hashes()!=core:raise ValueError('Base changed')
        final_hash=list(tensor_fingerprints(repair))
        for key,h in initial[0].items():
            if '.readout.' in key or ('.low_rank.' in key and arm!='trainable_loop'):
                if final_hash[0][key]!=h:raise ValueError('Frozen/unused parameter changed')
        if final_hash[1]['input_rms']!=initial[1]['input_rms']:raise ValueError('RMS changed')
        if file_hash(warm)!=config['warm_checkpoint_sha256']:raise ValueError('Warm checkpoint bytes changed')
        proof_inputs()
        for n,h in config['source_sha256'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('Source changed')
        path=out/'repair_lbfgs_final.pt';torch.save(dict(**repair.portable_state(),source_step420=True,optimizer_audit=optimizer.audit,
            final_accepted_iterate=True,physical_only_capacity_diagnostic=True,optimizer_state_not_resumable=True),path)
        write_json(out/'training_complete.json',dict(optimizer_audit=optimizer.audit,initial_loss=before,final_loss=final_loss,
            final_gradient_norm=final_grad,optimizer_walltime_seconds=seconds,checkpoint_sha256=file_hash(path),
            initial_tensor_hashes=initial,final_tensor_hashes=final_hash,base_unchanged=True,base_gradients=False,
            physical_capacity_gate_passed=passed,nominal_feature_capture_velocities=41,terminal_rollout_velocities=0,
            extra_initial_objective_forwards=1,extra_final_objective_forwards=1,extra_final_backward=1,
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,FLOPs=None))
        status('complete',optimizer_counters=optimizer.audit,physical_capacity_gate_passed=passed)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
