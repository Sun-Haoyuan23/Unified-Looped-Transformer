"""Actual new process loss, original-state normalizer and full14 gradients."""
import json,shutil,time
import torch
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_recurrent_protocol import target_on_device
from nonlinear_recurrent_protocol import tensor_fingerprints
from process_four_loop_protocol import ROOT,WORK,CHECK,PROTOCOL,NAMES,inputs,build,aligned_hidden
from process_four_loop_repair import ARMS,process_tokens,hidden_loss,process_loss,process_gradient_groups
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta


def main():
    CHECK.mkdir(exist_ok=False);started=time.perf_counter()
    def status(phase,**kw):write_json(CHECK/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,controller_started=False,**kw))
    try:
        corpus,op_path,old_rms,warm=inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct));teachers=sorted(corpus.teachers)
        cfg=dict(source_sha256={n:file_hash(ROOT/n) for n in NAMES},protocol_sha256=file_hash(ROOT/PROTOCOL),
            warm_checkpoint_sha256=file_hash(warm),corpus_config_sha256=file_hash(ROOT/'expanded_process_corpus_v70/config.json'),
            v76_proof_sha256=file_hash(ROOT/'v76_portable_verification.json'),v71_target_feature_manifest_sha256=file_hash(ROOT/'expanded_recurrent_gradient_v71/target_feature_manifest.json'),
            state_ids=ids,teachers=teachers,state_sha256={s:file_hash(ROOT/'states/repair_train_targets_v36'/(s+'.pt')) for s in ids},
            arms=ARMS,loss_weights=dict(physical=.5,terminal=.5,hidden=1.),feedback_initialization='zero',head_initialization='fresh zero output seed20261008',
            common_rms='41 original states times four routes, equal token weighting, no targets',new_deployment_parameters=0,FLOPs=None)
        freeze_config(CHECK/'config.json',cfg);(CHECK/'source').mkdir()
        for n in NAMES:shutil.copy2(ROOT/n,CHECK/'source'/n)
        shutil.copy2(ROOT/PROTOCOL,CHECK/'frozen_protocol.md')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('loading');runtime=Runtime(WORK,vae=False);torch.cuda.reset_peak_memory_stats()
        base={n:tensor_hash(p) for n,p in runtime.model.named_parameters()};repair=build(runtime,ARMS[0],op_path,old_rms,warm)
        initial_without_rms=list(tensor_fingerprints(repair));core=repair.operator.core_hashes();checks=[];targets=[];rms_sum=None;rms_count=0;cache=None
        nominal=0;calls=dict(outer=0,inner=0)
        def count(c):
            for n in calls:calls[n]+=c[n]
        with full_model_context(runtime),torch.no_grad():
            for sid in ids:
                status('freezing_targets_rms_and_zero_endpoints',state_id=sid,states=len(checks))
                state=corpus.state(sid);target=target_on_device(corpus,state,runtime.device)
                features,options=capture_features(runtime,state,repair.operator);nominal+=1
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                truth,scales,row=aligned_hidden(corpus,state,target,features,options['modality_positions'],repair.operator);targets.append(row)
                zero={};identities={}
                for arm in ARMS:
                    # RMS reads only the same original-state inputs and frozen parameters.
                    tokens,hidden,c=process_tokens(repair.operator,features,options['modality_positions'],arm);count(c)
                    v=tokens.double().square().sum((0,1));rms_sum=v if rms_sum is None else rms_sum+v;rms_count+=tokens.shape[0]*tokens.shape[1]
                    repair.process_arm=arm;delta,hidden=repair.forward_state(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                    if torch.count_nonzero(delta):raise ValueError('Fresh zero head nonzero physical update')
                    identities[arm]=all(torch.equal(hidden[n],features[n][0][:1,options['modality_positions'][0,0,0]+1:sum(options['modality_positions'][0,0])]) for n in ('outer','inner')) if arm.startswith('anchor') or arm=='no_loop' else None
                    if arm.startswith('anchor') or arm=='no_loop':
                        if not identities[arm]:raise ValueError('Zero feedback source hidden identity failed')
                    if arm==ARMS[0] or sid in teachers:
                        final=replay_delta(runtime,state,delta);nominal+=14
                        if not torch.equal(final.cpu(),state['baseline_final_latent']):raise ValueError('Fresh full14 zero endpoint differs')
                        zero[arm]=True
                checks.append(dict(state_id=sid,preservation=target['preservation'],all_four_zero_deltas=True,zero_full14_endpoints=zero,source_hidden_identity=identities))
                write_json(CHECK/'state_checks.json',checks);write_json(CHECK/'target_manifest.json',targets)
                if sid==teachers[0]:cache=dict(state=state,target=target,features=features,options=options,truth=truth,scales=scales)
        rms=(rms_sum/rms_count).sqrt().float().cpu()
        if not torch.isfinite(rms).all() or not (rms>0).all():raise ValueError('Invalid common RMS')
        torch.save(dict(input_rms=rms,state_routes=41*4,target_used=False,token_observations=rms_count),CHECK/'common_input_rms.pt')
        if list(tensor_fingerprints(repair))!=initial_without_rms:raise ValueError('Preparation mutated parameters')
        del repair;torch.cuda.empty_cache();initials={};proofs=[];state=cache['state'];target=cache['target'];features=cache['features'];options=cache['options']
        truth=cache['truth'];scales=cache['scales'];zero_terminal=None;zero_hidden=None
        with full_model_context(runtime):
            for arm in ARMS:
                status('new_process_loss_gradients',arm=arm);repair=build(runtime,arm,op_path,rms,warm)
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                initial=list(tensor_fingerprints(repair));initials[arm]=initial
                original=state['latent'].to(runtime.device).float();baseline=state['baseline_final_latent'].to(runtime.device).float()
                if arm==ARMS[0]:
                    delta,h=repair.forward_state(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                    final=replay_delta(runtime,state,delta,True);nominal+=14
                    if not torch.equal(final.cpu(),state['baseline_final_latent']):raise ValueError('Fresh zero terminal backward differs')
                    (final-target['final']).square().mean().backward();zero_terminal=process_gradient_groups(repair)
                    if zero_terminal['head']<=0:raise ValueError('Zero head terminal gradient missing')
                    repair.zero_grad(set_to_none=True);del delta,h,final
                    delta,h=repair.forward_state(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                    hidden_loss(h,truth,scales).backward();zero_hidden=process_gradient_groups(repair)
                    if zero_hidden['feedback']<=0:raise ValueError('Actual process loss cannot start zero feedback')
                    repair.zero_grad(set_to_none=True);del delta,h
                    torch.save(repair.portable_state(),CHECK/'common_initialization.pt')
                saved_head={n:v.detach().clone() for n,v in repair.head.state_dict().items()}
                saved_fb={n:getattr(repair.operator,n).feedback.state_dict() for n in ('outer','inner')}
                saved_fb={n:{k:v.detach().clone() for k,v in q.items()} for n,q in saved_fb.items()}
                torch.manual_seed(77);torch.cuda.manual_seed_all(77)
                with torch.no_grad():
                    repair.head.output.weight.normal_(0,1e-4)
                    for n in ('outer','inner'):getattr(repair.operator,n).feedback.weight.normal_(0,1e-4)
                before=list(tensor_fingerprints(repair));delta,h=repair.forward_state(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                final=replay_delta(runtime,state,delta,True);nominal+=14
                loss,p,t,q=process_loss(delta,target,original,final,baseline,h,truth,scales);loss.backward();g=process_gradient_groups(repair)
                if min(g['head'],g['feedback'],g['evidence'])<=0:raise ValueError('Process probe trainable path missing')
                if arm in (ARMS[0],ARMS[2]) and min(g['outer_core'],g['inner_core'])<=0:raise ValueError('Process probe core gradient missing')
                if arm in (ARMS[1],ARMS[3]) and (g['outer_core'] or g['inner_core']):raise ValueError('Frozen/skipped core gradient')
                expected=dict(outer=10,inner=10) if arm.startswith('anchor') else (dict(outer=8,inner=8) if arm.startswith('plain') else dict(outer=0,inner=0))
                if repair.last_shared_calls!=expected:raise ValueError('Actual four-route call count differs')
                entry=dict(arm=arm,loss=float(loss.detach()),physical=float(p.detach()),terminal=float(t.detach()),hidden=float(q.detach()),gradient_abs_sum=g,actual_shared_calls=expected,full_remaining_velocities=14,probe_restored=True)
                repair.zero_grad(set_to_none=True);del delta,h,final,loss,p,t,q
                if arm==ARMS[0]:
                    delta,h=repair.forward_state(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                    final=replay_delta(runtime,state,delta,True);nominal+=14;(final-target['final']).square().mean().backward();terminal=process_gradient_groups(repair)
                    if min(terminal['outer_core'],terminal['inner_core'],terminal['evidence'])<=0:raise ValueError('Terminal-only core/evidence path missing')
                    entry['terminal_only_gradient_abs_sum']=terminal
                    repair.zero_grad(set_to_none=True);del delta,h,final
                if any(p.grad is not None for p in runtime.model.parameters()) or repair.operator.core_hashes()!=core or list(tensor_fingerprints(repair))!=before:raise ValueError('Base/parameter mutation or gradient')
                repair.head.load_state_dict(saved_head)
                for n in ('outer','inner'):getattr(repair.operator,n).feedback.load_state_dict(saved_fb[n])
                if list(tensor_fingerprints(repair))!=initial:raise ValueError('Probe not restored')
                proofs.append(entry);del repair;torch.cuda.empty_cache()
        if any(v!=initials[ARMS[0]] for v in initials.values()):raise ValueError('Four initializations differ')
        if base!={n:tensor_hash(p) for n,p in runtime.model.named_parameters()} or any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base changed')
        for n,digest in cfg['source_sha256'].items():
            if file_hash(ROOT/n)!=digest:raise ValueError('Frozen source changed')
        inputs();write_json(CHECK/'arm_proofs.json',proofs);write_json(CHECK/'initial_tensor_hashes.json',initials)
        status('passed',all_four_initial_tensors_equal=True,initial_tensor_hashes=initials,base_hash_unchanged=True,base_gradients=False,original_core_hash_unchanged=True,
            probes_restored=True,all41_four_route_zero_delta_checks=True,full14_zero_endpoint_checks=59,full14_backwards=6,
            zero_head_terminal_gradient=zero_terminal,zero_feedback_hidden_gradient=zero_hidden,hidden_targets_and_scales_frozen_before_training=True,
            target_manifest_sha256=file_hash(CHECK/'target_manifest.json'),common_input_rms_file_sha256=file_hash(CHECK/'common_input_rms.pt'),common_input_rms_tensor_sha256=tensor_hash(rms),
            rms_range=[float(rms.min()),float(rms.max())],nominal_velocity_calls=nominal,actual_shared_calls=calls,
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,walltime_seconds=time.perf_counter()-started,backward_checkpoint_recomputation_additional=True,FLOPs=None)
    except Exception as e:status('failed',error=str(e));raise
if __name__=='__main__':main()
