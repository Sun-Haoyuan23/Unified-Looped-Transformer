"""Finite read-only v78 mapping/hidden audit and zero-feedback reference VJPs."""
import json,math,shutil,time
from contextlib import contextmanager
import torch
from analytic_readout_operator import load_adapter
from anchored_recurrent_repair import branch_inputs
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_recurrent_protocol import target_on_device
from multibudget_response_repair import shared_response_prefixes
from nonlinear_recurrent_protocol import tensor_fingerprints
from nonlinear_token_readout import patch_delta
from process_alignment_diagnostics import direction_stats,recomputed_reference,gradient_pair_stats
from process_four_loop_protocol import ROOT,WORK,CHECK,TRAIN,NAMES,verified_training_inputs,build,aligned_hidden
from process_four_loop_repair import ARMS,normalized_hidden,process_loss,hidden_loss
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta

OUT=ROOT/'process_four_loop_alignment_v79'
PROTOCOL='STAGE3_PROCESS_ALIGNMENT_PROTOCOL_20261009.md'
NEW=('process_alignment_diagnostics.py','test_process_alignment_diagnostics.py','check_process_alignment.py','run_process_alignment.py')
PAIRS=((1,1),(2,2),(4,4),(4,2),(2,4))


@contextmanager
def count_shared(operator,counts):
    previous=[]
    for name in ('outer','inner'):
        branch=getattr(operator,name);run=branch.run_core;had='run_core' in branch.__dict__;previous.append((branch,run,had))
        def counted(index,h,kwargs,name=name,run=run):
            counts[name]+=1;return run(index,h,kwargs)
        branch.run_core=counted
    try:yield
    finally:
        for branch,run,had in previous:
            if had:branch.run_core=run
            else:del branch.run_core


def relative_physical(delta,target,original):
    scale=(original.square().mean() if target['preservation'] else target['delta'].square().mean()).clamp_min(1e-5)
    return float((delta-target['delta']).square().mean()/scale)


def branch_statistics(hidden,features,positions,truth,scales):
    begin=int(positions[0,0,0])+1;end=int(positions[0,0].sum());rows={}
    for n in ('outer','inner'):
        source=normalized_hidden(features[n][0][:1,begin:end]);prediction=normalized_hidden(hidden[n])
        rows[n]=dict(normalized_hidden_loss=float((prediction-truth[n]).square().mean()/scales[n]),
                     response=direction_stats(prediction-source,truth[n]-source),source_rms=float(source.square().mean().sqrt()))
    return rows


def prefix_predictions(repair,runtime,state,features,positions):
    values={};hidden={}
    for n in ('outer','inner'):
        branch=getattr(repair.operator,n);h,kw=features[n]
        field,coords,fb,begin,end=branch_inputs(branch,h,repair.operator.condition,positions)
        hs,_=shared_response_prefixes(branch,h,fb,kw,(1,2,4),True)
        hidden[n]={k:v[:1,begin:end] for k,v in hs.items()}
        values[n]={k:torch.cat((branch.norm(hidden[n][k]).float(),field,coords),-1) for k in hs}
    return {p:(patch_delta(runtime,repair.head(torch.cat((values['outer'][p[0]],values['inner'][p[1]]),-1)),state['latent']),
               dict(outer=hidden['outer'][p[0]],inner=hidden['inner'][p[1]])) for p in PAIRS}


def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter();nominal=0;calls=dict(outer=0,inner=0)
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,controller_started=False,**kw))
    try:
        corpus,op_path,rms,warm,proof=verified_training_inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct));teachers=sorted(corpus.teachers)
        portable=json.loads((ROOT/'v78_complete_portable_verification.json').read_text())
        if not portable['passed'] or portable['optimizer_updates']!=1680:raise ValueError('Require complete local v78 verification')
        if json.loads((TRAIN/'pipeline_status.json').read_text())['phase']!='complete':raise ValueError('v78 not complete')
        paths={a:TRAIN/a/'repair_step420.pt' for a in ARMS};hashes={a:file_hash(p) for a,p in paths.items()}
        cfg=dict(source_sha256={n:file_hash(ROOT/n) for n in tuple(dict.fromkeys(NAMES+NEW))},protocol_sha256=file_hash(ROOT/PROTOCOL),
                 checkpoint_sha256=hashes,portable_v78_sha256=file_hash(ROOT/'v78_complete_portable_verification.json'),
                 state_sha256={s:file_hash(ROOT/'states/repair_train_targets_v36'/(s+'.pt')) for s in ids},state_ids=ids,teachers=teachers,
                 target_manifest_sha256=file_hash(CHECK/'target_manifest.json'),common_rms_sha256=file_hash(CHECK/'common_input_rms.pt'),
                 fixed_prefix_pairs=PAIRS,capacity_reproduction_rtol=1e-4,capacity_reproduction_atol=1e-6,
                 planned_endpoint_replays=24,planned_terminal_backward_replays=6,new_training_or_fit=False,FLOPs=None)
        freeze_config(OUT/'config.json',cfg);(OUT/'source').mkdir()
        for n in cfg['source_sha256']:shutil.copy2(ROOT/n,OUT/'source'/n)
        shutil.copy2(ROOT/PROTOCOL,OUT/'frozen_protocol.md');status('loading')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False);torch.cuda.reset_peak_memory_stats()
        base={n:tensor_hash(p) for n,p in runtime.model.named_parameters()};repair=build(runtime,ARMS[0],op_path,rms,warm);core=repair.operator.core_hashes()
        ledger={r['state_id']:r for r in json.loads((CHECK/'target_manifest.json').read_text())};cache={}
        with full_model_context(runtime),torch.no_grad():
            for sid in ids:
                state=corpus.state(sid);target=target_on_device(corpus,state,runtime.device)
                features,options=capture_features(runtime,state,repair.operator);nominal+=1
                truth,scales,row=aligned_hidden(corpus,state,target,features,options['modality_positions'],repair.operator)
                if row!=ledger[sid]:raise ValueError('Frozen target/source identity changed')
                cache[sid]=(state,target,features,options['modality_positions'],truth,scales)
        del repair;torch.cuda.empty_cache();capacity=[];prefix=[];endpoints=[];gradients=[];final_hashes={}
        for arm in ARMS:
            status('auditing_fixed_checkpoints',arm=arm)
            repair=build(runtime,arm,op_path,rms,warm);payload=torch.load(paths[arm],map_location='cpu',weights_only=False)
            done=json.loads((TRAIN/arm/'training_complete.json').read_text())
            if payload['step']!=420 or payload['process_arm']!=arm or hashes[arm]!=done['checkpoint_sha256']:raise ValueError('Wrong fixed checkpoint')
            load_adapter(repair.operator,payload['operator']);repair.head.load_state_dict(payload['head']);del payload
            before=list(tensor_fingerprints(repair));final_hashes[arm]=before
            if before!=done['final_tensor_hashes']:raise ValueError('Loaded checkpoint tensors differ')
            recorded={r['state_id']:r for r in json.loads((TRAIN/arm/'capacity_report.json').read_text())['states']}
            with full_model_context(runtime),count_shared(repair.operator,calls),torch.no_grad():
                for sid in ids:
                    state,target,features,positions,truth,scales=cache[sid];repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                    delta,h=repair.forward_state(runtime,state,features,positions);original=state['latent'].to(runtime.device).float()
                    physical=relative_physical(delta,target,original);hidden=float(hidden_loss(h,truth,scales))
                    if not math.isclose(physical,recorded[sid]['physical_relative_target_mse'],rel_tol=1e-4,abs_tol=1e-6) or not math.isclose(hidden,recorded[sid]['normalized_hidden_loss'],rel_tol=1e-4,abs_tol=1e-6):raise ValueError('v78 capacity reproduction failed')
                    capacity.append(dict(arm=arm,state_id=sid,preservation=target['preservation'],physical_relative_target_mse=physical,
                                         normalized_hidden_loss=hidden,branches=branch_statistics(h,features,positions,truth,scales)))
                    if arm==ARMS[0]:
                        for pair,(d,v) in prefix_predictions(repair,runtime,state,features,positions).items():
                            if pair==(4,4) and not torch.equal(d,delta):raise ValueError('Nested fixed four-prefix mismatch')
                            prefix.append(dict(state_id=sid,preservation=target['preservation'],pair=pair,physical_relative_target_mse=relative_physical(d,target,original),
                                               branches=branch_statistics(v,features,positions,truth,scales)))
                    if sid in teachers:
                        final=replay_delta(runtime,state,delta);nominal+=14;baseline=state['baseline_final_latent'].to(runtime.device).float()
                        scale=(target['final']-baseline).square().mean().clamp_min(1e-5)
                        endpoints.append(dict(arm=arm,state_id=sid,normalized_terminal_mse=float((final-target['final']).square().mean()/scale),
                                              final_tensor_sha256=tensor_hash(final),full_remaining_velocity_calls=14,post_update_prefix_or_KV_used=False))
                        (OUT/'latents').mkdir(exist_ok=True);torch.save(dict(final=final.cpu(),state_id=sid,arm=arm,diagnostic_only=True),OUT/'latents'/(arm+'__'+sid+'.pt'))
                write_json(OUT/'capacity_reproduction.json',capacity);write_json(OUT/'prefix_alignment.json',prefix);write_json(OUT/'endpoint_alignment.json',endpoints)
            if arm==ARMS[0]:
                named=[(n,p) for n,p in repair.named_parameters() if p.requires_grad];names=[n for n,p in named];params=[p for n,p in named]
                with full_model_context(runtime),count_shared(repair.operator,calls):
                    for sid in teachers:
                        status('component_gradients',state_id=sid)
                        state,target,features,positions,truth,scales=cache[sid];repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                        delta,h=repair.forward_state(runtime,state,features,positions);final=replay_delta(runtime,state,delta,True);nominal+=14
                        loss,p,t,q=process_loss(delta,target,state['latent'].to(runtime.device).float(),final,state['baseline_final_latent'].to(runtime.device).float(),h,truth,scales)
                        components={};scalar=dict(physical=float(p.detach()),terminal=float(t.detach()),hidden=float(q.detach()))
                        for i,(name,value) in enumerate((('physical',p),('terminal',t),('hidden',q))):
                            gs=torch.autograd.grad(value,params,retain_graph=i<2,allow_unused=True)
                            if any(g is not None and not torch.isfinite(g).all() for g in gs):raise ValueError('Nonfinite component gradient')
                            components[name]=[None if g is None else g.detach().cpu() for g in gs]
                        gradients.append(dict(state_id=sid,losses=scalar,pairs={a+'_vs_'+b:gradient_pair_stats(components[a],components[b],names) for a,b in (('physical','terminal'),('physical','hidden'),('terminal','hidden'))}))
                        del components,delta,h,final,loss,p,t,q,gs
                write_json(OUT/'component_gradients.json',gradients)
            if list(tensor_fingerprints(repair))!=before or repair.operator.core_hashes()!=core:raise ValueError('Read-only checkpoint mutated')
            del repair;torch.cuda.empty_cache()
        # The diagnostic starts again from the exact v77 zero-feedback initialization.
        repair=build(runtime,ARMS[0],op_path,rms,warm);initial=list(tensor_fingerprints(repair))
        if initial!=proof['initial_tensor_hashes'][ARMS[0]]:raise ValueError('Zero-feedback probe initialization changed')
        state,target,features,positions,truth,scales=cache[teachers[0]];repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
        numerical=[];raw_gradients={}
        with full_model_context(runtime),count_shared(repair.operator,calls):
            for name in ('outer','inner'):
                branch=getattr(repair.operator,name);hidden,kwargs=features[name]
                named=list(branch.low_rank.named_parameters());params=[p for n,p in named]+list(branch.feedback.parameters())
                for budget in (1,2,4):
                    for variant in ('cached_connected','recomputed_connected'):
                        field,coords,fb,begin,end=branch_inputs(branch,hidden,repair.operator.condition,positions)
                        if torch.count_nonzero(fb):raise ValueError('Numerical probe requires exact zero feedback')
                        if variant=='cached_connected':out=shared_response_prefixes(branch,hidden,fb,kwargs,(budget,),True)[0][budget]
                        else:out=recomputed_reference(branch,hidden,fb,kwargs,budget)
                        if not torch.equal(out,hidden):raise ValueError('Zero feedback forward identity failed')
                        loss=.5*(normalized_hidden(out[:1,begin:end])-truth[name]).square().mean()/scales[name]
                        gs=torch.autograd.grad(loss,params,allow_unused=True);low=gs[:len(named)];feedback=gs[len(named):]
                        if any(g is not None and not torch.isfinite(g).all() for g in gs):raise ValueError('Nonfinite zero-feedback gradient')
                        key=name+'__'+str(budget)+'__'+variant
                        raw_gradients[key]={n:g.detach().cpu() for (n,p),g in zip(named,low) if g is not None}
                        numerical.append(dict(branch=name,budget=budget,variant=variant,hidden_dtype=str(hidden.dtype),low_rank_parameter_dtype=str(params[0].dtype),
                                              forward_strict_identity=True,loss=float(loss.detach()),core_gradient_abs_sum=sum(float(g.abs().sum()) for g in low if g is not None),
                                              core_gradient_l2=math.sqrt(sum(float(g.double().square().sum()) for g in low if g is not None)),
                                              feedback_gradient_l2=math.sqrt(sum(float(g.double().square().sum()) for g in feedback if g is not None))))
                        del out,loss,gs,low,feedback,field,fb
        torch.save(dict(gradients=raw_gradients,diagnostic_only=True,optimizer_updates=0),OUT/'zero_feedback_gradients.pt')
        write_json(OUT/'zero_feedback_numerics.json',numerical)
        if list(tensor_fingerprints(repair))!=initial or repair.operator.core_hashes()!=core:raise ValueError('Numerical probe mutated parameters')
        if base!={n:tensor_hash(p) for n,p in runtime.model.named_parameters()} or any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base changed or acquired gradients')
        for a,p in paths.items():
            if file_hash(p)!=hashes[a]:raise ValueError('Fixed checkpoint bytes changed')
        for n,h in cfg['source_sha256'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('Frozen source changed')
        if file_hash(ROOT/PROTOCOL)!=cfg['protocol_sha256']:raise ValueError('Protocol changed')
        verified_training_inputs();write_json(OUT/'loaded_tensor_hashes.json',final_hashes)
        status('complete',capacity_reproductions=len(capacity),prefix_predictions=len(prefix),teacher_endpoint_replays=len(endpoints),terminal_backwards=len(gradients),
               zero_feedback_numerical_checks=len(numerical),base_hash_unchanged=True,base_gradients=False,all_adapter_head_and_checkpoint_hashes_unchanged=True,
               original_core_hash_unchanged=True,nominal_velocity_calls=nominal,actual_shared_calls=calls,backward_checkpoint_recomputation_additional=True,
               peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,walltime_seconds=time.perf_counter()-started,FLOPs=None)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
