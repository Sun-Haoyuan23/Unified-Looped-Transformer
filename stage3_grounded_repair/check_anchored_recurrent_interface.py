"""v75 same-state representation audit and real anchored terminal gradients."""
import json,shutil,time
import torch
from analytic_readout_operator import load_adapter
from anchored_recurrent_repair import AnchoredRecurrentRepair,branch_inputs,revisit_hidden
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_recurrent_protocol import ROOT,WORK,NAMES as PRIOR_NAMES,proof_inputs,make_repair,target_on_device
from nonlinear_recurrent_protocol import tensor_fingerprints,gradient_groups
from nonlinear_recurrent_repair import NonlinearRecurrentRepair
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta

OUT=ROOT/'anchored_recurrent_interface_v75';PROTOCOL='STAGE3_ANCHORED_INTERFACE_PROTOCOL_20261009.md'
NAMES=tuple(dict.fromkeys(PRIOR_NAMES+('anchored_recurrent_repair.py','test_anchored_recurrent_repair.py',
    'check_anchored_recurrent_interface.py','run_anchored_recurrent_interface.py')))

def metric(branch,h,source,target):
    a=branch.norm(h.float()).double();s=branch.norm(source.float()).double();t=branch.norm(target.float()).double()
    d=a-s;g=t-s;scale=g.square().mean();err=(a-t).square().mean()
    return dict(normalized_hidden_mse=float(err),target_relative_hidden_mse=float(err/scale) if scale>1e-12 else None,
        normalized_hidden_displacement_rms=float(d.square().mean().sqrt()),target_displacement_rms=float(scale.sqrt()),
        displacement_target_cosine=float((d*g).sum()/(d.norm()*g.norm())) if d.norm()>0 and g.norm()>0 else None,
        selected_hidden_sha256=tensor_hash(h),finite=bool(torch.isfinite(h).all()))

def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter()
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,analytical_fits=0,new_scoring_images=0,controller_started=False,**kw))
    try:
        corpus,op_path,rms,prior=proof_inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct));teachers=sorted(corpus.teachers)
        proof_path=ROOT/'v74_complete_portable_verification.json';proof=json.loads(proof_path.read_text())
        if not proof['passed'] or proof['total_accepted_parameter_updates']!=300:raise ValueError('v74 complete portable proof missing')
        if file_hash(ROOT/'transfers/stage3_expanded_v74_complete.tgz')!=proof['archive_sha256']:raise ValueError('v74 archive changed')
        warm=ROOT/'expanded_fullbatch_capacity_v74/trainable_loop/repair_lbfgs_final.pt';done=json.loads((warm.parent/'training_complete.json').read_text())
        if file_hash(warm)!=done['checkpoint_sha256']:raise ValueError('Fixed final accepted warm operator changed')
        feature_manifest_path=ROOT/'expanded_recurrent_gradient_v71/target_feature_manifest.json'
        feature_manifest={q['state_id']:q for q in json.loads(feature_manifest_path.read_text())}
        for q in feature_manifest.values():
            if file_hash(q['path'])!=q['sha256']:raise ValueError('Actual additive teacher hidden file changed')
        config=dict(source_sha256={n:file_hash(ROOT/n) for n in NAMES},protocol_sha256=file_hash(ROOT/PROTOCOL),
            v74_complete_proof_sha256=file_hash(proof_path),v74_checkpoint_sha256=file_hash(warm),
            v71_target_feature_manifest_sha256=file_hash(feature_manifest_path),target_feature_artifact_sha256={s:q['sha256'] for s,q in feature_manifest.items()},
            corpus_config_sha256=file_hash(ROOT/'expanded_process_corpus_v70/config.json'),state_ids=ids,teachers=teachers,
            state_sha256={s:file_hash(ROOT/'states/repair_train_targets_v36'/(s+'.pt')) for s in ids},
            readout='fresh zero output v71 architecture seed20261008; no old fitted head deployed',reference_gradient_connected=True,
            reference_reuse='same forward only',new_deployment_parameters=0,extra_shared_blocks_per_branch={'budget1':4,'budget2':6},
            no_optimizer_no_fit_no_new_scoring=True,FLOPs=None)
        freeze_config(OUT/'config.json',config);(OUT/'source').mkdir()
        for n in NAMES:shutil.copy2(ROOT/n,OUT/'source'/n)
        shutil.copy2(ROOT/PROTOCOL,OUT/'frozen_protocol.md')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('loading');runtime=Runtime(WORK,vae=False);torch.cuda.reset_peak_memory_stats()
        base={k:tensor_hash(v) for k,v in runtime.model.named_parameters()};payload=torch.load(warm,map_location=runtime.device,weights_only=False)
        template=make_repair(runtime,'trainable_loop',op_path,rms);load_adapter(template.operator,payload['operator'])
        if list(tensor_fingerprints(template))[0]!=done['final_tensor_hashes'][0]:raise ValueError('Warm operator tensors differ')
        repair=AnchoredRecurrentRepair(template.operator,template.head,'trainable_loop',2)
        frozen=list(tensor_fingerprints(repair));core=repair.operator.core_hashes();records=[];checks=[];saved_features={};cache=None
        nominal=0;calls={'outer':0,'inner':0}
        def count(c):
            for n in calls:calls[n]+=c[n]
        with full_model_context(runtime),torch.no_grad():
            for sid in ids:
                status('auditing_state',state_id=sid,states=len(checks))
                state=corpus.state(sid);target=target_on_device(corpus,state,runtime.device)
                features,options=capture_features(runtime,state,repair.operator);nominal+=1
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                target_features=None
                if not target['preservation']:
                    q=feature_manifest[sid];target_payload=torch.load(q['path'],map_location=runtime.device,weights_only=False)
                    if target_payload['state_id']!=sid or target_payload['target_input_sha256']!=tensor_hash(target['input']):raise ValueError('Aligned teacher input differs')
                    if target_payload['target_final_sha256']!=tensor_hash(target['final']):raise ValueError('Aligned teacher final differs')
                    target_features=target_payload['features']
                    if not torch.equal(target_payload['modality_positions'],options['modality_positions']):raise ValueError('Teacher visual span differs')
                    for n,v in target_features.items():
                        if tensor_hash(v)!=q['tensor_sha256'][n]:raise ValueError('Teacher hidden tensor hash differs')
                for name in ('outer','inner'):
                    branch=getattr(repair.operator,name);hidden,kwargs=features[name]
                    field,coords,fb,begin,end=branch_inputs(branch,hidden,repair.operator.condition,options['modality_positions'])
                    source=hidden[:1,begin:end];truth=source if target['preservation'] else target_features[name]
                    blank=torch.zeros_like(hidden)
                    pz,cz=revisit_hidden(branch,hidden,blank,kwargs,(1,2),False)
                    az,ca=revisit_hidden(branch,hidden,blank,kwargs,(1,2),True)
                    pf,cf=revisit_hidden(branch,hidden,fb,kwargs,(1,2),False)
                    af,caf=revisit_hidden(branch,hidden,fb,kwargs,(1,2),True)
                    calls[name]+=cz+ca+cf+caf
                    if any(not torch.equal(h,hidden) for h in az.values()):raise ValueError('Actual anchored zero-feedback hidden not strict identity')
                    row=dict(state_id=sid,branch=name,preservation=target['preservation'],zero_feedback_anchored_exact=True,
                        source=metric(branch,source,source,truth),plain_zero={str(k):metric(branch,h[:1,begin:end],source,truth) for k,h in pz.items()},
                        plain_feedback={str(k):metric(branch,h[:1,begin:end],source,truth) for k,h in pf.items()},
                        anchored_feedback={str(k):metric(branch,h[:1,begin:end],source,truth) for k,h in af.items()})
                    records.append(row)
                    if not target['preservation']:
                        saved_features[sid+'/'+name]=dict(source=source.cpu(),target=truth.cpu(),plain_zero2=pz[2][:1,begin:end].cpu(),plain_feedback2=pf[2][:1,begin:end].cpu(),anchored_feedback1=af[1][:1,begin:end].cpu(),anchored_feedback2=af[2][:1,begin:end].cpu())
                budgets=(2,) if target['preservation'] else (1,2)
                for budget in budgets:
                    repair.loops=budget;delta=repair.latent_delta(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                    final=replay_delta(runtime,state,delta);nominal+=14
                    if torch.count_nonzero(delta) or not torch.equal(final.cpu(),state['baseline_final_latent']):raise ValueError('Fresh anchored zero endpoint not exact')
                if not target['preservation']:
                    no=NonlinearRecurrentRepair(repair.operator,repair.head,'no_loop');no.operator.condition=repair.operator.condition
                    delta=no.latent_delta(runtime,state,features,options['modality_positions']);count(no.last_shared_calls)
                    final=replay_delta(runtime,state,delta);nominal+=14
                    if torch.count_nonzero(delta) or not torch.equal(final.cpu(),state['baseline_final_latent']):raise ValueError('Fresh auxiliary no-loop zero not exact')
                if sid==teachers[0]:cache=dict(state=state,target=target,features=features,options=options)
                checks.append(dict(state_id=sid,preservation=target['preservation'],anchored_zero_exact_budgets=list(budgets),no_loop_zero_exact=not target['preservation']))
                write_json(OUT/'state_checks.json',checks);write_json(OUT/'representation_results.json',records)
        if list(tensor_fingerprints(repair))!=frozen:raise ValueError('Read-only representation parameters changed')
        torch.save(saved_features,OUT/'teacher_representation_tensors.pt');del repair,template,saved_features;torch.cuda.empty_cache()
        proofs=[];initials={};state=cache['state'];features=cache['features'];options=cache['options'];target=cache['target']
        with full_model_context(runtime):
            for arm in ('frozen_loop','trainable_loop'):
                template=make_repair(runtime,arm,op_path,rms);load_adapter(template.operator,payload['operator'])
                repair=AnchoredRecurrentRepair(template.operator,template.head,arm,2);repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                initials[arm]=list(tensor_fingerprints(repair))
                if arm=='trainable_loop':
                    status('fresh_zero_terminal_gradient',arm=arm)
                    delta=repair.latent_delta(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                    final=replay_delta(runtime,state,delta,True);nominal+=14
                    if not torch.equal(final.cpu(),state['baseline_final_latent']):raise ValueError('Gradient zero endpoint differs')
                    (final-target['final']).square().mean().backward();zero=gradient_groups(repair)
                    if zero['head']<=0:raise ValueError('Zero readout has no terminal gradient')
                    repair.zero_grad(set_to_none=True);del delta,final
                for budget in (1,2):
                    status('probe_full_terminal_gradient',arm=arm,budget=budget)
                    saved={k:v.detach().clone() for k,v in repair.head.state_dict().items()};torch.manual_seed(75);torch.cuda.manual_seed_all(75)
                    with torch.no_grad():repair.head.output.weight.normal_(0,1e-4)
                    before=list(tensor_fingerprints(repair));repair.loops=budget
                    delta=repair.latent_delta(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                    final=replay_delta(runtime,state,delta,True);nominal+=14;(final-target['final']).square().mean().backward();groups=gradient_groups(repair)
                    if groups['head']<=0 or groups['evidence']<=0:raise ValueError('Head/evidence terminal gradient missing')
                    if arm=='trainable_loop' and min(groups['outer_core'],groups['inner_core'])<=0:raise ValueError('Anchored shared-core gradient missing')
                    if arm=='frozen_loop' and (groups['outer_core'] or groups['inner_core']):raise ValueError('Frozen core gradient')
                    if any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base gradient')
                    if list(tensor_fingerprints(repair))!=before or repair.operator.core_hashes()!=core:raise ValueError('Probe parameter mutation')
                    if repair.last_shared_calls!=dict(outer=2+2*budget,inner=2+2*budget):raise ValueError('Actual shared calls differ')
                    proofs.append(dict(arm=arm,budget=budget,state_id=state['state_id'],gradient_abs_sum=groups,shared_calls=repair.last_shared_calls,full_remaining_velocities=14,probe_restored=True))
                    repair.head.load_state_dict(saved);repair.zero_grad(set_to_none=True)
                    if list(tensor_fingerprints(repair))!=initials[arm]:raise ValueError('Probe not restored')
                    del delta,final
                del repair,template;torch.cuda.empty_cache()
        if initials['frozen_loop']!=initials['trainable_loop']:raise ValueError('Anchored matched initialization differs')
        if any(p.grad is not None for p in runtime.model.parameters()) or base!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}:raise ValueError('Base changed')
        if file_hash(warm)!=config['v74_checkpoint_sha256']:raise ValueError('Warm checkpoint changed')
        for n,h in config['source_sha256'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('Frozen interface source changed')
        proof_inputs();write_json(OUT/'arm_proofs.json',proofs);write_json(OUT/'initial_tensor_hashes.json',initials)
        if nominal!=853:raise ValueError('Nominal velocity accounting differs: '+str(nominal))
        status('passed',base_hash_unchanged=True,base_gradients=False,original_core_hash_unchanged=True,adapter_and_head_unmodified=True,
            teacher_zero_budget1_checks=6,all_state_zero_budget2_checks=41,no_loop_zero_checks=6,anchored_zero_feedback_full_hidden_checks=82,
            full_terminal_backwards=5,matched_loop_initializations=True,zero_head_terminal_gradient=zero,actual_shared_calls=calls,
            nominal_velocity_calls=nominal,backward_checkpoint_recomputation_additional=True,peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
            walltime_seconds=time.perf_counter()-started,FLOPs=None,no_training_or_quality_claim=True)
    except Exception as e:status('failed',error=str(e));raise
if __name__=='__main__':main()
