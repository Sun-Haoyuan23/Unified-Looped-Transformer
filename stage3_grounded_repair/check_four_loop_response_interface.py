"""Real four-loop interface, independently increased outer/inner calls."""
import json,shutil,time
import torch
from analytic_readout_operator import load_adapter
from anchored_recurrent_repair import anchored_token_features
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_recurrent_protocol import ROOT,WORK,proof_inputs,make_repair,target_on_device
from check_anchored_recurrent_interface import NAMES as PRIOR_NAMES
from multibudget_response_repair import MultiBudgetResponseRepair,budget_token_features
from nonlinear_recurrent_protocol import tensor_fingerprints,gradient_groups
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta
OUT=ROOT/'four_loop_response_interface_v76';PROTOCOL='STAGE3_FOUR_LOOP_PROTOCOL_20261009.md'
NAMES=tuple(dict.fromkeys(PRIOR_NAMES+('multibudget_response_repair.py','test_multibudget_response_repair.py',
    'check_four_loop_response_interface.py','run_four_loop_response_interface.py')))
def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter()
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,analytical_fits=0,controller_started=False,**kw))
    try:
        corpus,op_path,rms,old=proof_inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct));teachers=sorted(corpus.teachers)
        p75=ROOT/'v75_portable_verification.json';proof=json.loads(p75.read_text())
        if not proof['passed'] or file_hash(ROOT/'transfers/stage3_anchored_v75_results.tgz')!=proof['archive_sha256']:raise ValueError('v75 completed portable proof required')
        warm=ROOT/'expanded_fullbatch_capacity_v74/trainable_loop/repair_lbfgs_final.pt';done=json.loads((warm.parent/'training_complete.json').read_text())
        if file_hash(warm)!=done['checkpoint_sha256']:raise ValueError('Fixed warm operator changed')
        cfg=dict(source_sha256={n:file_hash(ROOT/n) for n in NAMES},protocol_sha256=file_hash(ROOT/PROTOCOL),
            v75_proof_sha256=file_hash(p75),v75_archive_sha256=proof['archive_sha256'],v74_checkpoint_sha256=file_hash(warm),
            corpus_config_sha256=file_hash(ROOT/'expanded_process_corpus_v70/config.json'),state_ids=ids,teachers=teachers,
            state_sha256={s:file_hash(ROOT/'states/repair_train_targets_v36'/(s+'.pt')) for s in ids},
            seed=20261008,budgets=[1,2,4],independent_outer_inner_pairs=[[4,2],[2,4],[4,4]],
            new_deployment_parameters=0,reference_connected=True,optimizer_updates=0,new_scoring_images=0,FLOPs=None)
        freeze_config(OUT/'config.json',cfg);(OUT/'source').mkdir()
        for n in NAMES:shutil.copy2(ROOT/n,OUT/'source'/n)
        shutil.copy2(ROOT/PROTOCOL,OUT/'frozen_protocol.md')
        status('loading');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False);torch.cuda.reset_peak_memory_stats();base={n:tensor_hash(p) for n,p in runtime.model.named_parameters()}
        payload=torch.load(warm,map_location=runtime.device,weights_only=False)
        def build(arm,outer=4,inner=4):
            template=make_repair(runtime,arm,op_path,rms);load_adapter(template.operator,payload['operator'])
            return MultiBudgetResponseRepair(template.operator,template.head,arm,outer,inner)
        repair=build('trainable_loop');frozen=list(tensor_fingerprints(repair));core=repair.operator.core_hashes();checks=[];cache=None;nominal=0;calls=dict(outer=0,inner=0)
        def count(c):
            for n in calls:calls[n]+=c[n]
        with full_model_context(runtime),torch.no_grad():
            for sid in ids:
                status('four_loop_zero_state',state_id=sid,states=len(checks));state=corpus.state(sid);target=target_on_device(corpus,state,runtime.device)
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                features,options=capture_features(runtime,state,repair.operator);nominal+=1
                prefix_exact=None
                if sid in teachers:
                    old_tokens,oc=anchored_token_features(repair.operator,features,options['modality_positions'],(1,2));count(oc)
                    new_tokens,nc=budget_token_features(repair.operator,features,options['modality_positions'],((1,1),(2,2),(4,4)));count(nc)
                    prefix_exact=all(torch.equal(old_tokens[k],new_tokens[(k,k)]) for k in (1,2))
                    if not prefix_exact:raise ValueError('Frozen v75 prefix tokens differ')
                delta=repair.latent_delta(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                final=replay_delta(runtime,state,delta);nominal+=14
                if torch.count_nonzero(delta) or not torch.equal(final.cpu(),state['baseline_final_latent']):raise ValueError('Four-loop fresh zero endpoint differs')
                if repair.last_shared_calls!=dict(outer=10,inner=10):raise ValueError('Four-loop actual shared call count')
                checks.append(dict(state_id=sid,preservation=target['preservation'],zero_endpoint_exact=True,v75_prefix_tokens_exact=prefix_exact,actual_shared_calls=repair.last_shared_calls))
                write_json(OUT/'state_checks.json',checks)
                if sid==teachers[0]:cache=dict(state=state,target=target,features=features,options=options)
        if list(tensor_fingerprints(repair))!=frozen:raise ValueError('Read-only zero checks mutated adapter/head')
        del repair;torch.cuda.empty_cache();state=cache['state'];target=cache['target'];features=cache['features'];options=cache['options'];initials=[];proofs=[]
        with full_model_context(runtime):
            for arm,outer,inner in (('trainable_loop',4,2),('trainable_loop',2,4),('trainable_loop',4,4),('frozen_loop',4,4)):
                status('four_loop_terminal_gradient',arm=arm,outer=outer,inner=inner);repair=build(arm,outer,inner)
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device);initial=list(tensor_fingerprints(repair));initials.append(initial)
                if arm=='trainable_loop' and (outer,inner)==(4,4):
                    delta=repair.latent_delta(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                    final=replay_delta(runtime,state,delta,True);nominal+=14
                    if not torch.equal(final.cpu(),state['baseline_final_latent']):raise ValueError('Zero-head gradient endpoint differs')
                    (final-target['final']).square().mean().backward();zero=gradient_groups(repair)
                    if zero['head']<=0:raise ValueError('Fresh zero head has no gradient')
                    repair.zero_grad(set_to_none=True);del delta,final
                saved={n:v.detach().clone() for n,v in repair.head.state_dict().items()};torch.manual_seed(76);torch.cuda.manual_seed_all(76)
                with torch.no_grad():repair.head.output.weight.normal_(0,1e-4)
                before=list(tensor_fingerprints(repair));delta=repair.latent_delta(runtime,state,features,options['modality_positions']);count(repair.last_shared_calls)
                final=replay_delta(runtime,state,delta,True);nominal+=14;(final-target['final']).square().mean().backward();g=gradient_groups(repair)
                if min(g['head'],g['evidence'])<=0:raise ValueError('Four-loop head/evidence gradient missing')
                if arm=='trainable_loop' and min(g['outer_core'],g['inner_core'])<=0:raise ValueError('Four-loop shared core gradient missing')
                if arm=='frozen_loop' and (g['outer_core'] or g['inner_core']):raise ValueError('Frozen core gradient')
                if any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base gradient')
                if list(tensor_fingerprints(repair))!=before or repair.operator.core_hashes()!=core:raise ValueError('Probe parameter mutation')
                if repair.last_shared_calls!=dict(outer=2+2*outer,inner=2+2*inner):raise ValueError('Independent call budgets differ')
                proofs.append(dict(arm=arm,outer_budget=outer,inner_budget=inner,gradient_abs_sum=g,shared_calls=repair.last_shared_calls,probe_restored=True,full_remaining_velocities=14))
                repair.head.load_state_dict(saved);repair.zero_grad(set_to_none=True)
                if list(tensor_fingerprints(repair))!=initial:raise ValueError('Four-loop probe not restored')
                del repair,delta,final;torch.cuda.empty_cache()
        if any(p!=initials[0] for p in initials):raise ValueError('Matched budget initial tensors differ')
        if base!={n:tensor_hash(p) for n,p in runtime.model.named_parameters()} or any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base mutation/gradient')
        if file_hash(warm)!=cfg['v74_checkpoint_sha256']:raise ValueError('Source checkpoint changed')
        for n,h in cfg['source_sha256'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('Frozen source changed')
        proof_inputs();write_json(OUT/'arm_proofs.json',proofs);write_json(OUT/'initial_tensor_hashes.json',initials)
        if nominal!=685:raise ValueError('Nominal velocity accounting differs')
        status('passed',zero_endpoint_checks=41,v75_prefix_teacher_checks=6,full_terminal_backwards=5,all_initial_tensors_equal=True,
            base_hash_unchanged=True,base_gradients=False,core_hash_unchanged=True,adapter_and_head_unmodified=True,zero_head_terminal_gradient=zero,
            nominal_velocity_calls=nominal,actual_shared_calls=calls,peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
            walltime_seconds=time.perf_counter()-started,backward_checkpoint_recomputation_additional=True,FLOPs=None,no_training_or_quality_claim=True)
    except Exception as e:status('failed',error=str(e));raise
if __name__=='__main__':main()
