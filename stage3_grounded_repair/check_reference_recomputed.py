"""v80 same-forward reference graph proof, no optimizer or quality scoring."""
import json,shutil,time
import torch
from analytic_readout_operator import load_adapter
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_recurrent_protocol import target_on_device
from nonlinear_recurrent_protocol import tensor_fingerprints
from process_four_loop_protocol import ROOT,WORK,CHECK,TRAIN,NAMES,verified_training_inputs,aligned_hidden
from process_four_loop_repair import process_loss,hidden_loss,process_gradient_groups
from reference_recomputed_repair import ARMS,OLD_ARMS,build_reference
from check_process_alignment import count_shared
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta

OUT=ROOT/'reference_recomputed_interface_v80';PROTOCOL='STAGE3_REFERENCE_RECOMPUTE_PROTOCOL_20261009.md'
NEW=('process_alignment_diagnostics.py','check_process_alignment.py','reference_recomputed_repair.py','test_reference_recomputed_repair.py','check_reference_recomputed.py','run_reference_recomputed.py')

def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter();calls=dict(outer=0,inner=0);nominal=0
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,controller_started=False,**kw))
    try:
        corpus,op_path,rms,warm,proof=verified_training_inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct));teachers=sorted(corpus.teachers)
        prior=json.loads((ROOT/'v79_portable_verification.json').read_text())
        if not prior['passed']:raise ValueError('Require complete v79 local proof')
        cfg=dict(source_sha256={n:file_hash(ROOT/n) for n in tuple(dict.fromkeys(NAMES+NEW))},protocol_sha256=file_hash(ROOT/PROTOCOL),
            v79_portable_sha256=file_hash(ROOT/'v79_portable_verification.json'),v78_checkpoint_sha256=file_hash(TRAIN/'anchor_trainable4/repair_step420.pt'),
            v77_target_manifest_sha256=file_hash(CHECK/'target_manifest.json'),v77_common_rms_sha256=file_hash(CHECK/'common_input_rms.pt'),
            state_ids=ids,teachers=teachers,arms=ARMS,state_sha256={s:file_hash(ROOT/'states/repair_train_targets_v36'/(s+'.pt')) for s in ids},
            common_rms='reuse fixed v77 original-state four-route RMS, including old plain route; never target normalized',
            loss_weights=dict(physical=.5,terminal=.5,hidden=1.),shared_calls=dict(cached=10,recomputed=16,no_loop=0),new_deployment_parameters=0,FLOPs=None)
        freeze_config(OUT/'config.json',cfg);(OUT/'source').mkdir()
        for n in cfg['source_sha256']:shutil.copy2(ROOT/n,OUT/'source'/n)
        shutil.copy2(ROOT/PROTOCOL,OUT/'frozen_protocol.md');status('loading')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False);torch.cuda.reset_peak_memory_stats();base={n:tensor_hash(p) for n,p in runtime.model.named_parameters()}
        repair=build_reference(runtime,ARMS[0],op_path,rms,warm);core=repair.operator.core_hashes();cache={};ledger={r['state_id']:r for r in json.loads((CHECK/'target_manifest.json').read_text())}
        with full_model_context(runtime),torch.no_grad():
            for sid in ids:
                state=corpus.state(sid);target=target_on_device(corpus,state,runtime.device);features,options=capture_features(runtime,state,repair.operator);nominal+=1
                truth,scales,row=aligned_hidden(corpus,state,target,features,options['modality_positions'],repair.operator)
                if row!=ledger[sid]:raise ValueError('Frozen aligned target changed')
                cache[sid]=(state,target,features,options['modality_positions'],truth,scales)
        payload=torch.load(TRAIN/'anchor_trainable4/repair_step420.pt',map_location='cpu',weights_only=False)
        load_adapter(repair.operator,payload['operator']);repair.head.load_state_dict(payload['head']);del payload
        before=list(tensor_fingerprints(repair));comparisons=[]
        endpoint={q['state_id']:q for q in json.loads((ROOT/'process_four_loop_alignment_v79/endpoint_alignment.json').read_text()) if q['arm']=='anchor_trainable4'}
        with full_model_context(runtime),count_shared(repair.operator,calls),torch.no_grad():
            for sid in ids:
                status('checking_actual_nonzero_same_forward',state_id=sid)
                state,target,features,positions,truth,scales=cache[sid];repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                repair.reference_arm=ARMS[0];a,ha=repair.forward_state(runtime,state,features,positions)
                repair.reference_arm=ARMS[1];b,hb=repair.forward_state(runtime,state,features,positions)
                if not torch.equal(a,b) or any(not torch.equal(ha[n],hb[n]) for n in ('outer','inner')):raise ValueError('Reference recomputation changed actual forward')
                same_endpoint=None
                if sid in teachers:
                    final=replay_delta(runtime,state,b);nominal+=14
                    if tensor_hash(final)!=endpoint[sid]['final_tensor_sha256']:raise ValueError('Actual same-forward endpoint differs from v79')
                    same_endpoint=True
                comparisons.append(dict(state_id=sid,actual_v78_delta_and_hidden_strict_equal=True,full14_v79_endpoint_strict_equal=same_endpoint))
        if list(tensor_fingerprints(repair))!=before:raise ValueError('Fixed checkpoint mutated')
        del repair;torch.cuda.empty_cache();initials={};zero=[];backwards=[];zero_terminal=None;zero_hidden=None
        for arm in ARMS:
            repair=build_reference(runtime,arm,op_path,rms,warm);initial=list(tensor_fingerprints(repair));initials[arm]=initial
            if initial!=proof['initial_tensor_hashes'][OLD_ARMS[arm]]:raise ValueError('Common actual initialization changed')
            with full_model_context(runtime),count_shared(repair.operator,calls),torch.no_grad():
                for sid in ids:
                    state,target,features,positions,truth,scales=cache[sid];repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                    delta,h=repair.forward_state(runtime,state,features,positions)
                    if torch.count_nonzero(delta):raise ValueError('Fresh zero head is not physical identity')
                    full=None
                    if arm==ARMS[1] or sid in teachers:
                        final=replay_delta(runtime,state,delta);nominal+=14
                        if not torch.equal(final.cpu(),state['baseline_final_latent']):raise ValueError('Zero full14 endpoint differs')
                        full=True
                    zero.append(dict(arm=arm,state_id=sid,zero_delta_strict=True,full14_baseline_strict=full))
            state,target,features,positions,truth,scales=cache[teachers[0]];repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
            with full_model_context(runtime),count_shared(repair.operator,calls):
                if arm==ARMS[1]:
                    delta,h=repair.forward_state(runtime,state,features,positions);final=replay_delta(runtime,state,delta,True);nominal+=14
                    (final-target['final']).square().mean().backward();zero_terminal=process_gradient_groups(repair)
                    if zero_terminal['head']<=0:raise ValueError('Fresh zero head gradient missing')
                    repair.zero_grad(set_to_none=True);del delta,h,final
                    delta,h=repair.forward_state(runtime,state,features,positions);hidden_loss(h,truth,scales).backward();zero_hidden=process_gradient_groups(repair)
                    if zero_hidden['feedback']<=0:raise ValueError('Zero feedback process gradient missing')
                    repair.zero_grad(set_to_none=True);del delta,h
                saved_head={n:v.detach().clone() for n,v in repair.head.state_dict().items()};saved_fb={n:{k:v.detach().clone() for k,v in getattr(repair.operator,n).feedback.state_dict().items()} for n in ('outer','inner')}
                torch.manual_seed(80);torch.cuda.manual_seed_all(80)
                with torch.no_grad():
                    repair.head.output.weight.normal_(0,1e-4)
                    for n in ('outer','inner'):getattr(repair.operator,n).feedback.weight.normal_(0,1e-4)
                probe_hashes=list(tensor_fingerprints(repair));delta,h=repair.forward_state(runtime,state,features,positions);final=replay_delta(runtime,state,delta,True);nominal+=14
                loss,p,t,q=process_loss(delta,target,state['latent'].to(runtime.device).float(),final,state['baseline_final_latent'].to(runtime.device).float(),h,truth,scales)
                loss.backward();g=process_gradient_groups(repair)
                if min(g['head'],g['feedback'],g['evidence'])<=0:raise ValueError('Nonzero probe path missing')
                if arm in ARMS[:2] and min(g['outer_core'],g['inner_core'])<=0:raise ValueError('Trainable core path missing')
                if arm in ARMS[2:] and (g['outer_core'] or g['inner_core']):raise ValueError('Frozen/skipped core gradient')
                backwards.append(dict(arm=arm,loss=float(loss.detach()),gradient_abs_sum=g,actual_shared_calls=repair.last_shared_calls,
                    final_tensor_sha256=tensor_hash(final),physical_delta_sha256=tensor_hash(delta),probe_restored=True,full_remaining_velocities=14))
                repair.zero_grad(set_to_none=True);del delta,h,final,loss,p,t,q
                if list(tensor_fingerprints(repair))!=probe_hashes:raise ValueError('Probe mutated parameters')
                repair.head.load_state_dict(saved_head)
                for n in ('outer','inner'):getattr(repair.operator,n).feedback.load_state_dict(saved_fb[n])
            if list(tensor_fingerprints(repair))!=initial or repair.operator.core_hashes()!=core:raise ValueError('Probe failed to restore or core changed')
            del repair;torch.cuda.empty_cache()
        if any(v!=initials[ARMS[0]] for v in initials.values()):raise ValueError('All initializations must match')
        if backwards[0]['physical_delta_sha256']!=backwards[1]['physical_delta_sha256'] or backwards[0]['final_tensor_sha256']!=backwards[1]['final_tensor_sha256']:raise ValueError('Matched nonzero probe forward differs')
        if base!={n:tensor_hash(p) for n,p in runtime.model.named_parameters()} or any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base changed or acquired gradients')
        for n,h in cfg['source_sha256'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('Frozen source changed')
        if file_hash(ROOT/PROTOCOL)!=cfg['protocol_sha256']:raise ValueError('Protocol changed')
        verified_training_inputs();write_json(OUT/'actual_forward_comparisons.json',comparisons);write_json(OUT/'zero_endpoint_checks.json',zero)
        write_json(OUT/'gradient_proofs.json',backwards);write_json(OUT/'initial_tensor_hashes.json',initials)
        status('passed',actual_nonzero_forward_comparisons=41,actual_v78_v79_teacher_endpoints=6,zero_delta_checks=164,full14_zero_endpoint_checks=59,
               full14_backwards=5,all_initializations_matched=True,zero_head_terminal_gradient=zero_terminal,zero_feedback_hidden_gradient=zero_hidden,
               probes_restored=True,base_hash_unchanged=True,base_gradients=False,original_core_hash_unchanged=True,nominal_velocity_calls=nominal,
               actual_shared_calls=calls,backward_checkpoint_recomputation_additional=True,peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
               walltime_seconds=time.perf_counter()-started,FLOPs=None)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
