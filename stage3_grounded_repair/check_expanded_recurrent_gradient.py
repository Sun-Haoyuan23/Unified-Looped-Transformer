"""v71 actual six additive endpoints, 41 fresh-zero endpoints and gradients."""
import json,shutil,time
import torch
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_process_corpus import NATIVE_IDS
from expanded_recurrent_protocol import ROOT,WORK,NAMES,frozen_corpus,make_repair,target_on_device
from nonlinear_recurrent_protocol import tensor_fingerprints,gradient_groups
from nonlinear_recurrent_repair import ARMS
from nonlinear_token_readout import joint_token_features
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta

OUT=ROOT/'expanded_recurrent_gradient_v71'


def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter()
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,**kw))
    try:
        corpus,checkpoint=frozen_corpus()
        config=dict(source_sha256={n:file_hash(ROOT/n) for n in NAMES},corpus_config_sha256=file_hash(ROOT/'expanded_process_corpus_v70/config.json'),
            checkpoint_sha256=file_hash(checkpoint),normalizer='equal state and route mean square over all41 original TRAIN states, clamp RMS min1e-4',
            feature_routes=['no_loop','loop2'],normalizer_uses_targets=False,seed=20261008,
            gradient_probe='separate deterministic random head output weight std1e-4, restored; terminal-only full14 gradient',
            no_optimizer=True,FLOPs=None)
        freeze_config(OUT/'config.json',config);(OUT/'source').mkdir()
        for name in NAMES:shutil.copy2(ROOT/name,OUT/'source'/name)
        status('loading');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False)
        base_hash={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}
        template=make_repair(runtime,'frozen_loop',checkpoint,torch.ones(3716,device=runtime.device))
        initial_core=template.operator.core_hashes();cache={};rows=[];square_sum=None;target_feature_manifest=[]
        with full_model_context(runtime),torch.no_grad():
            for sid in sorted(set(corpus.teachers)|set(corpus.correct)):
                status('verifying_state',state_id=sid,verified_states=len(rows))
                state=corpus.state(sid);target=target_on_device(corpus,state,runtime.device)
                features,options=capture_features(runtime,state,template.operator)
                template.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                for mode in ('loop2','no_loop'):
                    tokens,calls=joint_token_features(template.operator,features,options['modality_positions'],mode)
                    square=tokens.double().square().mean((0,1)).cpu()
                    square_sum=square if square_sum is None else square_sum+square
                zero=template.latent_delta(runtime,state,features,options['modality_positions'])
                final=replay_delta(runtime,state,zero)
                if torch.count_nonzero(zero) or not torch.equal(final.cpu(),state['baseline_final_latent']):
                    raise ValueError('Fresh loop2 zero endpoint differs: '+sid)
                row=dict(state_id=sid,preservation=target['preservation'],fresh_loop2_zero_endpoint_exact=True,zero_velocity_calls=14)
                if not target['preservation']:
                    native_zero=make_repair(runtime,'no_loop',checkpoint,torch.ones(3716,device=runtime.device))
                    native_zero.operator.condition=template.operator.condition
                    z=native_zero.latent_delta(runtime,state,features,options['modality_positions'])
                    zfinal=replay_delta(runtime,state,z)
                    if torch.count_nonzero(z) or not torch.equal(zfinal.cpu(),state['baseline_final_latent']):
                        raise ValueError('Fresh no-loop zero endpoint differs')
                    actual=replay_delta(runtime,state,target['delta'])
                    if not torch.equal(actual,target['final']):
                        raise ValueError('Scored additive teacher endpoint not strict: '+sid)
                    cpu_target,target_features,target_options=corpus.target_features(runtime,state,template.operator)
                    native_feature_exact=None
                    if sid in NATIVE_IDS:
                        payload=torch.load(corpus.native_process[sid]['artifact_path'],map_location='cpu',weights_only=False)
                        native_feature_exact=all(torch.equal(target_features[n][0].cpu(),payload['target_features'][n]) for n in ('outer','inner'))
                        if not native_feature_exact:raise ValueError('Native additive features changed')
                    offset,length=target_options['modality_positions'][0,0].tolist();begin,end=offset+1,offset+length
                    path=OUT/(sid+'_aligned_features.pt')
                    # Only conditional visual tokens are supervision targets.
                    selected={n:target_features[n][0][:1,begin:end].detach().cpu() for n in ('outer','inner')}
                    torch.save(dict(state_id=sid,split='train',features=selected,modality_positions=target_options['modality_positions'].cpu(),
                                    target_input_sha256=tensor_hash(target['input']),target_final_sha256=tensor_hash(actual)),path)
                    target_feature_manifest.append(dict(state_id=sid,path=str(path),sha256=file_hash(path),
                        tensor_sha256={n:tensor_hash(v) for n,v in selected.items()},native_existing_features_exact=native_feature_exact))
                    row.update(fresh_no_loop_zero_endpoint_exact=True,scored_additive_endpoint_exact=True,teacher_velocity_calls=14)
                    del native_zero,z,zfinal,actual,target_features,target_options
                # Store only one state for gradient checks, not a deployment cache.
                if sid==sorted(corpus.teachers)[0]:cache=dict(state=state,features=features,options=options,target=target)
                rows.append(row);write_json(OUT/'state_checks.json',rows)
                del tokens,final,zero
        rms=(square_sum/(2*len(rows))).sqrt().clamp_min(1e-4).float()
        if not bool(torch.isfinite(rms).all()) or rms.numel()!=3716:raise ValueError('Invalid common feature normalizer')
        normalizer_path=OUT/'common_input_rms.pt';torch.save(dict(input_rms=rms,state_ids=[q['state_id'] for q in rows],routes=['loop2','no_loop']),normalizer_path)
        write_json(OUT/'target_feature_manifest.json',target_feature_manifest)
        del template;torch.cuda.empty_cache()
        initials={};proofs=[];state=cache['state'];features=cache['features'];options=cache['options'];target=cache['target']
        baseline=state['baseline_final_latent'].to(runtime.device).float()
        with full_model_context(runtime):
            for arm in ARMS:
                status('full_endpoint_gradient',arm=arm)
                repair=make_repair(runtime,arm,checkpoint,rms)
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                initials[arm]=list(tensor_fingerprints(repair))
                # At the true zero initialization the readout receives endpoint gradient.
                delta=repair.latent_delta(runtime,state,features,options['modality_positions'])
                final=replay_delta(runtime,state,delta,checkpoint_rollout=True)
                if not torch.equal(final,baseline):raise ValueError('Fresh arm zero not exact')
                terminal=(final-target['final']).square().mean()
                terminal.backward();zero_groups=gradient_groups(repair)
                if zero_groups['head']<=0:raise ValueError('Fresh readout lacks endpoint gradient')
                repair.zero_grad(set_to_none=True);del delta,final,terminal
                # Separate nonzero probe bypasses zero-readout gradient blocking.
                saved={k:v.detach().clone() for k,v in repair.head.state_dict().items()}
                torch.manual_seed(71);torch.cuda.manual_seed_all(71)
                with torch.no_grad():repair.head.output.weight.normal_(0,1e-4)
                before=list(tensor_fingerprints(repair))
                delta=repair.latent_delta(runtime,state,features,options['modality_positions'])
                final=replay_delta(runtime,state,delta,checkpoint_rollout=True)
                loss=(final-target['final']).square().mean();loss.backward();groups=gradient_groups(repair)
                if groups['head']<=0 or groups['evidence']<=0:raise ValueError('Missing terminal head/evidence gradient')
                if arm=='trainable_loop' and min(groups['outer_core'],groups['inner_core'])<=0:raise ValueError('Missing shared core gradient')
                if arm!='trainable_loop' and (groups['outer_core'] or groups['inner_core']):raise ValueError('Frozen or skipped core gradient')
                if any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base gradient')
                if before!=list(tensor_fingerprints(repair)) or repair.operator.core_hashes()!=initial_core:
                    raise ValueError('Dry run changed weights')
                calls=dict(outer=0,inner=0) if arm=='no_loop' else dict(outer=4,inner=4)
                if repair.last_shared_calls!=calls:raise ValueError('Wrong actual recurrence calls')
                repair.head.load_state_dict(saved);repair.zero_grad(set_to_none=True)
                if initials[arm]!=list(tensor_fingerprints(repair)):raise ValueError('Probe not restored')
                proofs.append(dict(arm=arm,state_id=state['state_id'],fresh_zero_terminal_gradient=zero_groups,
                    nonzero_probe_terminal_gradient=groups,actual_shared_calls=calls,probe_restored=True,full_velocities_per_backward=14))
                write_json(OUT/'arm_proofs.json',proofs)
                del repair,delta,final,loss;torch.cuda.empty_cache()
        if any(value!=initials[ARMS[0]] for value in initials.values()):raise ValueError('Three-arm common initialization differs')
        if base_hash!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}:raise ValueError('Base hash changed')
        if corpus.ledger()!=json.loads((ROOT/'expanded_process_corpus_v70/corpus_manifest.json').read_text()):raise ValueError('Corpus changed')
        status('passed',all_three_initializations_equal=True,initial_tensor_hashes=initials,base_hash_unchanged=True,base_gradients=False,
            teachers_strict_additive_replays=6,loop2_zero_endpoint_checks=41,no_loop_zero_endpoint_checks=6,
            full_terminal_backwards=6,common_input_rms_file_sha256=file_hash(normalizer_path),common_input_rms_tensor_sha256=tensor_hash(rms),
            common_input_rms_min=float(rms.min()),common_input_rms_max=float(rms.max()),
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,complete_walltime_seconds=time.perf_counter()-started,
            nominal_velocity_calls=41+6+41*14+6*14+6*14+6*14,
            velocity_checkpoint_backward_recomputation_additional=True,FLOPs=None,actual_generation_success=False)
    except Exception as error:status('failed',error=str(error));raise

if __name__=='__main__':main()
