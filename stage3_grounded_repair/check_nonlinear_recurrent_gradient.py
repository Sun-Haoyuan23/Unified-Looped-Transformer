"""Actual nonlinear head/shared core/full endpoint interface, zero optimizer."""
import json,shutil,time
import torch
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from nonlinear_recurrent_repair import ARMS,normalized_losses
from nonlinear_recurrent_protocol import ROOT,WORK,frozen_inputs,make_repair,training_target,tensor_fingerprints,gradient_groups
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta

OUT=ROOT/'nonlinear_recurrent_gradient_v63'
NAMES=('nonlinear_recurrent_repair.py','nonlinear_recurrent_protocol.py','check_nonlinear_recurrent_gradient.py','run_nonlinear_recurrent_gradient.py',
       'test_nonlinear_recurrent_repair.py','nonlinear_token_readout.py','process_training_targets.py',
       'recurrent_training_protocol.py','trainable_recurrent_state.py','state_space_revisit.py',
       'runtime.py','candidate_provenance.py','evidence_encoder.py','training_pairs.py','analytic_readout_operator.py')


def main():
    OUT.mkdir(exist_ok=False)
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,**kw))
    try:
        corpus,checkpoint,heads,labels,sources=frozen_inputs()
        freeze_config(OUT/'config.json',dict(source_sha256={n:file_hash(ROOT/n) for n in NAMES},inputs=sources,
            exact_zero_routes=['loop2','no_loop'],gradient_arms=ARMS,full_remaining_velocities=14,
            loss_weights=dict(physical=.5,terminal=.5),no_optimizer=True,FLOPs=None))
        (OUT/'source').mkdir()
        for name in NAMES:shutil.copy2(ROOT/name,OUT/'source'/name)
        shutil.copy2(ROOT/'STAGE3_NONLINEAR_RECURRENT_PROTOCOL_20261008.md',OUT/'frozen_protocol.md')
        status('loading');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False);base_hash={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}
        sid=sorted(corpus.teachers)[1];state=corpus.state(sid);baseline=state['baseline_final_latent'].to(runtime.device)
        original=state['latent'].to(runtime.device).float();proof=[];initials={}
        with full_model_context(runtime):
            for arm in ARMS:
                status('checking_arm',arm=arm)
                repair=make_repair(runtime,arm,checkpoint,heads)
                repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                features,options=capture_features(runtime,state,repair.operator)
                target=training_target(runtime,corpus,state,features,repair,labels)
                initial=tensor_fingerprints(repair);initials[arm]=initial
                if arm!='trainable_loop':
                    zero=make_repair(runtime,arm,checkpoint,heads,fresh_zero=True)
                    zero.operator.condition=repair.operator.condition
                    with torch.no_grad():
                        delta=zero.latent_delta(runtime,state,features,options['modality_positions'])
                        final=replay_delta(runtime,state,delta)
                    if torch.count_nonzero(delta) or not torch.equal(final,baseline):raise ValueError('Fresh zero full endpoint not strict')
                    del zero,final,delta
                delta=repair.latent_delta(runtime,state,features,options['modality_positions'])
                final=replay_delta(runtime,state,delta,checkpoint_rollout=True)
                loss,physical,terminal=normalized_losses(delta,target['delta'],original,final,target['final'],baseline,False)
                loss.backward();groups=gradient_groups(repair)
                if groups['head']<=0 or groups['evidence']<=0:raise ValueError('Head or evidence missing full endpoint gradient')
                if arm=='trainable_loop' and min(groups['outer_core'],groups['inner_core'])<=0:raise ValueError('Missing true recurrent core gradient')
                if arm!='trainable_loop' and (groups['outer_core'] or groups['inner_core']):raise ValueError('Frozen/unused core received gradient')
                if any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base gradient')
                if tensor_fingerprints(repair)!=initial:raise ValueError('Dry-run changed parameters')
                calls={'outer':0,'inner':0} if arm=='no_loop' else {'outer':4,'inner':4}
                if repair.last_shared_calls!=calls:raise ValueError('Shared calls mismatch')
                proof.append(dict(arm=arm,state_id=sid,gradient_abs_sum=groups,actual_shared_calls=calls,
                    physical_loss=float(physical.detach()),terminal_loss=float(terminal.detach()),
                    trainable_parameters=sum(p.numel() for p in repair.trainable_parameters()),full_remaining_velocities=14))
                write_json(OUT/'arm_proofs.json',proof)
                del repair,features,options,target,delta,final,loss,physical,terminal;torch.cuda.empty_cache()
        if initials['frozen_loop']!=initials['trainable_loop']:raise ValueError('Primary matched loop initialization differs')
        if base_hash!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}:raise ValueError('Frozen base changed')
        _,_,_,_,after=frozen_inputs()
        if after!=sources:raise ValueError('Source checkpoint changed')
        status('passed',fresh_zero_endpoint_routes_exact=['loop2','no_loop'],full_gradient_arms=list(ARMS),
            matched_frozen_trainable_loop_initializations=True,source_checkpoint_unchanged=True,
            base_hash_unchanged=True,base_gradients=False,initial_tensor_hashes=initials,
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,arm_proofs=proof,
            actual_training_or_quality_result=False)
    except Exception as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
