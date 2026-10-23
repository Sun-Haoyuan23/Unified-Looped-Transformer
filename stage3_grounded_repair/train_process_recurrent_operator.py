"""Fixed100 matched nested recurrence: terminal control vs process supervision.

All arms use the same aligned residual-interface teachers, true correct pool,
initialization and sample sequence. Only auxiliary TRAIN loss differs between
joint control/process; deployment forward and parameter count stay identical.
"""
import argparse
import json
import random
from pathlib import Path
import time

import torch
from candidate_provenance import file_hash
from evidence_encoder import condition_tensor
from process_supervision import recurrent_trace,frozen_feature_normalizers,process_terms
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator,nested_loss
from runtime import Runtime,rng_restore,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta


def main():
    p=argparse.ArgumentParser()
    for name in ('work','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--study',choices=('outer_ilsd','inner_ilsd','joint_ilsd'),required=True)
    p.add_argument('--objective',choices=('terminal_control','process'),required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('No resume or checkpoint selection; new directory required')
    args.output.mkdir()
    root=args.work/'stage3_grounded_repair'
    proof=json.loads((root/'process_shared_gradient_v51/pipeline_status.json').read_text())
    if (proof['phase']!='passed' or proof['zero_terminal_budgets_exact']!=[1,2] or
            proof['optimizer_updates']!=0 or proof['base_parameter_gradients'] or
            not proof['shared_core_hash_unchanged'] or
            min(proof['initially_zero_readout_aux_core_gradient_abs_sum'].values())<=0):
        raise ValueError('Require actual process/nested core gradient and zero terminal proof')
    for name,digest in json.loads((root/'process_shared_gradient_v51/config.json').read_text())['configuration']['source_sha256'].items():
        if file_hash(root/name)!=digest:raise ValueError('Proven process interface changed: '+name)
    corpus=ProcessCorpus(root)
    runtime=Runtime(args.work,vae=False)
    torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
    op=build_operator(runtime,args.study)
    initial={k:tensor_hash(v) for k,v in op.adapter_state().items()}
    old=json.loads((root/'recurrent_train_v46'/args.study/'training_config.json').read_text())
    if initial!=old['adapter_initialization_hashes']:
        raise ValueError('Actual adapter initialization differs from frozen matched v46')
    torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
    zero=build_operator(runtime,args.study)
    if {k:tensor_hash(v) for k,v in zero.adapter_state().items()}!=initial:
        raise ValueError('Feature normalization operator must be the same fixed initial model')
    optimizer=torch.optim.AdamW(op.trainable_parameters(),lr=1e-4,weight_decay=0)
    core_hash=op.core_hashes();base=list(runtime.model.parameters())
    rng=random.Random(20261004);keys=sorted(corpus.teachers)
    names=('train_process_recurrent_operator.py','process_supervision.py','process_training_targets.py',
        'trainable_recurrent_state.py','recurrent_training_protocol.py','state_space_revisit.py',
        'runtime.py','evidence_encoder.py','training_pairs.py','candidate_provenance.py')
    config=dict(work=str(args.work),output=str(args.output),study=args.study,objective=args.objective,
        operator_kind='recurrent_'+args.study,steps=100,width=64,level=2,window=1,lr=1e-4,
        maximum_recurrent_loops=2,low_rank_rank=4,adapter_initialization_seed=20261005,
        adapter_initialization_hashes=initial,state_sampling_seed=20261004,preservation_every=3,
        training_gate=corpus.gate,corpus_input_sha256=corpus.input_hashes,
        original_baseline_quality_deltas=corpus.original_baseline_deltas,
        paired_report_sha256=file_hash(root/'repair_train_targets_v36/paired_report.json'),
        aligned_teacher_report_sha256=file_hash(root/'teacher_additive_reachability_v50/paired_report.json'),
        training_code_sha256={n:file_hash(root/n) for n in names},
        process_loss_weights=dict(state=1.,hidden=1.) if args.objective=='process' else dict(state=0.,hidden=0.),
        feature_normalizers='frozen initial model, per state/branch/prefix; teacher targets detached',
        terminal_objective='same nested terminal/rank/linear ILSD schedule in all four arms',
        teacher_interface='v50 scored original+float32(teacher-original), not direct-v49 replacement',
        checkpoint_selection='fixed final step100 before TRAIN diagnostic; no val/controller',
        runtime=runtime.metadata(),trainable_parameters=sum(q.numel() for q in op.trainable_parameters()),
        inference_forward_unchanged=True,added_process_parameters=0,
        post_update_prefix_or_KV_used=False,controller=False,FLOPs=None)
    write_json(args.output/'training_config.json',config)
    features_cache={};targets={};normalizers={}
    with full_model_context(runtime),(args.output/'train.jsonl').open('w') as log:
        for step in range(1,101):
            preserve=step%3==0
            sid=rng.choice(corpus.correct if preserve else keys)
            state=corpus.state(sid)
            op.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
            cache_hit=sid in features_cache
            if not cache_hit:
                features_cache[sid],_=capture_features(runtime,state,op)
                targets[sid]=corpus.target(runtime,state,features_cache[sid],op,preserve)
                zero.condition=op.condition.detach()
                with torch.no_grad(),recurrent_trace(zero) as initial_trace:
                    zero_deltas=zero.latent_deltas(runtime,features_cache[sid],runtime.inputs(state['prompt'])['modality_positions'],
                        state['latent'].to(runtime.device),(1,2))
                normalizers[sid]=frozen_feature_normalizers(initial_trace,targets[sid]['features'],runtime.inputs(state['prompt'])['modality_positions'])
                if step==1:
                    with torch.no_grad():
                        for budget in (1,2):
                            if not torch.equal(replay_delta(runtime,state,zero_deltas[budget]),state['baseline_final_latent'].to(runtime.device)):
                                raise ValueError('Training zero terminal is not strictly identical')
            rng_restore(state['rng']);optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
            original=state['latent'].to(runtime.device).float()
            baseline=state['baseline_final_latent'].to(runtime.device).float()
            positions=runtime.inputs(state['prompt'])['modality_positions']
            with recurrent_trace(op) as trace:
                deltas=op.latent_deltas(runtime,features_cache[sid],positions,original,(1,2))
            terms=process_terms(deltas,trace,original,targets[sid]['input'],targets[sid]['features'],
                positions,preserve,normalizers[sid])
            high=replay_delta(runtime,state,deltas[2],checkpoint_rollout=True)
            low=replay_delta(runtime,state,deltas[1],checkpoint_rollout=True)
            terminal,parts=nested_loss(high,low,targets[sid]['final'],baseline,preserve,step,100)
            loss=terminal+(terms['state']+terms['hidden'] if args.objective=='process' else 0.)
            loss.backward()
            if any(q.grad is not None for q in base):raise ValueError('Frozen base receives gradient')
            grad_norm=torch.nn.utils.clip_grad_norm_(op.trainable_parameters(),1.)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad_norm)):
                raise ValueError('Nonfinite process training loss or gradient')
            optimizer.step();torch.cuda.synchronize()
            row=dict(step=step,state_id=sid,preservation_step=preserve,loss=float(loss.detach()),
                terminal_loss=float(terminal.detach()),state_loss=float(terms['state'].detach()),
                feature_loss=float(terms['hidden'].detach()),grad_norm=float(grad_norm),
                low_rank_gradient_abs_sum=sum(float(q.grad.abs().sum()) for name,q in op.named_parameters()
                    if 'low_rank' in name and q.grad is not None),
                shared_evidence_gradient_abs_sum=sum(float(q.grad.abs().sum()) for q in op.encoder.parameters() if q.grad is not None),
                pre_feedback_features_cached=cache_hit,post_update_prefix_or_KV_used=False,
                feature_scales={n:[float(q) for q in values] for n,values in normalizers[sid].items()},
                loss_weights=config['process_loss_weights'],remaining_velocity_calls_per_update=28,
                wall_time_s=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
                FLOPs=None)
            log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps(row),flush=True)
        if op.core_hashes()!=core_hash:raise ValueError('Frozen shared base weights changed')
        torch.save(dict(operator=op.adapter_state(),metadata=config,step=100,level=2,width=64),args.output/'operator_step100.pt')
        write_json(args.output/'training_complete.json',dict(steps=100,optimizer_updates=100,
            outer_core_exact=True,base_parameter_gradients=False,checkpoint=str(args.output/'operator_step100.pt'),
            successful_teacher_states=3,preservation_pool=35,objective=args.objective,
            shared_base_core_hash_unchanged=True,validation_images=0,controller_started=False))


if __name__=='__main__':main()
