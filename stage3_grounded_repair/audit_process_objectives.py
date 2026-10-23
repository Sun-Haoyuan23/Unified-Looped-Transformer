"""Read-only final checkpoint diagnosis: alignment and component gradient geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from candidate_provenance import file_hash
from evidence_encoder import condition_tensor
from process_supervision import recurrent_trace, frozen_feature_normalizers, process_terms, normalized_visual
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator, nested_loss
from runtime import Runtime, tensor_hash, write_json
from state_space_revisit import full_model_context, capture_features
from trainable_recurrent_state import replay_delta


def gradient_geometry(component_grads, names, objective):
    groups={'all':list(range(len(names)))}
    for i,name in enumerate(names):
        if 'low_rank' in name:
            group=name.split('.')[0]+'_low_rank_core'
        elif '.readout.' in name:group='latent_readout'
        elif '.feedback.' in name:group='hidden_feedback'
        elif '.encoder.' in name:group='evidence_encoder'
        else:group='token_position_projection'
        groups.setdefault(group,[]).append(i)
    result={}
    for group,indices in groups.items():
        vectors={term:torch.cat([grads[i].detach().float().cpu().reshape(-1) for i in indices])
                 for term,grads in component_grads.items()}
        squared={term:float(v.double().square().sum()) for term,v in vectors.items()}
        pairs={}
        for a,b in [('terminal','state'),('terminal','hidden'),('state','hidden')]:
            dot=float((vectors[a].double()*vectors[b].double()).sum())
            denom=(squared[a]*squared[b])**.5
            pairs[a+'__'+b]=dict(dot=dot,cosine=dot/denom if denom>0 else None)
        direction=vectors['terminal'].clone()
        if objective=='process':direction+=vectors['state']+vectors['hidden']
        result[group]=dict(squared_gradient_norms=squared,pairs=pairs,
            directional_derivative_along_actual_negative_total_gradient={
                term:-float((v.double()*direction.double()).sum()) for term,v in vectors.items()})
    return result


def inspect_state(runtime,corpus,state,op,zero,config,records):
    sid=state['state_id'];preserve=sid in corpus.correct
    op.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
    zero.condition=op.condition
    features,options=capture_features(runtime,state,op)
    target=corpus.target(runtime,state,features,op,preserve)
    positions=options['modality_positions'];original=state['latent'].to(runtime.device).float()
    baseline=state['baseline_final_latent'].to(runtime.device).float()
    with torch.no_grad(),recurrent_trace(zero) as initial:
        zero.latent_deltas(runtime,features,positions,original,(1,2))
    scales=frozen_feature_normalizers(initial,target['features'],positions)
    with recurrent_trace(op) as trace:
        deltas=op.latent_deltas(runtime,features,positions,original,(1,2))
    terms=process_terms(deltas,trace,original,target['input'],target['features'],positions,preserve,scales)
    high=replay_delta(runtime,state,deltas[2],checkpoint_rollout=True)
    low=replay_delta(runtime,state,deltas[1],checkpoint_rollout=True)
    for budget,final in [(1,low),(2,high)]:
        row=records[(sid,'conditional_budget1' if budget==1 else 'conditional_revisit')]
        if file_hash(row['final_latent'])!=row['final_latent_sha256']:
            raise ValueError('v53 final bytes changed')
        saved=torch.load(row['final_latent'],map_location=runtime.device,weights_only=False)['latent']
        if not torch.equal(final.detach(),saved):raise ValueError('v53 endpoint not strictly reproduced')
    terminal,detail=nested_loss(high,low,target['final'],baseline,preserve,100,100)
    losses=dict(terminal=terminal,state=terms['state'],hidden=terms['hidden'])
    named=[(n,p) for n,p in op.named_parameters() if '.shared_core.' not in n and p.requires_grad]
    grads={}
    for term,loss in losses.items():
        values=torch.autograd.grad(loss,[p for n,p in named],retain_graph=True,allow_unused=True)
        grads[term]=[torch.zeros_like(p) if g is None else g.detach() for (n,p),g in zip(named,values)]
        if any(not torch.isfinite(g).all() for g in grads[term]):raise ValueError('Nonfinite diagnostic gradient')
    physical={};hidden={};gold=(target['input']-original).detach()
    for budget in (1,2):
        delta=deltas[budget].detach().float();gn=float(gold.square().mean());dn=float(delta.square().mean())
        denom=(gn*dn)**.5
        physical[str(budget)]=dict(predicted_delta_rms=dn**.5,teacher_delta_rms=gn**.5,
            delta_to_teacher_rms_ratio=(dn/gn)**.5 if gn>0 else None,
            delta_teacher_cosine=float((delta*gold).mean())/denom if denom>0 else None,
            physical_relative_target_mse=float((original+delta-target['input']).square().mean()/
                (original.square().mean() if preserve else gold.square().mean()).clamp_min(1e-5)))
    for branch,states in trace.items():
        target_hidden=normalized_visual(target['features'][branch],positions)
        hidden[branch]=[dict(normalized_hidden_mse=float(F.mse_loss(normalized_visual(h,positions),target_hidden).detach()),
            initial_normalizer=float(scales[branch][i]),
            relative_hidden_mse=float((F.mse_loss(normalized_visual(h,positions),target_hidden)/scales[branch][i]).detach()))
            for i,h in enumerate(states)]
    return dict(state_id=sid,split='train',preservation=preserve,endpoint_budgets_strictly_reproduced=[1,2],
        losses={k:float(v.detach()) for k,v in losses.items()},physical=physical,hidden=hidden,
        gradient_geometry=gradient_geometry(grads,[n for n,p in named],config['objective']),
        target_and_checkpoint_used_only_for_training_diagnosis=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True)
    args=parser.parse_args();work=Path('/home/shy_23/iterative_visual_stage1_20261003');root=work/'stage3_grounded_repair'
    out=root/'process_objective_audit_v54'/args.arm;out.mkdir()
    checkpoint=root/'process_recurrent_train_v52'/args.arm/'operator_step100.pt'
    fit=root/'process_recurrent_trainfit_v53';config=json.loads((checkpoint.parent/'training_config.json').read_text())
    def status(phase,**kwargs):write_json(out/'status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,**kwargs))
    try:
        status('loading');corpus=ProcessCorpus(root);runtime=Runtime(work,vae=False)
        torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        op=build_operator(runtime,config['study'])
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['metadata']!=config or payload['step']!=100:raise ValueError('Require fixed final100 checkpoint')
        op.load_state_dict(payload['operator'],strict=False)
        torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        zero=build_operator(runtime,config['study'])
        if {k:tensor_hash(v) for k,v in zero.adapter_state().items()}!=config['adapter_initialization_hashes']:
            raise ValueError('Untrained normalizer initialization differs')
        initial_adapter={k:tensor_hash(v) for k,v in op.adapter_state().items()};core=op.core_hashes();digest=file_hash(checkpoint)
        records={(q['state_id'],q['action']):q for q in map(json.loads,(fit/args.arm/'all_actions.jsonl').read_text().splitlines())}
        cohort=json.loads((fit/'cohort.json').read_text());results=[]
        start=time.perf_counter()
        with full_model_context(runtime):
            for sid in cohort['state_ids']:
                status('inspecting',state_id=sid,completed_states=len(results))
                row=inspect_state(runtime,corpus,corpus.state(sid),op,zero,config,records)
                results.append(row);write_json(out/'results.json',results)
                if any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base gradient populated')
                if any(p.grad is not None for p in op.trainable_parameters()):raise ValueError('Read-only autograd populated adapter gradients')
        if op.core_hashes()!=core or {k:tensor_hash(v) for k,v in op.adapter_state().items()}!=initial_adapter or file_hash(checkpoint)!=digest:
            raise ValueError('Read-only diagnostic changed parameters/checkpoint')
        status('complete',states=6,strict_endpoint_replays=12,checkpoint_sha256=digest,base_gradients=False,
            adapter_gradients_populated=False,base_and_adapter_hashes_unchanged=True,time_s=time.perf_counter()-start,
            peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,FLOPs=None)
    except Exception as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
