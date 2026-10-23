"""Read-only fixed v64 objectives, spatial residuals and effective core audit."""
import json,shutil,time
import torch
from torch.nn import functional as F
from analytic_readout_operator import load_adapter
from audit_condition_separation import flip_evidence
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from nonlinear_recurrent_protocol import ROOT,WORK,frozen_inputs,make_repair,training_target,tensor_fingerprints
from nonlinear_recurrent_repair import normalized_losses
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta

OUT=ROOT/'nonlinear_recurrent_mapping_audit_v65'
ARMS=('frozen_loop','trainable_loop')
NAMES=('audit_nonlinear_recurrent_mapping.py','run_nonlinear_recurrent_mapping_audit.py',
 'nonlinear_recurrent_repair.py','nonlinear_recurrent_protocol.py','nonlinear_token_readout.py',
 'audit_condition_separation.py','analytic_readout_operator.py','process_training_targets.py',
 'recurrent_training_protocol.py','trainable_recurrent_state.py','state_space_revisit.py',
 'runtime.py','candidate_provenance.py','evidence_encoder.py','training_pairs.py')

def vectors(names,params,grads,core):
    values=[(torch.zeros_like(p) if g is None else g).detach().float().cpu().flatten()
            for n,p,g in zip(names,params,grads) if ('.low_rank.' in n)==core]
    return torch.cat(values) if values else torch.empty(0)

def gram_report(ids,values):
    x=torch.stack(values).double();gram=x@x.T
    norms=gram.diag().clamp_min(0).sqrt();den=norms[:,None]*norms[None,:]
    cosine=torch.where(den>0,gram/den.clamp_min(1e-30),torch.zeros_like(gram))
    return dict(state_ids=ids,dot=gram.tolist(),l2=norms.tolist(),cosine=cosine.tolist(),
                flattened_gradient_sha256=[tensor_hash(v) for v in values])

def residual_summary(pred,gold):
    a=pred[:1].detach().float();b=gold[:1].detach().float();err=a-b
    energy=err.square().mean(1).flatten();count=energy.numel();k=max(1,(count+9)//10)
    top=float(energy.topk(k).values.sum()/energy.sum().clamp_min(1e-30))
    denom=b.square().mean().sqrt();return dict(prediction_rms=float(a.square().mean().sqrt()),
      target_rms=float(denom),residual_rms=float(err.square().mean().sqrt()),
      prediction_target_cosine=float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0)) if denom else None,
      residual_energy_in_top10percent_spatial_cells=top)

def core_changes(repair,initial):
    rows=[]
    with torch.no_grad():
        for level in ('outer','inner'):
            branch=getattr(repair.operator,level)
            for i,targets in enumerate(branch.targets):
                weights=dict(branch.shared_core[i].named_parameters())
                for name,key in targets.items():
                    oldA=initial[f'{level}.low_rank.{key}.A'].to(weights[name].device)
                    oldB=initial[f'{level}.low_rank.{key}.B'].to(weights[name].device)
                    before=oldB@oldA;after=branch.low_rank[key].delta();w=weights[name]
                    eb=w+before.to(w.dtype);ea=w+after.to(w.dtype)
                    rows.append(dict(level=level,block=i,projection=name,
                       storage_dtype=str(w.dtype),elements=w.numel(),
                       effective_elements_changed=int(torch.count_nonzero(eb!=ea)),
                       effective_fraction_changed=float((eb!=ea).float().mean()),
                       effective_relative_rms_change=float((ea.float()-eb.float()).square().mean().sqrt()/eb.float().square().mean().sqrt()),
                       lowrank_relative_rms_change=float((after-before).square().mean().sqrt()/before.square().mean().sqrt().clamp_min(1e-30))))
    return rows

def main():
    OUT.mkdir(exist_ok=False)
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),
        optimizer_updates=0,new_scoring_images=0,analytical_fits=0,controller_started=False,**kw))
    try:
        prior=ROOT/'nonlinear_recurrent_train_v64'
        if json.loads((prior/'pipeline_status.json').read_text())['phase']!='complete':raise ValueError('v64 must be complete')
        corpus,op_path,heads,labels,inputs=frozen_inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct))
        cohort_path=ROOT/'condition_separation_audit_v61/config.json'
        cohort=json.loads(cohort_path.read_text())['configuration'];focus=cohort['state_ids']
        if len(focus)!=6 or not set(focus)<=set(ids):raise ValueError('Require same metadata-selected six TRAIN states')
        checks={a:prior/a/'repair_step210.pt' for a in ARMS}
        config=dict(source_sha256={n:file_hash(ROOT/n) for n in NAMES},inputs=inputs,
            checkpoint_sha256={a:file_hash(p) for a,p in checks.items()},
            capacity_report_sha256={a:file_hash(prior/a/'capacity_report.json') for a in ARMS},
            state_ids=ids,full_endpoint_state_ids=focus,cohort_config_sha256=file_hash(cohort_path),
            pairs=cohort['pairs'],controls=cohort['controls'],
            protocol='fixed step210; all38 physical gradients; metadata-selected6 terminal gradients and reversed evidence',
            reference_tolerance=dict(rtol=1e-4,atol=1e-6),FLOPs=None)
        freeze_config(OUT/'config.json',config);(OUT/'source').mkdir()
        for n in NAMES:shutil.copy2(ROOT/n,OUT/'source'/n)
        shutil.copy2(ROOT/'STAGE3_NONLINEAR_MAPPING_AUDIT_PROTOCOL_20261008.md',OUT/'frozen_protocol.md')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('loading');runtime=Runtime(WORK,vae=False)
        base={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}
        states={};objectives=[];records=[];core=[];endpoints={};gram={}
        for arm in ARMS:
            status('auditing_arm',arm=arm)
            repair=make_repair(runtime,arm,op_path,heads)
            initial=repair.operator.adapter_state()
            payload=torch.load(checks[arm],map_location=runtime.device,weights_only=False)
            if payload['step']!=210 or payload['arm']!=arm:raise ValueError('Wrong final checkpoint')
            load_adapter(repair.operator,payload['operator']);repair.head.load_state_dict(payload['head'])
            frozen=tensor_fingerprints(repair)
            # Core comparison is against the actual v64 warm start, before any new audit.
            core.extend([dict(arm=arm,**q) for q in core_changes(repair,initial)])
            refs={q['state_id']:q for q in json.loads((prior/arm/'capacity_report.json').read_text())['states']}
            named=[(n,p) for n,p in repair.named_parameters() if p.requires_grad];names=[n for n,p in named];params=[p for n,p in named]
            p_common=[];p_core=[];t_common=[];t_core=[];c_common=[];c_core=[]
            with full_model_context(runtime):
                for sid in ids:
                    status('auditing_state',arm=arm,state_id=sid)
                    state=corpus.state(sid);preserve=sid in corpus.correct
                    repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                    features,options=capture_features(runtime,state,repair.operator)
                    target=training_target(runtime,corpus,state,features,repair,labels)
                    original=state['latent'].to(runtime.device).float();baseline=state['baseline_final_latent'].to(runtime.device).float()
                    delta=repair.latent_delta(runtime,state,features,options['modality_positions'])
                    scale=(original.square().mean() if preserve else target['delta'].square().mean()).clamp_min(1e-5)
                    physical=(delta-target['delta']).square().mean()/scale
                    expected=refs[sid]['physical_relative_target_mse']
                    if abs(float(physical)-expected)>1e-6+1e-4*abs(expected):raise ValueError('v64 physical reproduction failed')
                    gp=torch.autograd.grad(physical,params,retain_graph=sid in focus,allow_unused=True)
                    pc=vectors(names,params,gp,False);pk=vectors(names,params,gp,True);p_common.append(pc);p_core.append(pk)
                    row=dict(arm=arm,state_id=sid,preservation=preserve,expected_mse=expected,
                        reproduced_mse=float(physical.detach()),residual=residual_summary(delta,target['delta']))
                    if sid in focus:
                        final=replay_delta(runtime,state,delta,checkpoint_rollout=True)
                        _,_,terminal=normalized_losses(delta,target['delta'],original,final,target['final'],baseline,preserve)
                        gt=torch.autograd.grad(terminal,params+[delta],allow_unused=True)
                        tc=vectors(names,params,gt[:-1],False);tk=vectors(names,params,gt[:-1],True)
                        t_common.append(tc);t_core.append(tk);c_common.append(.5*(pc+tc));c_core.append(.5*(pk+tk))
                        pd=2*(delta.detach()-target['delta'])/delta.numel()/scale
                        td=gt[-1].detach()
                        pair=lambda a,b:dict(physical_l2=float(a.norm()),terminal_l2=float(b.norm()),
                           cosine=float(F.cosine_similarity(a.double(),b.double(),dim=0)) if a.numel() and a.norm() and b.norm() else None)
                        objectives.append(dict(arm=arm,state_id=sid,preservation=preserve,
                           physical_loss=float(physical.detach()),terminal_loss=float(terminal.detach()),
                           common_parameters=pair(pc,tc),core_parameters=pair(pk,tk),
                           physical_state=pair(pd.flatten().cpu(),td.flatten().cpu()),
                           endpoint_sha256=tensor_hash(final),full_remaining_velocities=14,post_update_prefix_or_KV_used=False))
                        endpoints[arm+'/'+sid]=final.detach().cpu()
                        with torch.no_grad():
                            repair.operator.condition=condition_tensor(flip_evidence(state['evidence']),state['true_t'],runtime.device)
                            reverse=repair.latent_delta(runtime,state,features,options['modality_positions'])
                            row['opposite_evidence_mse']=float((reverse-target['delta']).square().mean()/scale)
                            row['opposite_evidence_delta_relative_change']=float((reverse-delta.detach()).square().mean().sqrt()/delta.detach().square().mean().sqrt().clamp_min(1e-30))
                        del final,terminal,gt,pd,td
                    records.append(row)
                    del features,options,target,delta,physical,gp
            gram[arm]=dict(physical_common=gram_report(ids,p_common),physical_core=gram_report(ids,p_core),
                terminal_common=gram_report(focus,t_common),terminal_core=gram_report(focus,t_core),
                combined_common=gram_report(focus,c_common),combined_core=gram_report(focus,c_core))
            if tensor_fingerprints(repair)!=frozen:raise ValueError('Read-only audit changed repair parameters')
            if any(p.grad is not None for p in repair.parameters()):raise ValueError('autograd.grad must leave parameter .grad unset')
            del repair,payload,initial;torch.cuda.empty_cache()
        if base!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()} or any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base changed or gradient')
        if {a:file_hash(p) for a,p in checks.items()}!=config['checkpoint_sha256'] or frozen_inputs()[-1]!=inputs:raise ValueError('Frozen sources changed')
        write_json(OUT/'state_results.json',records);write_json(OUT/'objective_results.json',objectives)
        write_json(OUT/'gradient_grams.json',gram);write_json(OUT/'core_changes.json',core)
        torch.save(endpoints,OUT/'diagnostic_endpoints.pt')
        status('complete',physical_reproductions=len(records),full_endpoint_diagnostics=len(objectives),
            base_and_adapter_hashes_unchanged=True,no_parameter_grad_buffers=True,
            source_checkpoint_unchanged=True,no_new_quality_or_semantic_measurement=True)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
