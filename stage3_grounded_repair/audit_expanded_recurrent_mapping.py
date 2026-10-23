"""Read-only v72 all41 residuals, six teacher objectives, and actual cores."""
import json,shutil,tarfile,time
import torch
from torch.nn import functional as F
from analytic_readout_operator import load_adapter
from audit_condition_separation import flip_evidence
from audit_nonlinear_recurrent_mapping import vectors,gram_report,core_changes
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_recurrent_protocol import ROOT,WORK,NAMES as PRIOR_NAMES,proof_inputs,make_repair,target_on_device
from nonlinear_recurrent_protocol import tensor_fingerprints
from nonlinear_recurrent_repair import ARMS,normalized_losses
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
from trainable_recurrent_state import replay_delta

OUT=ROOT/'expanded_recurrent_mapping_audit_v73'
PROTOCOL='STAGE3_EXPANDED_MAPPING_AUDIT_PROTOCOL_20261009.md'
NAMES=tuple(dict.fromkeys(PRIOR_NAMES+('audit_expanded_recurrent_mapping.py','run_expanded_recurrent_mapping_audit.py',
    'audit_nonlinear_recurrent_mapping.py','audit_condition_separation.py')))


def residual_summary(pred,gold):
    a=pred[:1].detach().double();b=gold[:1].detach().double();err=a-b
    energy=err.square().mean(1).flatten();count=energy.numel();k=max(1,(count+9)//10)
    aa=a.square().mean();bb=b.square().mean();ab=(a*b).mean()
    result=dict(prediction_rms=float(aa.sqrt()),target_rms=float(bb.sqrt()),residual_rms=float(err.square().mean().sqrt()),
        residual_energy_top10percent_spatial=float(energy.topk(k).values.sum()/energy.sum().clamp_min(1e-30)))
    if bb>0:
        scale=ab/aa if aa>0 else aa
        result.update(prediction_target_cosine=float(ab/(aa*bb).sqrt()) if aa>0 else None,
            best_scalar_diagnostic=float(scale),best_scalar_relative_mse=float((scale*a-b).square().mean()/bb))
    return result


def pair(a,b):
    a=a.detach().double();b=b.detach().double();na=a.norm();nb=b.norm()
    return dict(physical_l2=float(na),terminal_l2=float(nb),dot=float(a@b),
                cosine=float((a@b)/(na*nb)) if a.numel() and na>0 and nb>0 else None)


def cross_report(left_ids,left,right_ids,right):
    a=torch.stack(left).double();b=torch.stack(right).double();dot=a@b.T
    den=a.norm(dim=1)[:,None]*b.norm(dim=1)[None,:]
    cosine=torch.where(den>0,dot/den.clamp_min(1e-30),torch.zeros_like(dot))
    return dict(physical_ids=left_ids,terminal_ids=right_ids,dot=dot.tolist(),cosine=cosine.tolist())


def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter()
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),
        optimizer_updates=0,new_scoring_images=0,analytical_fits=0,controller_started=False,**kw))
    try:
        prior=ROOT/'expanded_recurrent_train_v72'
        if json.loads((prior/'pipeline_status.json').read_text())['phase']!='complete':raise ValueError('v72 incomplete')
        portable_path=ROOT/'v72_complete_portable_verification.json'
        portable=json.loads(portable_path.read_text());archive=ROOT/'transfers/stage3_expanded_v72_complete.tgz'
        if not portable['passed'] or portable['optimizer_updates']!=1260 or file_hash(archive)!=portable['archive_sha256']:
            raise ValueError('Completed portable proof/archive mismatch')
        with tarfile.open(archive) as stream:
            manifest=json.load(stream.extractfile('expanded_recurrent_train_v72/artifact_manifest.json'))
        if json.loads((prior/'artifact_manifest.json').read_text())!=manifest:raise ValueError('v72 manifest changed')
        for name,digest in manifest.items():
            if file_hash(prior/name)!=digest:raise ValueError('Archived v72 input changed: '+name)
        corpus,op_path,rms,proof=proof_inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct));teachers=sorted(corpus.teachers)
        if len(ids)!=41 or len(teachers)!=6:raise ValueError('Expanded TRAIN cohort changed')
        checks={a:prior/a/'repair_step420.pt' for a in ARMS}
        config=dict(source_sha256={n:file_hash(ROOT/n) for n in NAMES},protocol_sha256=file_hash(ROOT/PROTOCOL),
            checkpoint_sha256={a:file_hash(p) for a,p in checks.items()},
            capacity_report_sha256={a:file_hash(prior/a/'capacity_report.json') for a in ARMS},
            corpus_config_sha256=file_hash(ROOT/'expanded_process_corpus_v70/config.json'),
            v72_portable_proof_sha256=file_hash(portable_path),v72_archive_sha256=portable['archive_sha256'],
            v72_manifest_sha256=file_hash(prior/'artifact_manifest.json'),state_ids=ids,teacher_ids=teachers,
            state_sha256={s:file_hash(ROOT/'states/repair_train_targets_v36'/(s+'.pt')) for s in ids},
            reference_tolerance=dict(rtol=1e-4,atol=1e-6),physical_gradient_states=41,terminal_gradient_states=6,
            preservation_terminal_gradients_measured=False,physical_scale='v72 normalized_losses',
            no_optimizer_no_fits_no_scored_images=True,FLOPs=None)
        freeze_config(OUT/'config.json',config);(OUT/'source').mkdir()
        for n in NAMES:shutil.copy2(ROOT/n,OUT/'source'/n)
        shutil.copy2(ROOT/PROTOCOL,OUT/'frozen_protocol.md')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('loading');runtime=Runtime(WORK,vae=False);torch.cuda.reset_peak_memory_stats()
        base={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}
        records=[];objectives=[];core=[];grams={};tensors={};actual_shared={'outer':0,'inner':0}
        nominal_velocity_calls=0
        for arm in ARMS:
            status('auditing_arm',arm=arm)
            repair=make_repair(runtime,arm,op_path,rms);initial=repair.operator.adapter_state()
            payload=torch.load(checks[arm],map_location=runtime.device,weights_only=False)
            if payload['step']!=420 or payload['arm']!=arm:raise ValueError('Fixed checkpoint mismatch')
            load_adapter(repair.operator,payload['operator']);repair.head.load_state_dict(payload['head'])
            frozen=tensor_fingerprints(repair);core.extend([dict(arm=arm,**q) for q in core_changes(repair,initial)])
            refs={q['state_id']:q for q in json.loads((prior/arm/'capacity_report.json').read_text())['states']}
            named=[(n,p) for n,p in repair.named_parameters() if p.requires_grad];names=[n for n,p in named];params=[p for n,p in named]
            teacher_physical={False:[],True:[]};teacher_terminal={False:[],True:[]};preserve_sum={False:None,True:None}
            with full_model_context(runtime):
                for sid in ids:
                    status('auditing_state',arm=arm,state_id=sid)
                    state=corpus.state(sid);target=target_on_device(corpus,state,runtime.device);preserve=target['preservation']
                    repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                    features,options=capture_features(runtime,state,repair.operator);nominal_velocity_calls+=1
                    original=state['latent'].to(runtime.device).float();baseline=state['baseline_final_latent'].to(runtime.device).float()
                    delta=repair.latent_delta(runtime,state,features,options['modality_positions'])
                    for level in actual_shared:actual_shared[level]+=repair.last_shared_calls[level]
                    scale=(original.square().mean() if preserve else target['delta'].square().mean()).clamp_min(1e-5)
                    physical=(delta-target['delta']).square().mean()/scale;expected=refs[sid]['physical_relative_target_mse']
                    if abs(float(physical)-expected)>1e-6+1e-4*abs(expected):raise ValueError('v72 MSE reproduction failed')
                    gp=torch.autograd.grad(physical,params,retain_graph=not preserve,allow_unused=True)
                    pv={flag:vectors(names,params,gp,flag) for flag in (False,True)}
                    for flag,v in pv.items():
                        if preserve:preserve_sum[flag]=v.clone() if preserve_sum[flag] is None else preserve_sum[flag]+v
                        else:teacher_physical[flag].append(v)
                    row=dict(arm=arm,state_id=sid,preservation=preserve,expected_mse=expected,reproduced_mse=float(physical.detach()),
                        source_rms=float(original.square().mean().sqrt()),residual=residual_summary(delta,target['delta']))
                    if not preserve:
                        final=replay_delta(runtime,state,delta,checkpoint_rollout=True);nominal_velocity_calls+=14
                        _,_,terminal=normalized_losses(delta,target['delta'],original,final,target['final'],baseline,False)
                        gt=torch.autograd.grad(terminal,params+[delta],allow_unused=True)
                        tv={flag:vectors(names,params,gt[:-1],flag) for flag in (False,True)}
                        for flag,v in tv.items():teacher_terminal[flag].append(v)
                        pd=2*(delta.detach()-target['delta'])/delta.numel()/scale;td=gt[-1].detach()
                        objectives.append(dict(arm=arm,state_id=sid,physical_loss=float(physical.detach()),terminal_loss=float(terminal.detach()),
                            common_parameters=pair(pv[False],tv[False]),core_parameters=pair(pv[True],tv[True]),
                            physical_state=pair(pd.flatten().cpu(),td.flatten().cpu()),endpoint_sha256=tensor_hash(final),
                            full_remaining_velocities=14,post_update_prefix_or_KV_used=False))
                        tensors[arm+'/'+sid]=dict(delta=delta.detach().cpu(),final=final.detach().cpu())
                        with torch.no_grad():
                            repair.operator.condition=condition_tensor(flip_evidence(state['evidence']),state['true_t'],runtime.device)
                            reverse=repair.latent_delta(runtime,state,features,options['modality_positions'])
                            for level in actual_shared:actual_shared[level]+=repair.last_shared_calls[level]
                            row['opposite_evidence_mse']=float((reverse-target['delta']).square().mean()/scale)
                            row['opposite_evidence_update_relative_change']=float((reverse-delta.detach()).square().mean().sqrt()/delta.detach().square().mean().sqrt().clamp_min(1e-30))
                        del final,terminal,gt,pd,td,tv,reverse
                    records.append(row);del features,options,target,delta,physical,gp,pv
            arm_grams={}
            for flag,label in ((False,'common'),(True,'core')):
                p=teacher_physical[flag]+[preserve_sum[flag]/35];labels=teachers+['mean35_preservation_physical']
                t=teacher_terminal[flag]
                arm_grams[label]=dict(physical=gram_report(labels,p),teacher_terminal=gram_report(teachers,t),
                    physical_vs_teacher_terminal=cross_report(labels,p,teachers,t),
                    equal_teacher_combined=gram_report(teachers,[.5*(a+b) for a,b in zip(p[:6],t)]))
            grams[arm]=arm_grams
            if tensor_fingerprints(repair)!=frozen or any(p.grad is not None for p in repair.parameters()):raise ValueError('Read-only adapter mutation/grad buffer')
            del repair,payload,initial,teacher_physical,teacher_terminal,preserve_sum,arm_grams;torch.cuda.empty_cache()
        if base!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()} or any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base mutation/gradient')
        if {a:file_hash(p) for a,p in checks.items()}!=config['checkpoint_sha256']:raise ValueError('Checkpoint bytes changed')
        proof_inputs()
        for n,h in config['source_sha256'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('Frozen audit source changed')
        if nominal_velocity_calls!=375:raise ValueError('Diagnostic velocity count differs')
        write_json(OUT/'state_results.json',records);write_json(OUT/'objective_results.json',objectives)
        write_json(OUT/'gradient_grams.json',grams);write_json(OUT/'core_changes.json',core)
        torch.save(tensors,OUT/'diagnostic_tensors.pt')
        status('complete',physical_reproductions=len(records),full_endpoint_diagnostics=len(objectives),
            base_and_adapter_hashes_unchanged=True,no_parameter_grad_buffers=True,source_checkpoint_unchanged=True,
            no_new_quality_or_semantic_measurement=True,nominal_velocity_calls=nominal_velocity_calls,
            actual_shared_calls=actual_shared,backward_checkpoint_recomputation_additional=True,
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,diagnostic_process_walltime_seconds=time.perf_counter()-started,FLOPs=None)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
