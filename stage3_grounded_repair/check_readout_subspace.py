"""v82: two fixed v81 readout-subspace fits, no optimizer or model mutation."""
import json,math,shutil,time
import numpy as np
from PIL import Image
import torch
from analytic_readout_operator import load_adapter
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from expanded_recurrent_protocol import target_on_device
from nonlinear_recurrent_protocol import tensor_fingerprints
from nonlinear_token_readout import patch_delta
from process_four_loop_protocol import ROOT,WORK,CHECK,verified_training_inputs,aligned_hidden
from reference_recomputed_repair import build_reference
from train_reference_recomputed import TRAIN,ARMS,NAMES as PRIOR_NAMES
from check_process_alignment import count_shared,relative_physical
from readout_subspace_solver import weighted_block,solve_fixed_subspace,patch_target_from_permutation
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context
OUT=ROOT/'readout_subspace_audit_v82';PROTOCOL='STAGE3_READOUT_SUBSPACE_PROTOCOL_20261010.md'
NEW=('readout_subspace_solver.py','test_readout_subspace_solver.py','check_readout_subspace.py','run_readout_subspace.py')

def state_results(x,y,c,rows):
    residual=x@c-y;result=[]
    for row in rows:
        q=dict(row);i=q['row_start'];n=q['row_count']
        q['physical_relative_target_mse']=float(residual[i:i+n].square().sum())/q['state_weight'];result.append(q)
    teacher=[q['physical_relative_target_mse'] for q in result if not q['preservation']]
    correct=[q['physical_relative_target_mse'] for q in result if q['preservation']]
    return dict(states=result,teacher_mean=sum(teacher)/6,teacher_max=max(teacher),preservation_mean=sum(correct)/35,preservation_max=max(correct),
        diagnostic_linear_subproblem_gate_passed=all(v<.01 for v in teacher) and max(correct)<.001,weighted_objective=float(residual.square().sum()))

def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter();calls=dict(outer=0,inner=0);nominal=0;fits=0
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,analytic_fits=fits,
        model_parameter_mutations=0,new_scoring_images=0,controller_started=False,**kw))
    try:
        corpus,op_path,rms,warm,_=verified_training_inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct));teachers=sorted(corpus.teachers)
        portable=json.loads((ROOT/'v81_complete_portable_verification.json').read_text())
        if not portable['passed'] or portable['optimizer_updates']!=840 or json.loads((TRAIN/'pipeline_status.json').read_text())['phase']!='complete':
            raise ValueError('Require complete locally verified v81')
        paths={a:TRAIN/a/'repair_step420.pt' for a in ARMS};hashes={a:file_hash(p) for a,p in paths.items()}
        for a in ARMS:
            old=json.loads((TRAIN/a/'config.json').read_text())['configuration']
            for n,h in old['source_sha256'].items():
                if file_hash(ROOT/n)!=h:raise ValueError('v81 frozen source changed '+n)
        cfg=dict(source_sha256={n:file_hash(ROOT/n) for n in tuple(dict.fromkeys(PRIOR_NAMES+NEW))},protocol_sha256=file_hash(ROOT/PROTOCOL),
            checkpoint_sha256=hashes,portable_v81_sha256=file_hash(ROOT/'v81_complete_portable_verification.json'),
            state_sha256={s:file_hash(ROOT/'states/repair_train_targets_v36'/(s+'.pt')) for s in ids},state_ids=ids,teachers=teachers,arms=ARMS,
            target_manifest_sha256=file_hash(CHECK/'target_manifest.json'),common_rms_sha256=file_hash(CHECK/'common_input_rms.pt'),
            corpus_config_sha256=file_hash(ROOT/'expanded_process_corpus_v70/config.json'),
            state_weights=dict(teacher=1/12,preservation=1/70),columns=129,outputs=64,
            solver='CPU FP64 thin SVD; rcond=max(rows,columns)*float64eps; no tuning or ridge',
            capacity_reproduction_rtol=1e-4,capacity_reproduction_atol=1e-6,new_deployment_parameters=0,FLOPs=None,
            inference='unchanged frozen v81 forward; teacher only matrix target; coefficients not loaded into model',
            scope='fixed 128 SiLU activation plus bias output-layer physical subproblem, not global model reachability or generation quality')
        freeze_config(OUT/'config.json',cfg);(OUT/'source').mkdir();(OUT/'inputs').mkdir()
        for n in cfg['source_sha256']:shutil.copy2(ROOT/n,OUT/'source'/n)
        for p in (ROOT/PROTOCOL,ROOT/'v81_complete_portable_verification.json',CHECK/'target_manifest.json',CHECK/'common_input_rms.pt'):
            shutil.copy2(p,OUT/'inputs'/p.name)
        shutil.copy2(ROOT/PROTOCOL,OUT/'frozen_protocol.md');status('loading')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False);torch.cuda.reset_peak_memory_stats();base={n:tensor_hash(p) for n,p in runtime.model.named_parameters()}
        ledger={q['state_id']:q for q in json.loads((CHECK/'target_manifest.json').read_text())};all_results={};loaded={}
        for arm in ARMS:
            status('capturing_fixed_activations',arm=arm);armout=OUT/arm;armout.mkdir()
            repair=build_reference(runtime,arm,op_path,rms,warm);payload=torch.load(paths[arm],map_location='cpu',weights_only=False)
            done=json.loads((TRAIN/arm/'training_complete.json').read_text())
            if payload['step']!=420 or payload['reference_arm']!=arm or done['checkpoint_sha256']!=hashes[arm]:raise ValueError('Wrong fixed checkpoint')
            load_adapter(repair.operator,payload['operator']);repair.head.load_state_dict(payload['head']);del payload
            before=list(tensor_fingerprints(repair));core=repair.operator.core_hashes();loaded[arm]=before
            if before!=done['final_tensor_hashes']:raise ValueError('Loaded actual tensor hash mismatch')
            recorded={q['state_id']:q for q in json.loads((TRAIN/arm/'capacity_report.json').read_text())['states']}
            shutil.copy2(TRAIN/arm/'training_complete.json',armout/'v81_training_complete.json')
            shutil.copy2(TRAIN/arm/'capacity_report.json',armout/'v81_capacity_report.json')
            blocks=[];targets=[];rows=[];offset=0
            original_c=torch.cat((repair.head.output.weight.detach().cpu().double().T,repair.head.output.bias.detach().cpu().double()[None]),0)
            with full_model_context(runtime),count_shared(repair.operator,calls),torch.no_grad():
                for sid in ids:
                    state=corpus.state(sid);target=target_on_device(corpus,state,runtime.device)
                    features,options=capture_features(runtime,state,repair.operator);nominal+=1;positions=options['modality_positions']
                    _,_,aligned=aligned_hidden(corpus,state,target,features,positions,repair.operator)
                    if aligned!=ledger[sid]:raise ValueError('Frozen target/source changed')
                    repair.operator.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                    saved=[]
                    handle=repair.head.output.register_forward_pre_hook(lambda module,args:saved.append(args[0].detach().cpu()))
                    try:delta,hidden=repair.forward_state(runtime,state,features,positions)
                    finally:handle.remove()
                    if len(saved)!=1 or saved[0].shape[0]!=1 or saved[0].shape[-1]!=128:raise ValueError('Unexpected actual readout activation')
                    if repair.last_shared_calls!=dict(outer=16,inner=16):raise ValueError('Actual shared budget mismatch')
                    physical=relative_physical(delta,target,state['latent'].to(runtime.device).float())
                    if not math.isclose(physical,recorded[sid]['physical_relative_target_mse'],rel_tol=1e-4,abs_tol=1e-6):raise ValueError('Original v81 capacity changed')
                    if not torch.equal(target['delta'][:1],target['delta'][1:]):raise ValueError('CFG target not shared')
                    activation=saved[0][0];shape=(1,activation.shape[0],64);numel=math.prod(shape)
                    if numel>=2**24:raise ValueError('FP32 index permutation unsafe')
                    index=torch.arange(numel,device=runtime.device,dtype=torch.float32).reshape(shape)
                    permutation=patch_delta(runtime,index,state['latent'])[:1].flatten().long()
                    patches=patch_target_from_permutation(target['delta'][:1],permutation,shape)
                    if not torch.equal(patch_delta(runtime,patches,state['latent']),target['delta']):raise ValueError('Real physical to patch inverse failed')
                    scale=float((state['latent'].to(runtime.device).float().square().mean() if target['preservation'] else target['delta'].square().mean()).clamp_min(1e-5))
                    weight=1/70 if target['preservation'] else 1/12;x,y=weighted_block(activation,patches[0],scale,weight)
                    blocks.append(x);targets.append(y)
                    rows.append(dict(state_id=sid,split='train',preservation=target['preservation'],row_start=offset,row_count=x.shape[0],state_weight=weight,scale=scale,
                        activation_sha256=tensor_hash(activation),target_delta_sha256=tensor_hash(target['delta']),
                        target_patch_sha256=tensor_hash(patches),unpatchify_permutation_sha256=tensor_hash(permutation),inverse_strict_roundtrip=True,
                        recorded_original_physical_mse=physical,original_state_fingerprint=state['state_fingerprint']))
                    offset+=x.shape[0]
                    write_json(armout/'capture_progress.json',dict(states=len(rows),shared_calls=calls,nominal_velocity_calls=nominal))
                    del delta,hidden,features,patches,index,permutation,saved,activation,target
            x=torch.cat(blocks);y=torch.cat(targets);del blocks,targets
            original=state_results(x,y,original_c,rows)
            for q in original['states']:
                if not math.isclose(q['physical_relative_target_mse'],q['recorded_original_physical_mse'],rel_tol=1e-4,abs_tol=1e-6):raise ValueError('FP64 linear surrogate differs from actual FP32 head')
            status('solving_fixed_subspace',arm=arm);c,stats=solve_fixed_subspace(x,y);fits+=1
            if stats['normal_equation_relative_l2']>1e-7:raise ValueError('Numerical linear solve stationarity failed')
            result=dict(original=original,analytic_reference=state_results(x,y,c,rows),solver=stats,analytic_fit_only=True,coefficients_loaded_into_model=False,
                no_generated_endpoint_or_quality=True,checkpoint_source_sha256=hashes[arm])
            for n,v in (('design',x),('targets',y),('coefficients',c),('original_coefficients',original_c)):
                np.save(armout/(n+'.npy'),v.numpy(),allow_pickle=False)
            write_json(armout/'fit_results.json',result);all_results[arm]=result
            if list(tensor_fingerprints(repair))!=before or repair.operator.core_hashes()!=core or any(p.grad is not None for p in repair.parameters()):raise ValueError('Frozen repair changed or acquired gradients')
            del repair,x,y,c,original_c;torch.cuda.empty_cache()
        if base!={n:tensor_hash(p) for n,p in runtime.model.named_parameters()} or any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Base changed or acquired gradients')
        for a,p in paths.items():
            if file_hash(p)!=hashes[a]:raise ValueError('Fixed checkpoint bytes changed')
        for n,h in cfg['source_sha256'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('Frozen source changed')
        if file_hash(ROOT/PROTOCOL)!=cfg['protocol_sha256']:raise ValueError('Protocol changed')
        verified_training_inputs();write_json(OUT/'loaded_tensor_hashes.json',loaded);write_json(OUT/'base_tensor_hashes.json',base)
        status('complete',state_captures=82,nominal_velocity_calls=nominal,actual_shared_calls=calls,
            backward_recomputation_calls=0,base_hash_unchanged=True,base_gradients=False,all_adapter_head_and_checkpoint_hashes_unchanged=True,
            original_core_hash_unchanged=True,coefficients_loaded_into_model=False,peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
            walltime_seconds=time.perf_counter()-started,FLOPs=None)
    except Exception as e:status('failed',error=str(e));raise
if __name__=='__main__':main()
