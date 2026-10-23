"""Matched bounded full-batch L-BFGS at fixed v60 nonlinear head structure."""
import inspect,json,math,os,shutil,subprocess,time
from pathlib import Path
import torch
from analytic_readout_operator import load_adapter
from bounded_lbfgs import BoundedLBFGS
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from launch_grounding_diagnostic import GPUS
from nonlinear_token_readout import NonlinearTokenReadout,joint_token_features
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003');ROOT=WORK/'stage3_grounded_repair'
OUT=ROOT/'optimizer_separation_v62';MODES=('trained_loop2','trained_no_loop')
OPTIONS=dict(lr=1,max_iter=100,max_eval=125,history_size=20,tolerance_grad=1e-7,
             tolerance_change=1e-9,line_search_fn='strong_wolfe')


def per_state(prediction,target,keys):
    selected=[];offset=0
    for key in keys:
        n=key['rows'];mse=float((prediction[offset:offset+n]-target[offset:offset+n]).square().mean()/key['denominator'])
        selected.append(dict(key,physical_relative_target_mse=mse));offset+=n
    if offset!=prediction.shape[0]:raise ValueError('Invalid per-state token boundaries')
    return selected


def main():
    OUT.mkdir(exist_ok=False);counters={}
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),
        optimizer_counters=counters.copy(),shared_core_optimizer_updates=0,new_scoring_images=0,
        controller_started=False,**kw))
    try:
        q=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if q[0].strip()!=GPUS[0] or int(q[1])>=2000 or os.environ['CUDA_VISIBLE_DEVICES']!=GPUS[0]:raise ValueError('GPU0 occupied/UUID mismatch')
        prior=ROOT/'nonlinear_readout_capacity_v60';v59=ROOT/'preservation_rank_audit_v59'
        if json.loads((prior/'pipeline_status.json').read_text())['phase']!='complete':raise ValueError('Fixed v60 not complete')
        previous=json.loads((prior/'config.json').read_text())['configuration']
        prior_results=json.loads((prior/'results.json').read_text())
        refs={r['state_id']:r for r in json.loads((v59/'design_fingerprints.json').read_text())}
        corpus=ProcessCorpus(ROOT);ids=sorted(set(corpus.teachers)|set(corpus.correct))
        if ids!=previous['state_ids'] or len(ids)!=38 or corpus.input_hashes!=previous['process_corpus_hashes']:
            raise ValueError('Strict frozen TRAIN corpus changed')
        checkpoint=ROOT/'process_recurrent_train_v52/joint_process/operator_step100.pt'
        if file_hash(checkpoint)!=previous['checkpoint_sha256']:raise ValueError('Prior operator changed')
        target_path=prior/'training_targets.pt';manifest=json.loads((prior/'archive_file_manifest.json').read_text())
        if file_hash(target_path)!=manifest[str(target_path.relative_to(ROOT))]:raise ValueError('Frozen TRAIN targets changed')
        target_cache=torch.load(target_path,map_location='cpu',weights_only=False)
        keys=target_cache['keys'];labels=target_cache['labels']
        if [k['state_id'] for k in keys]!=ids or len(labels)!=38:raise ValueError('Cached TRAIN ordering changed')
        for key,y in zip(keys,labels):
            if tensor_hash(y)!=refs[key['state_id']]['target_patch_sha256']:raise ValueError('Frozen target patch changed')
            if key['preservation']!=(key['state_id'] in corpus.correct):raise ValueError('Teacher/preservation identity changed')
        head_paths={mode:prior/mode/'head_step1000.pt' for mode in MODES}
        for mode,path in head_paths.items():
            if file_hash(path)!=prior_results[mode]['head_checkpoint_sha256']:raise ValueError('Fixed prior head changed')
        names=('train_optimizer_separation.py','bounded_lbfgs.py','test_bounded_lbfgs.py','run_optimizer_separation.py',
            'nonlinear_token_readout.py','process_training_targets.py','analytic_readout_operator.py',
            'recurrent_training_protocol.py','trainable_recurrent_state.py','state_space_revisit.py',
            'runtime.py','candidate_provenance.py','evidence_encoder.py','training_pairs.py')
        for name in set(names)&set(previous['source_sha256']):
            if file_hash(ROOT/name)!=previous['source_sha256'][name]:raise ValueError('Frozen shared source changed')
        (OUT/'source').mkdir()
        torch_source=OUT/'source/torch_lbfgs_runtime.py'
        torch_source.write_text(inspect.getsource(torch.optim.LBFGS),encoding='utf-8')
        config=dict(source_sha256={n:file_hash(ROOT/n) for n in names},torch_version=torch.__version__,
            torch_lbfgs_source_sha256=file_hash(torch_source),checkpoint_sha256=file_hash(checkpoint),
            input_head_checkpoint_sha256={m:file_hash(p) for m,p in head_paths.items()},
            v60_config_sha256=file_hash(prior/'config.json'),v60_results_sha256=file_hash(prior/'results.json'),
            training_targets_sha256=file_hash(target_path),state_ids=ids,state_sha256=previous['state_sha256'],
            v59_fingerprints_sha256=file_hash(v59/'design_fingerprints.json'),process_corpus_hashes=corpus.input_hashes,
            options=OPTIONS,hard_closure_limit=125,optimizer_step_calls_per_mode=1,
            budget_exhaustion='restore last accepted parameters, never keep unevaluated trial; optimizer state not resumed',
            diagnostic_objective_evaluations_per_mode=2,diagnostic_backward_evaluations_per_mode=1,
            loss=previous['loss'],width=128,input_normalization='exact frozen v60 buffers',head_structure_unchanged=True,
            starts='each mode own fixed v60 Adam step1000; no equality claim for trained starting weights',
            physical_capacity_gate=previous['physical_capacity_gate'],reference_tolerance=dict(rtol=1e-6,atol=1e-9),
            no_automatic_generation_or_val_or_controller=True,FLOPs=None)
        freeze_config(OUT/'config.json',config)
        for name in names:shutil.copy2(ROOT/name,OUT/'source'/name)
        status('loading',states=38)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False);torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        op=build_operator(runtime,'joint_ilsd');load_adapter(op,torch.load(checkpoint,map_location='cpu',weights_only=False)['operator'])
        op.requires_grad_(False)
        base_hash={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}
        adapter_hash={k:tensor_hash(v) for k,v in op.adapter_state().items()}
        designs={mode:[] for mode in MODES};checks=[]
        with torch.no_grad(),full_model_context(runtime):
            for index,key in enumerate(keys):
                sid=key['state_id'];status('capturing_fixed_features',state_id=sid,completed_states=index)
                path=ROOT/'states/repair_train_targets_v36'/(sid+'.pt')
                if file_hash(path)!=previous['state_sha256'][sid]:raise ValueError('Frozen TRAIN state bytes changed')
                state=corpus.state(sid);original=state['latent'].to(runtime.device).float();y=labels[index]
                denominator=float((original.square().mean() if key['preservation'] else y.to(runtime.device).square().mean()).clamp_min(1e-5))
                if not math.isclose(denominator,key['denominator'],rel_tol=1e-6,abs_tol=1e-9):raise ValueError('Physical normalization changed')
                op.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                features,options=capture_features(runtime,state,op)
                for mode in MODES:
                    x,calls=joint_token_features(op,features,options['modality_positions'],mode.split('_',1)[1])
                    x=x.squeeze(0).cpu();xb=torch.cat((x,torch.ones(x.shape[0],1)),1)
                    if tensor_hash(xb)!=refs[sid]['feature_sha256'][mode]:raise ValueError('Frozen feature mismatch')
                    expected={'outer':4,'inner':4} if mode=='trained_loop2' else {'outer':0,'inner':0}
                    if calls!=expected:raise ValueError('Actual recurrence route mismatch')
                    designs[mode].append(x);checks.append(dict(state_id=sid,mode=mode,feature_sha256=tensor_hash(xb),
                        target_patch_sha256=tensor_hash(y),actual_shared_calls=calls))
        write_json(OUT/'feature_checks.json',checks)
        y=torch.cat(labels).to(runtime.device);weights=[]
        for key in keys:weights.append(torch.full((key['rows'],1),.5/((35 if key['preservation'] else 3)*key['denominator']*key['rows']*y.shape[1])))
        w=torch.cat(weights).to(runtime.device);results={};reference_checks=[]
        for mode in MODES:
            folder=OUT/mode;folder.mkdir();x=torch.cat(designs[mode]).to(runtime.device)
            payload=torch.load(head_paths[mode],map_location=runtime.device,weights_only=False)
            if payload['step']!=1000 or payload['mode']!=mode:raise ValueError('Prior fixed head metadata changed')
            head=NonlinearTokenReadout(payload['head']['input_rms'],64,128).to(runtime.device)
            head.load_state_dict(payload['head']);normalizer=tensor_hash(head.input_rms)
            write_json(folder/'initialization.json',dict(source_checkpoint_sha256=file_hash(head_paths[mode]),
                tensor_sha256={k:tensor_hash(v) for k,v in head.state_dict().items()},parameters=sum(p.numel() for p in head.parameters())))
            with torch.no_grad():
                prediction=head(x);initial_loss=float(((prediction-y).square()*w).sum());before=per_state(prediction,y,keys)
            prior_rows={r['state_id']:r for r in prior_results[mode]['states']}
            for row in before:
                expected=prior_rows[row['state_id']]['physical_relative_target_mse'];observed=row['physical_relative_target_mse']
                if not math.isclose(observed,expected,rel_tol=1e-6,abs_tol=1e-9):raise ValueError('Fixed v60 start reproduction failed')
                reference_checks.append(dict(mode=mode,state_id=row['state_id'],expected=expected,observed=observed,absolute_error=abs(observed-expected)))
            closure_log=(folder/'evaluations.jsonl').open('w');update_log=(folder/'accepted_updates.jsonl').open('w')
            def on_accept(row):
                update_log.write(json.dumps(dict(row,time=time.time()))+'\n');update_log.flush()
            optimizer=BoundedLBFGS(head.parameters(),on_accept=on_accept,**OPTIONS)
            def closure():
                optimizer.zero_grad(set_to_none=True)
                prediction=head(x);loss=((prediction-y).square()*w).sum();loss.backward()
                gradient=torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in head.parameters()]))
                if not torch.isfinite(loss) or not torch.isfinite(gradient):raise ValueError('Nonfinite L-BFGS objective/gradient')
                row=dict(evaluation=optimizer.audit['closure_evaluations'],loss=float(loss.detach()),gradient_norm=float(gradient),
                    attempted_internal_iteration=int(optimizer.state[optimizer._params[0]].get('n_iter',0)),
                    trial=optimizer._in_trial,time=time.time())
                closure_log.write(json.dumps(row)+'\n');closure_log.flush()
                if row['evaluation']==1 or row['evaluation']%10==0:
                    counters[mode]=optimizer.audit.copy();status('optimizing_head',mode=mode,loss=row['loss'])
                return loss
            start=time.monotonic()
            optimizer.step(closure)
            optimizer_seconds=time.monotonic()-start
            closure_log.close();update_log.close();counters[mode]=optimizer.audit.copy()
            optimizer.zero_grad(set_to_none=True)
            prediction=head(x);final_loss=((prediction-y).square()*w).sum();final_loss.backward()
            final_grad=float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in head.parameters()])))
            if not torch.isfinite(final_loss) or not math.isfinite(final_grad):raise ValueError('Nonfinite final accepted state')
            if float(final_loss)>initial_loss+1e-7+1e-5*abs(initial_loss):raise ValueError('Accepted L-BFGS objective increased')
            selected=per_state(prediction.detach(),y,keys)
            teacher=[r['physical_relative_target_mse'] for r in selected if not r['preservation']]
            correct=[r['physical_relative_target_mse'] for r in selected if r['preservation']]
            passed=all(v<.01 for v in teacher) and max(correct)<.001
            if tensor_hash(head.input_rms)!=normalizer:raise ValueError('Frozen input normalization changed')
            path=folder/'head_lbfgs_final.pt'
            torch.save(dict(head=head.state_dict(),source_adam_step=1000,source_checkpoint_sha256=file_hash(head_paths[mode]),
                optimizer_audit=optimizer.audit,mode=mode,final_accepted_iterate=True,optimizer_state_not_resumable=True,
                shared_core_optimizer_updates_this_experiment=0,diagnostic_capacity_only=True),path)
            results[mode]=dict(states=selected,head_checkpoint_sha256=file_hash(path),optimizer_audit=optimizer.audit,
                initial_loss=initial_loss,final_loss=float(final_loss.detach()),final_gradient_norm=final_grad,
                optimizer_walltime_seconds=optimizer_seconds,diagnostic_objective_forward_evaluations=2,diagnostic_backward_evaluations=1,
                teacher_mean=sum(teacher)/3,teacher_max=max(teacher),preservation_mean=sum(correct)/35,preservation_max=max(correct),
                physical_capacity_gate_passed=passed,normalization_unchanged=True)
            write_json(OUT/'results.json',results)
            write_json(folder/'training_complete.json',dict(optimizer_audit=optimizer.audit,head_checkpoint_sha256=file_hash(path),
                physical_capacity_gate_passed=passed,finite_training=True))
            del x,head,optimizer,prediction,final_loss;torch.cuda.empty_cache()
        write_json(OUT/'reference_checks.json',reference_checks)
        if base_hash!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()} or adapter_hash!={k:tensor_hash(v) for k,v in op.adapter_state().items()}:
            raise ValueError('Frozen base or prior adapter changed')
        if any(p.grad is not None for p in runtime.model.parameters()) or any(p.grad is not None for p in op.parameters()):raise ValueError('Frozen base/core gradient')
        if file_hash(checkpoint)!=config['checkpoint_sha256'] or any(file_hash(p)!=config['input_head_checkpoint_sha256'][m] for m,p in head_paths.items()):
            raise ValueError('Frozen input checkpoint changed')
        status('complete',states=38,modes=2,feature_checks_exact_to_v59=len(checks),starting_mse_reproductions=len(reference_checks),
            optimizer_step_calls=sum(r['optimizer_audit']['optimizer_step_calls'] for r in results.values()),
            accepted_parameter_updates=sum(r['optimizer_audit']['accepted_parameter_updates'] for r in results.values()),
            closure_evaluations=sum(r['optimizer_audit']['closure_evaluations'] for r in results.values()),
            full_base_and_prior_adapter_hashes_unchanged=True,base_and_prior_adapter_no_grad=True,input_checkpoints_unchanged=True,
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,no_new_semantic_or_quality_measurement=True)
    except Exception as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
