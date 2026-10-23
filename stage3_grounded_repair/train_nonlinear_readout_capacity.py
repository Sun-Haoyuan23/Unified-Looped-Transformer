"""Fixed 38-TRAIN-state nonlinear capacity study, not a generation-quality claim."""
import json,os,shutil,subprocess,time
from pathlib import Path
import torch
from analytic_readout_operator import load_adapter
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from launch_grounding_diagnostic import GPUS
from nonlinear_token_readout import NonlinearTokenReadout,joint_token_features,patch_delta
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003');ROOT=WORK/'stage3_grounded_repair'
OUT=ROOT/'nonlinear_readout_capacity_v60'
MODES=('trained_loop2','trained_no_loop');STEPS=1000;SEED=20261008


def main():
    OUT.mkdir(exist_ok=False);updates={mode:0 for mode in MODES}
    def status(phase,**kw):
        write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),
            head_optimizer_updates=updates.copy(),shared_core_optimizer_updates=0,
            new_scoring_images=0,controller_started=False,**kw))
    try:
        q=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'],text=True).strip().split(',')
        if q[0].strip()!=GPUS[0] or int(q[1])>=2000 or os.environ['CUDA_VISIBLE_DEVICES']!=GPUS[0]:
            raise ValueError('GPU0 occupied/UUID mismatch; no other task interrupted')
        prior=ROOT/'preservation_rank_audit_v59'
        if json.loads((prior/'pipeline_status.json').read_text())['phase']!='complete':
            raise ValueError('Require completed frozen v59')
        frozen=json.loads((prior/'config.json').read_text())['configuration']
        refs={r['state_id']:r for r in json.loads((prior/'design_fingerprints.json').read_text())}
        corpus=ProcessCorpus(ROOT);ids=sorted(set(corpus.teachers)|set(corpus.correct))
        if ids!=frozen['state_ids'] or len(ids)!=38 or len(corpus.teachers)!=3 or len(corpus.correct)!=35:
            raise ValueError('Frozen strict TRAIN corpus changed')
        checkpoint=ROOT/'process_recurrent_train_v52/joint_process/operator_step100.pt'
        if file_hash(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Fixed v52 checkpoint changed')
        for sid in ids:
            if file_hash(ROOT/'states/repair_train_targets_v36'/(sid+'.pt'))!=frozen['state_sha256'][sid]:
                raise ValueError('Frozen state bytes changed')
        names=('train_nonlinear_readout_capacity.py','nonlinear_token_readout.py','test_nonlinear_token_readout.py',
            'run_nonlinear_readout_capacity.py','process_training_targets.py','analytic_readout_operator.py',
            'recurrent_training_protocol.py','trainable_recurrent_state.py','state_space_revisit.py',
            'runtime.py','candidate_provenance.py','evidence_encoder.py','training_pairs.py')
        config=dict(source_sha256={n:file_hash(ROOT/n) for n in names},checkpoint_sha256=file_hash(checkpoint),
            v59_config_sha256=file_hash(prior/'config.json'),v59_fingerprints_sha256=file_hash(prior/'design_fingerprints.json'),
            state_ids=ids,state_sha256=frozen['state_sha256'],process_corpus_hashes=corpus.input_hashes,
            modes=MODES,teachers=3,preservation_states=35,width=128,steps_per_mode=STEPS,seed=SEED,
            optimizer='Adam',learning_rate=1e-3,weight_decay=0,full_batch=True,
            loss='0.5 mean_teacher(normalized physical MSE) + 0.5 mean35_preservation(normalized physical MSE)',
            input_normalization='Per-column RMS across 38 TRAIN states, no labels, clamp1e-4',
            initialization='Same trainable tensors in both modes; final output zero',
            fixed_checkpoint='head_step1000.pt; no endpoint/validation selection',
            core_weights_and_prior_adapter_frozen=True,head_optimizer_only=True,
            physical_capacity_gate=dict(each_teacher_mse_below=.01,max_preservation_mse_below=.001),
            gate_is_not_a_quality_test=True,no_automatic_generation_or_val_or_controller=True,FLOPs=None)
        freeze_config(OUT/'config.json',config);(OUT/'source').mkdir()
        for n in names:shutil.copy2(ROOT/n,OUT/'source'/n)
        status('loading',expected_states=38)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False);torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        op=build_operator(runtime,'joint_ilsd');payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        load_adapter(op,payload['operator']);op.requires_grad_(False)
        base_before={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}
        adapter_before={k:tensor_hash(v) for k,v in op.adapter_state().items()}
        designs={mode:[] for mode in MODES};labels=[];keys=[];checks=[]
        with full_model_context(runtime),torch.no_grad():
            for index,sid in enumerate(ids):
                status('capturing_fixed_features',completed_states=index,expected_states=38,state_id=sid)
                state=corpus.state(sid);preserve=sid in corpus.correct
                op.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                features,options=capture_features(runtime,state,op)
                target=corpus.target(runtime,state,features,op,preserve)
                original=state['latent'].to(runtime.device).float();gold=(target['input']-original)[:1]
                y=gold.flatten(2).transpose(1,2).contiguous().reshape(1,-1,runtime.channels*runtime.patch**2)
                if not torch.equal(patch_delta(runtime,y,original)[:1],gold):raise ValueError('Patch roundtrip failed')
                y=y.squeeze(0).cpu()
                if tensor_hash(y)!=refs[sid]['target_patch_sha256']:raise ValueError('v59 physical target mismatch')
                for mode in MODES:
                    tokens,calls=joint_token_features(op,features,options['modality_positions'],mode.split('_',1)[1])
                    tokens=tokens.squeeze(0).cpu()
                    with_bias=torch.cat((tokens,torch.ones(tokens.shape[0],1)),1)
                    if tensor_hash(with_bias)!=refs[sid]['feature_sha256'][mode]:raise ValueError('v59 feature mismatch')
                    expected={'outer':4,'inner':4} if mode=='trained_loop2' else {'outer':0,'inner':0}
                    if calls!=expected:raise ValueError('Actual shared-core route mismatch')
                    designs[mode].append(tokens);checks.append(dict(state_id=sid,mode=mode,
                        feature_sha256=tensor_hash(with_bias),target_patch_sha256=tensor_hash(y),actual_shared_calls=calls))
                labels.append(y)
                denom=float((original.square().mean() if preserve else gold.square().mean()).clamp_min(1e-5))
                keys.append(dict(state_id=sid,preservation=preserve,denominator=denom,rows=y.shape[0]))
        write_json(OUT/'feature_checks.json',checks)
        # Preserve small target tensors and exact input fingerprints. The large
        # feature matrices are reproducible from unchanged v52/state sources.
        torch.save(dict(labels=labels,keys=keys,diagnostic_train_targets=True),OUT/'training_targets.pt')
        y=torch.cat(labels).to(runtime.device);weights=[]
        for key in keys:
            weights.append(torch.full((key['rows'],1),.5/((35 if key['preservation'] else 3)*key['denominator']*key['rows']*y.shape[1])))
        w=torch.cat(weights).to(runtime.device);results={};initial_ref=None
        for mode in MODES:
            folder=OUT/mode;folder.mkdir()
            x=torch.cat(designs[mode]).to(runtime.device)
            rms=x.square().mean(0).sqrt().clamp_min(1e-4)
            torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
            head=NonlinearTokenReadout(rms,y.shape[1],128).to(runtime.device)
            initial={k:tensor_hash(v) for k,v in head.named_parameters()}
            if initial_ref is None:initial_ref=initial
            elif initial!=initial_ref:raise ValueError('Matched nonlinear head initialization failed')
            write_json(folder/'initialization.json',dict(trainable_sha256=initial,input_rms_sha256=tensor_hash(rms),
                parameters=sum(p.numel() for p in head.parameters())))
            with torch.no_grad():
                if not torch.equal(head(x),torch.zeros_like(y)):raise ValueError('Untrained readout not strict zero')
            optimizer=torch.optim.Adam(head.parameters(),lr=1e-3,weight_decay=0)
            with (folder/'train.jsonl').open('w') as log:
                for step in range(1,STEPS+1):
                    optimizer.zero_grad(set_to_none=True)
                    prediction=head(x);loss=((prediction-y).square()*w).sum()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward()
                    gradient=torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in head.parameters()]))
                    if not torch.isfinite(gradient) or float(gradient)==0:raise ValueError('Invalid nonlinear head gradient')
                    optimizer.step();updates[mode]=step
                    row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(gradient),time=time.time(),
                        head_optimizer_update=True,shared_core_optimizer_update=False)
                    log.write(json.dumps(row)+'\n');log.flush()
                    if step==1 or step%25==0:status('training_head',mode=mode,step=step,steps=STEPS,loss=row['loss'])
            with torch.no_grad():
                prediction=head(x);selected=[];offset=0
                for key in keys:
                    n=key['rows'];error=float((prediction[offset:offset+n]-y[offset:offset+n]).square().mean()/key['denominator'])
                    selected.append(dict(key,physical_relative_target_mse=error));offset+=n
                teacher_values=[r['physical_relative_target_mse'] for r in selected if not r['preservation']]
                correct_values=[r['physical_relative_target_mse'] for r in selected if r['preservation']]
                passed=all(v<.01 for v in teacher_values) and max(correct_values)<.001
            path=folder/'head_step1000.pt'
            torch.save(dict(head=head.state_dict(),step=STEPS,head_optimizer_updates=STEPS,
                prior_shared_core_checkpoint_sha256=file_hash(checkpoint),prior_shared_core_step=100,
                shared_core_optimizer_updates_this_experiment=0,mode=mode,
                diagnostic_capacity_only=True,never_automatically_deployed=True),path)
            results[mode]=dict(states=selected,head_checkpoint_sha256=file_hash(path),head_optimizer_updates=STEPS,
                teacher_mean=sum(teacher_values)/3,teacher_max=max(teacher_values),
                preservation_mean=sum(correct_values)/35,preservation_max=max(correct_values),physical_capacity_gate_passed=passed)
            write_json(OUT/'results.json',results)
            write_json(folder/'training_complete.json',dict(step=STEPS,head_optimizer_updates=STEPS,shared_core_optimizer_updates=0,
                head_checkpoint_sha256=file_hash(path),finite_training=True,physical_capacity_gate_passed=passed))
            del x,head,optimizer,prediction;torch.cuda.empty_cache()
        if base_before!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}:raise ValueError('Base tensor changed')
        if adapter_before!={k:tensor_hash(v) for k,v in op.adapter_state().items()}:raise ValueError('Prior adapter changed')
        if any(p.grad is not None for p in runtime.model.parameters()) or any(p.grad is not None for p in op.parameters()):
            raise ValueError('Frozen base or core gradient')
        if file_hash(checkpoint)!=config['checkpoint_sha256']:raise ValueError('Prior checkpoint bytes changed')
        status('complete',states=38,modes=2,total_head_optimizer_updates=sum(updates.values()),
            full_base_and_prior_adapter_hashes_unchanged=True,base_and_prior_adapter_no_grad=True,
            prior_checkpoint_unchanged=True,feature_checks_exact_to_v59=len(checks),
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
            no_new_semantic_or_quality_measurement=True,not_a_shared_core_training_or_architecture_gain=True)
    except Exception as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
