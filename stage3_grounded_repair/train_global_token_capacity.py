"""Fixed physical capacity experiment; frozen recurrent core, no endpoint scores."""
import argparse,json,shutil,time
import torch
from analytic_readout_operator import load_adapter
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from global_token_readout import GlobalTokenReadout
from nonlinear_recurrent_protocol import ROOT,WORK,frozen_inputs
from nonlinear_token_readout import joint_token_features,patch_delta
from recurrent_training_protocol import build_operator
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context

NAMES=('train_global_token_capacity.py','run_global_token_capacity.py','global_token_readout.py','test_global_token_readout.py',
 'nonlinear_recurrent_protocol.py','nonlinear_token_readout.py','process_training_targets.py','analytic_readout_operator.py',
 'recurrent_training_protocol.py','trainable_recurrent_state.py','state_space_revisit.py','runtime.py',
 'candidate_provenance.py','evidence_encoder.py','training_pairs.py')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('loop2','no_loop'),required=True);mode=parser.parse_args().mode
    out=ROOT/'global_token_capacity_v66'/mode;out.mkdir(exist_ok=False);updates=0
    def status(phase,**kw):write_json(out/'worker_status.json',dict(phase=phase,time=time.time(),head_optimizer_updates=updates,
        shared_core_optimizer_updates=0,new_scoring_images=0,controller_started=False,**kw))
    try:
        audit=ROOT/'nonlinear_recurrent_mapping_audit_v65/pipeline_status.json'
        if json.loads(audit.read_text())['phase']!='complete':raise ValueError('Require completed v65')
        corpus,_,_,labels,inputs=frozen_inputs();ids=sorted(set(corpus.teachers)|set(corpus.correct))
        source=ROOT/'nonlinear_recurrent_train_v64/trainable_loop/repair_step210.pt'
        done=json.loads((source.parent/'training_complete.json').read_text())
        if file_hash(source)!=done['checkpoint_sha256']:raise ValueError('Fixed v64 source changed')
        config=dict(mode=mode,source_sha256={n:file_hash(ROOT/n) for n in NAMES},inputs=inputs,state_ids=ids,
          v65_status_sha256=file_hash(audit),operator_checkpoint_sha256=file_hash(source),operator_source='v64 trainable_loop fixed step210',
          steps=1000,optimizer='Adam',lr=1e-3,seed=20261008,width=64,query_key_rank=32,head_parameters=483968,
          loss='0.5 mean3_teacher(normalized physical MSE) + 0.5 mean35_preservation(normalized physical MSE)',
          full_batch=True,core_frozen=True,automatic_endpoint_or_val=False,FLOPs=None)
        freeze_config(out/'config.json',config);(out/'source').mkdir()
        for n in NAMES:shutil.copy2(ROOT/n,out/'source'/n)
        shutil.copy2(ROOT/'STAGE3_GLOBAL_TOKEN_PROTOCOL_20261008.md',out/'frozen_protocol.md')
        status('loading');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False);op=build_operator(runtime,'joint_ilsd')
        payload=torch.load(source,map_location='cpu',weights_only=False);load_adapter(op,payload['operator']);op.requires_grad_(False)
        before={k:tensor_hash(v) for k,v in runtime.model.named_parameters()};adapter={k:tensor_hash(v) for k,v in op.adapter_state().items()}
        tokens=[];values=[];targets=[];keys=[];checks=[]
        with full_model_context(runtime),torch.no_grad():
            for i,sid in enumerate(ids):
                status('capturing_fixed_features',completed_states=i,state_id=sid)
                state=corpus.state(sid);preserve=sid in corpus.correct
                op.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                features,options=capture_features(runtime,state,op)
                x,calls=joint_token_features(op,features,options['modality_positions'],mode)
                original=state['latent'].to(runtime.device).float();gold=patch_delta(runtime,labels[sid][None].to(runtime.device),original)
                v=original[:1].flatten(2).transpose(1,2).contiguous().reshape(1,-1,runtime.channels*runtime.patch**2)
                if not torch.equal(patch_delta(runtime,v,original)[:1],original[:1]):raise ValueError('Released original latent patch roundtrip failed')
                if x.shape[:2]!=v.shape[:2]:raise ValueError('Spatial token/physical patch alignment mismatch')
                expected={'outer':4,'inner':4} if mode=='loop2' else {'outer':0,'inner':0}
                if calls!=expected:raise ValueError('Revisit route mismatch')
                y=labels[sid][None].to(runtime.device)
                denominator=float((original.square().mean() if preserve else gold.square().mean()).clamp_min(1e-5))
                keys.append(dict(state_id=sid,preservation=preserve,denominator=denominator))
                checks.append(dict(state_id=sid,feature_sha256=tensor_hash(x),original_patch_sha256=tensor_hash(v),
                                   target_patch_sha256=tensor_hash(labels[sid]),actual_shared_calls=calls))
                tokens.append(x.squeeze(0).cpu());values.append(v.squeeze(0).cpu());targets.append(y.squeeze(0).cpu())
        write_json(out/'feature_checks.json',checks)
        x=torch.stack(tokens).to(runtime.device);v=torch.stack(values).to(runtime.device);y=torch.stack(targets).to(runtime.device)
        rms=x.square().mean((0,1)).sqrt().clamp_min(1e-4)
        torch.manual_seed(20261008);torch.cuda.manual_seed_all(20261008)
        head=GlobalTokenReadout(rms).to(runtime.device)
        init={k:tensor_hash(p) for k,p in head.named_parameters()}
        if sum(p.numel() for p in head.parameters())!=483968:raise ValueError('Head parameter budget changed')
        write_json(out/'initialization.json',dict(trainable_tensor_sha256=init,input_rms_sha256=tensor_hash(rms)))
        with torch.no_grad():
            if torch.count_nonzero(head(x,v)):raise ValueError('New head not strict zero')
        weights=torch.tensor([.5/((35 if q['preservation'] else 3)*q['denominator']) for q in keys],device=runtime.device)
        optimizer=torch.optim.Adam(head.parameters(),lr=1e-3,weight_decay=0)
        with (out/'train.jsonl').open('w') as log:
            for step in range(1,1001):
                optimizer.zero_grad(set_to_none=True);prediction=head(x,v)
                loss=((prediction-y).square().mean((1,2))*weights).sum()
                loss.backward();grad=torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in head.parameters()]))
                if not torch.isfinite(loss) or not torch.isfinite(grad) or not grad>0:raise ValueError('Nonfinite or zero loss gradient')
                optimizer.step();updates=step
                row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),time=time.time(),head_optimizer_update=True,shared_core_optimizer_update=False)
                log.write(json.dumps(row)+'\n');log.flush()
                if step==1 or step%25==0:status('training_head',loss=row['loss'])
        with torch.no_grad():
            mse=(head(x,v)-y).square().mean((1,2));records=[dict(q,physical_relative_target_mse=float(e/q['denominator'])) for q,e in zip(keys,mse)]
        teachers=[q['physical_relative_target_mse'] for q in records if not q['preservation']];correct=[q['physical_relative_target_mse'] for q in records if q['preservation']]
        passed=max(teachers)<.01 and max(correct)<.001
        path=out/'head_step1000.pt';torch.save(dict(head=head.state_dict(),mode=mode,step=1000,head_optimizer_updates=1000,
          operator_checkpoint_sha256=file_hash(source),shared_core_optimizer_updates=0,diagnostic_capacity_only=True),path)
        report=dict(states=records,teacher_mean=sum(teachers)/3,teacher_max=max(teachers),preservation_mean=sum(correct)/35,
          preservation_max=max(correct),physical_capacity_gate_passed=passed,head_checkpoint_sha256=file_hash(path))
        write_json(out/'capacity_report.json',report)
        if before!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()} or adapter!={k:tensor_hash(v) for k,v in op.adapter_state().items()}:raise ValueError('Frozen base or adapter changed')
        if any(p.grad is not None for p in runtime.model.parameters()) or any(p.grad is not None for p in op.parameters()):raise ValueError('Frozen base or adapter has gradient')
        if frozen_inputs()[-1]!=inputs or file_hash(source)!=config['operator_checkpoint_sha256']:raise ValueError('Frozen source changed')
        write_json(out/'training_complete.json',dict(head_optimizer_updates=1000,shared_core_optimizer_updates=0,physical_capacity_gate_passed=passed,
          base_and_prior_adapter_hashes_unchanged=True,base_and_prior_adapter_no_grad=True,head_checkpoint_sha256=file_hash(path)))
        status('complete',physical_capacity_gate_passed=passed)
    except Exception as e:status('failed',error=str(e));raise
if __name__=='__main__':main()
