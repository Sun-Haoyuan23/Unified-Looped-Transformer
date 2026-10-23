"""TRAIN-only offline ridge probes of fixed features, never deployment checkpoints."""
import json
from pathlib import Path
import shutil
import time
import torch
from torch.nn import functional as F
from candidate_provenance import file_hash, freeze_config
from evidence_encoder import condition_tensor
from launch_grounding_diagnostic import GPUS
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator
from runtime import Runtime, tensor_hash, write_json
from state_space_revisit import capture_features, full_model_context

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003');ROOT=WORK/'stage3_grounded_repair'
OUT=ROOT/'process_readout_probe_v55'
PROBES=('initial_no_loop','initial_loop2','trained_no_loop','trained_loop2','trained_nested')


def readout_inputs(op,runtime,features,positions,latent):
    captured={};handles=[];direct={}
    try:
        for name,branch in [('outer',op.outer),('inner',op.inner)]:
            captured[name]=[]
            def hook(module,args,key=name):captured[key].append(args[0].detach().float().cpu())
            handles.append(branch.readout.register_forward_pre_hook(hook))
            hidden,kwargs=features[name];offset,length=positions[0,0].tolist();begin,end=offset+1,offset+length
            count=length-1;side=int(count**.5);assert side*side==count
            line=torch.linspace(-1,1,side,device=hidden.device);yy,xx=torch.meshgrid(line,line,indexing='ij')
            coords=torch.stack((xx,yy),-1).reshape(1,count,2)
            field=F.silu(branch.visual_project(branch.norm(hidden[:1,begin:end]).float())+
                branch.encoder(op.condition)[:,None]+branch.position_project(coords))
            delta=torch.zeros_like(hidden);delta[0,begin:end]=branch.feedback(field)[0].to(hidden.dtype)
            h=hidden+delta
            direct[name]=torch.cat((branch.norm(h[:1,begin:end]).float(),field,coords),-1).cpu()
        op.latent_deltas(runtime,features,positions,latent,(1,2))
        assert all(len(v)==2 for v in captured.values())
        def join(values):
            x=torch.cat([values['outer'],values['inner']],-1).squeeze(0)
            return torch.cat((x,torch.ones(x.shape[0],1)),1)
        return {'no_loop':join(direct),'loop1':join({n:v[0] for n,v in captured.items()}),
                'loop2':join({n:v[1] for n,v in captured.items()})}
    finally:
        for h in handles:h.remove()


def main():
    OUT.mkdir()
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,controller=False,**kw))
    try:
        import subprocess
        q=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if q[0].strip()!=GPUS[0] or int(q[1].strip())>=2000:raise ValueError('GPU0 occupied/identity mismatch; others untouched')
        if json.loads((ROOT/'process_objective_audit_v54/pipeline_status.json').read_text())['phase']!='complete':
            raise ValueError('Require completed process direction diagnosis')
        checkpoint=ROOT/'process_recurrent_train_v52/joint_process/operator_step100.pt'
        cohort_path=ROOT/'process_recurrent_trainfit_v53/cohort.json';cohort=json.loads(cohort_path.read_text())
        names=('probe_process_readout.py','process_training_targets.py','recurrent_training_protocol.py',
            'trainable_recurrent_state.py','state_space_revisit.py','runtime.py','evidence_encoder.py',
            'candidate_provenance.py','training_pairs.py')
        freeze_config(OUT/'config.json',dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            checkpoint_sha256=file_hash(checkpoint),cohort_sha256=file_hash(cohort_path),
            probes=PROBES,ridge=1e-4,column_scaling='weighted column L2 norm',
            fitting_scope='same six TRAIN states, equal state weights and original process denominators',
            nested_scope='one shared linear coefficient fitted jointly on loops1/2; additional prefix constraints',
            analytical_fit_only=True,no_deployed_model_or_checkpoint_update=True,
            target_not_used_for_inference=True,FLOPs=None))
        (OUT/'source').mkdir()
        for n in names:shutil.copy2(ROOT/n,OUT/'source'/n)
        status('loading');runtime=Runtime(WORK,vae=False);corpus=ProcessCorpus(ROOT)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        initial=build_operator(runtime,'joint_ilsd')
        initial_hashes={k:tensor_hash(v) for k,v in initial.adapter_state().items()}
        torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        trained=build_operator(runtime,'joint_ilsd');payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['step']==100 and initial_hashes==payload['metadata']['adapter_initialization_hashes']
        trained.load_state_dict(payload['operator'],strict=False)
        trained_hashes={k:tensor_hash(v) for k,v in trained.adapter_state().items()};cores=trained.core_hashes()
        designs={n:[] for n in PROBES};labels={n:[] for n in PROBES};weights={n:[] for n in PROBES};keys={n:[] for n in PROBES}
        with full_model_context(runtime),torch.no_grad():
            for sid in cohort['state_ids']:
                status('capturing_fixed_features',state_id=sid)
                state=corpus.state(sid);preserve=sid in corpus.correct
                trained.condition=initial.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                features,options=capture_features(runtime,state,trained)
                target=corpus.target(runtime,state,features,trained,preserve)
                original=state['latent'].to(runtime.device).float();gold=(target['input']-original)[:1]
                # Inverse of the actual released unpatchify, not an assumed image patch convention.
                y=gold.flatten(2).transpose(1,2).contiguous().reshape(1,-1,runtime.channels*runtime.patch**2)
                reconstructed=runtime.model.unpatchify(y,original.shape[-2]//runtime.patch,original.shape[-1]//runtime.patch)
                assert torch.equal(reconstructed.transpose(1,2).reshape_as(gold),gold)
                y=y.squeeze(0).cpu();denom=float((original.square().mean() if preserve else gold.square().mean()).clamp_min(1e-5))
                inputs={'initial':readout_inputs(initial,runtime,features,options['modality_positions'],original),
                        'trained':readout_inputs(trained,runtime,features,options['modality_positions'],original)}
                for name in PROBES:
                    source,kind=name.split('_',1);budgets=['loop1','loop2'] if kind=='nested' else [kind]
                    for b in budgets:
                        x=inputs[source][b];assert x.shape[0]==y.shape[0]
                        designs[name].append(x);labels[name].append(y)
                        weights[name].append(torch.full((x.shape[0],1),(denom*len(budgets))**-.5))
                        keys[name].append(dict(state_id=sid,preservation=preserve,budget=b,rows=x.shape[0],denominator=denom))
            results={}
            for name in PROBES:
                status('offline_analytical_fit',probe=name)
                x=torch.cat(designs[name]).to(runtime.device,dtype=torch.float64)
                y=torch.cat(labels[name]).to(runtime.device,dtype=torch.float64)
                w=torch.cat(weights[name]).to(runtime.device,dtype=torch.float64);xw=x*w;yw=y*w
                scale=xw.square().sum(0).sqrt().clamp_min(1e-12);z=xw/scale
                gram=z.T@z;gram.diagonal().add_(1e-4)
                factor=torch.linalg.cholesky(gram);coefficient=torch.cholesky_solve(z.T@yw,factor)/scale[:,None]
                predicted=x@coefficient;assert torch.isfinite(predicted).all()
                error=predicted-y;selected=[];offset=0
                for key in keys[name]:
                    n=key['rows'];e=float(error[offset:offset+n].square().mean()/key['denominator'])
                    selected.append(dict(key,physical_relative_target_mse=e));offset+=n
                torch.save(dict(coefficient=coefficient.cpu(),column_scale=scale.cpu(),ridge=1e-4,
                    diagnostic_only=True,never_loaded_into_deployment_model=True),OUT/(name+'_probe.pt'))
                results[name]=dict(rows=x.shape[0],feature_columns=x.shape[1],physical_patch_dimensions=y.shape[1],
                    coefficient_sha256=file_hash(OUT/(name+'_probe.pt')),states=selected)
                write_json(OUT/'results.json',results)
                del x,y,w,xw,yw,z,gram,factor,coefficient,predicted,error
        assert trained.core_hashes()==cores and {k:tensor_hash(v) for k,v in trained.adapter_state().items()}==trained_hashes
        assert {k:tensor_hash(v) for k,v in initial.adapter_state().items()}==initial_hashes
        assert file_hash(checkpoint)==json.loads((OUT/'config.json').read_text())['configuration']['checkpoint_sha256']
        assert not any(p.grad is not None for p in runtime.model.parameters())
        status('complete',states=6,probes=len(PROBES),analytical_fit_performed=True,base_and_adapter_hashes_unchanged=True,
            model_checkpoint_unchanged=True,limitation='In-sample regularized linear readout feasibility; no generated quality or causal recurrence benefit',FLOPs=None)
    except Exception as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
