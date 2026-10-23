"""Fixed balanced ridge probes: same three teachers, all 35 true correct TRAIN states."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import torch
from analytic_readout_operator import load_adapter
from candidate_provenance import file_hash, freeze_config
from evidence_encoder import condition_tensor
from launch_grounding_diagnostic import GPUS
from probe_process_readout import readout_inputs
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator
from runtime import Runtime, tensor_hash, write_json
from state_space_revisit import capture_features, full_model_context

WORK = Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT = WORK/'stage3_grounded_repair'
OUT = ROOT/'preservation_readout_probe_v58'


def main():
    OUT.mkdir(exist_ok=False)
    def status(phase, **kw):
        write_json(OUT/'pipeline_status.json', dict(phase=phase, time=time.time(), optimizer_updates=0,
            new_scoring_images=0, controller_started=False, **kw))
    try:
        q = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if q[0].strip()!=GPUS[0] or int(q[1])>=2000 or os.environ['CUDA_VISIBLE_DEVICES']!=GPUS[0]:
            raise ValueError('GPU occupied/UUID mismatch; other tasks untouched')
        if json.loads((ROOT/'analytic_core_contribution_v57/pipeline_status.json').read_text())['phase']!='complete':
            raise ValueError('Require completed fixed-head contribution diagnosis')
        corpus = ProcessCorpus(ROOT)
        ids = sorted(set(corpus.teachers) | set(corpus.correct))
        if len(ids)!=38 or len(corpus.teachers)!=3 or len(corpus.correct)!=35:
            raise ValueError('Strict TRAIN teachers/preservation corpus changed')
        checkpoint = ROOT/'process_recurrent_train_v52/joint_process/operator_step100.pt'
        cohort_path = ROOT/'analytic_readout_trainfit_v56/cohort.json'
        names = ('probe_preservation_readout.py','probe_process_readout.py','process_training_targets.py','analytic_readout_operator.py',
            'trainable_recurrent_state.py','recurrent_training_protocol.py','state_space_revisit.py',
            'runtime.py','evidence_encoder.py','candidate_provenance.py','training_pairs.py')
        freeze_config(OUT/'config.json', dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            checkpoint_sha256=file_hash(checkpoint), diagnostic_cohort_sha256=file_hash(cohort_path),
            state_ids=ids, state_sha256={sid:file_hash(ROOT/'states/repair_train_targets_v36'/(sid+'.pt')) for sid in ids},
            strict_process_corpus_hashes=corpus.input_hashes, modes=['trained_loop2','trained_no_loop'],
            teachers=3, preservation_states=35, ridge=1e-4,
            change_from_v55='Preservation coverage 3 -> all35; total teacher/preservation group weights remain equal',
            group_weights=dict(teachers=0.5,preservation=0.5), within_group_weight='uniform state weight',
            weighted_column_scaling='L2, identical in both branches',
            no_model_or_checkpoint_modification=True, no_automatic_endpoint_generation=True,
            no_val_or_controller=True, FLOPs=None))
        (OUT/'source').mkdir()
        for name in names: shutil.copy2(ROOT/name, OUT/'source'/name)
        status('loading', expected_states=38)
        runtime=Runtime(WORK,vae=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        op=build_operator(runtime,'joint_ilsd')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        load_adapter(op,payload['operator'])
        hashes={k:tensor_hash(v) for k,v in op.adapter_state().items()};cores=op.core_hashes()
        designs={mode:[] for mode in ('trained_loop2','trained_no_loop')}
        labels=[];weights=[];keys=[]
        with torch.no_grad(),full_model_context(runtime):
            for index,sid in enumerate(ids):
                status('capturing_fixed_features', completed_states=index, expected_states=38, state_id=sid)
                state=corpus.state(sid);preserve=sid in corpus.correct
                op.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                features,options=capture_features(runtime,state,op)
                target=corpus.target(runtime,state,features,op,preserve)
                original=state['latent'].to(runtime.device).float();gold=(target['input']-original)[:1]
                y=gold.flatten(2).transpose(1,2).contiguous().reshape(1,-1,runtime.channels*runtime.patch**2)
                check=runtime.model.unpatchify(y,original.shape[-2]//runtime.patch,original.shape[-1]//runtime.patch)
                if not torch.equal(check.transpose(1,2).reshape_as(gold),gold):
                    raise ValueError('Official patch mapping mismatch')
                inputs=readout_inputs(op,runtime,features,options['modality_positions'],original)
                designs['trained_loop2'].append(inputs['loop2']);designs['trained_no_loop'].append(inputs['no_loop'])
                y=y.squeeze(0).cpu();labels.append(y)
                denom=float((original.square().mean() if preserve else gold.square().mean()).clamp_min(1e-5))
                group_size=35 if preserve else 3
                weights.append(torch.full((y.shape[0],1),(denom*group_size)**-.5))
                keys.append(dict(state_id=sid,preservation=preserve,rows=y.shape[0],denominator=denom,
                    within_group_weight=1/group_size))
            results={};y=torch.cat(labels).to(runtime.device,dtype=torch.float64)
            w=torch.cat(weights).to(runtime.device,dtype=torch.float64)
            for mode in designs:
                status('fixed_analytical_fit', mode=mode, completed_states=38)
                x=torch.cat(designs[mode]).to(runtime.device,dtype=torch.float64)
                xw=x*w;yw=y*w;scale=xw.square().sum(0).sqrt().clamp_min(1e-12);z=xw/scale
                gram=z.T@z;gram.diagonal().add_(1e-4)
                factor=torch.linalg.cholesky(gram)
                coefficient=torch.cholesky_solve(z.T@yw,factor)/scale[:,None]
                predicted=x@coefficient
                if not torch.isfinite(predicted).all():raise ValueError('Nonfinite fixed fit')
                selected=[];offset=0
                for key in keys:
                    n=key['rows'];mse=float((predicted[offset:offset+n]-y[offset:offset+n]).square().mean()/key['denominator'])
                    selected.append(dict(key,physical_relative_target_mse=mse));offset+=n
                path=OUT/(mode+'_probe.pt')
                torch.save(dict(coefficient=coefficient.cpu(),column_scale=scale.cpu(),ridge=1e-4,
                    diagnostic_only=True,never_loaded_into_deployment_model=True),path)
                results[mode]=dict(rows=x.shape[0],feature_columns=x.shape[1],physical_patch_dimensions=y.shape[1],
                    coefficient_sha256=file_hash(path),states=selected)
                write_json(OUT/'results.json',results)
                del x,xw,yw,scale,z,gram,factor,coefficient,predicted
        if op.core_hashes()!=cores or {k:tensor_hash(v) for k,v in op.adapter_state().items()}!=hashes:
            raise ValueError('Fixed model or adapter changed')
        if any(p.grad is not None for p in runtime.model.parameters()):raise ValueError('Unexpected base gradient')
        if file_hash(checkpoint)!=json.loads((OUT/'config.json').read_text())['configuration']['checkpoint_sha256']:
            raise ValueError('Original fixed checkpoint bytes changed')
        status('complete', states=38, analytic_fits=2, base_and_adapter_hashes_unchanged=True,
            original_checkpoint_unchanged=True, endpoint_quality_not_measured=True,
            next='Compare three teacher and all35 preservation fits, then freeze separate deployment protocol if justified')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
