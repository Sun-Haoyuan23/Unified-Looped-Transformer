"""Same 38 TRAIN states: fixed old/new heads and QR/SVD preservation separation."""
import json
import math
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
from preservation_geometry import preservation_geometry
from probe_process_readout import readout_inputs
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator
from runtime import Runtime, tensor_hash, write_json
from state_space_revisit import capture_features, full_model_context

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT=WORK/'stage3_grounded_repair'
OUT=ROOT/'preservation_rank_audit_v59'
MODES=('trained_loop2','trained_no_loop')


def main():
    OUT.mkdir(exist_ok=False)
    def status(phase,**kw):
        write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,
            new_scoring_images=0,analytical_fits=0,controller_started=False,**kw))
    try:
        values=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'],text=True).strip().split(',')
        if values[0].strip()!=GPUS[0] or int(values[1])>=2000 or os.environ['CUDA_VISIBLE_DEVICES']!=GPUS[0]:
            raise ValueError('GPU0 occupied/UUID mismatch; no other task interrupted')
        old_root=ROOT/'process_readout_probe_v55';new_root=ROOT/'preservation_readout_probe_v58'
        for directory in (old_root,new_root):
            if json.loads((directory/'pipeline_status.json').read_text())['phase']!='complete':
                raise ValueError('Require completed frozen old/new probes')
        corpus=ProcessCorpus(ROOT);ids=sorted(set(corpus.teachers)|set(corpus.correct))
        if len(ids)!=38 or len(corpus.teachers)!=3 or len(corpus.correct)!=35:
            raise ValueError('Strict TRAIN corpus changed')
        cohort_path=ROOT/'analytic_readout_trainfit_v56/cohort.json'
        cohort=json.loads(cohort_path.read_text())
        old_config=json.loads((old_root/'config.json').read_text())['configuration']
        new_config=json.loads((new_root/'config.json').read_text())['configuration']
        checkpoint=ROOT/'process_recurrent_train_v52/joint_process/operator_step100.pt'
        checkpoint_sha=file_hash(checkpoint)
        if (old_config['checkpoint_sha256']!=checkpoint_sha or new_config['checkpoint_sha256']!=checkpoint_sha
                or ids!=new_config['state_ids'] or new_config['strict_process_corpus_hashes']!=corpus.input_hashes):
            raise ValueError('Frozen checkpoint/data source changed')
        names=('audit_preservation_rank.py','preservation_geometry.py','test_preservation_geometry.py',
            'run_preservation_rank_audit.py','analytic_readout_operator.py','probe_process_readout.py',
            'process_training_targets.py','recurrent_training_protocol.py','trainable_recurrent_state.py',
            'state_space_revisit.py','runtime.py','candidate_provenance.py','evidence_encoder.py','training_pairs.py')
        probe_paths={version:{mode:directory/(mode+'_probe.pt') for mode in MODES}
            for version,directory in (('old_v55',old_root),('new_v58',new_root))}
        reports={version:json.loads((directory/'results.json').read_text())
            for version,directory in (('old_v55',old_root),('new_v58',new_root))}
        for version,paths in probe_paths.items():
            for mode,path in paths.items():
                if file_hash(path)!=reports[version][mode]['coefficient_sha256']:
                    raise ValueError('Frozen coefficient changed')
        state_hashes={sid:file_hash(ROOT/'states/repair_train_targets_v36'/(sid+'.pt')) for sid in ids}
        if state_hashes!=new_config['state_sha256']:
            raise ValueError('Frozen 38 state bytes changed')
        config=dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            checkpoint_sha256=checkpoint_sha,cohort_sha256=file_hash(cohort_path),
            input_config_sha256=dict(v55=file_hash(old_root/'config.json'),v58=file_hash(new_root/'config.json')),
            probe_sha256={version:{mode:file_hash(path) for mode,path in paths.items()} for version,paths in probe_paths.items()},
            process_corpus_hashes=corpus.input_hashes,state_ids=ids,state_sha256=state_hashes,
            original_preservation_ids=cohort['preservation_state_ids'],teachers=3,preservation_states=35,
            reference_verification_tolerance=dict(rtol=1e-8,atol=1e-10),
            geometry='FP64 thin QR then gesvd SVD of R, no Gram-based rank; v58 weighted column scaling',
            rank_tolerance='largest_preservation_singular * max(rows,columns) * eps; FP64 and FP32 conventions',
            projected_teacher_resolution='same absolute tolerance as preservation SVD; no relative-to-roundoff rank',
            zero_new_readout_fits=True,no_model_or_checkpoint_modification=True,
            no_endpoint_generation_or_quality_inference=True,validation_images=0,FLOPs=None)
        freeze_config(OUT/'config.json',config)
        (OUT/'source').mkdir()
        for n in names:shutil.copy2(ROOT/n,OUT/'source'/n)
        (OUT/'input_snapshots').mkdir()
        for version,directory in (('old_v55',old_root),('new_v58',new_root)):
            for name in ('config.json','results.json'):
                shutil.copy2(directory/name,OUT/'input_snapshots'/(version+'_'+name))
        status('loading',expected_states=38)
        runtime=Runtime(WORK,vae=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        op=build_operator(runtime,'joint_ilsd')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['step']!=100:raise ValueError('Expected fixed final optimizer step100')
        load_adapter(op,payload['operator'])
        adapter_hash={k:tensor_hash(v) for k,v in op.adapter_state().items()}
        base_hash={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}
        probes={version:{mode:torch.load(path,map_location='cpu',weights_only=False) for mode,path in paths.items()}
            for version,paths in probe_paths.items()}
        designs={mode:[] for mode in MODES};labels=[];weights=[];keys=[];fingerprints=[]
        with torch.no_grad(),full_model_context(runtime):
            for index,sid in enumerate(ids):
                status('capturing_fixed_features',completed_states=index,expected_states=38,state_id=sid)
                state=corpus.state(sid);preserve=sid in corpus.correct
                op.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                features,options=capture_features(runtime,state,op)
                target=corpus.target(runtime,state,features,op,preserve)
                original=state['latent'].to(runtime.device).float();gold=(target['input']-original)[:1]
                y=gold.flatten(2).transpose(1,2).contiguous().reshape(1,-1,runtime.channels*runtime.patch**2)
                reconstructed=runtime.model.unpatchify(y,original.shape[-2]//runtime.patch,original.shape[-1]//runtime.patch)
                if not torch.equal(reconstructed.transpose(1,2).reshape_as(gold),gold):
                    raise ValueError('Official patch roundtrip failed')
                inputs=readout_inputs(op,runtime,features,options['modality_positions'],original)
                for mode,kind in (('trained_loop2','loop2'),('trained_no_loop','no_loop')):
                    designs[mode].append(inputs[kind])
                y=y.squeeze(0).cpu();labels.append(y)
                denom=float((original.square().mean() if preserve else gold.square().mean()).clamp_min(1e-5))
                weights.append(torch.full((y.shape[0],1),(denom*(35 if preserve else 3))**-.5))
                keys.append(dict(state_id=sid,preservation=preserve,rows=y.shape[0],denominator=denom))
                fingerprints.append(dict(state_id=sid,target_patch_sha256=tensor_hash(y),
                    feature_sha256={mode:tensor_hash(designs[mode][-1]) for mode in MODES}))
            write_json(OUT/'design_fingerprints.json',fingerprints)
            y=torch.cat(labels).to(runtime.device,dtype=torch.float64)
            w=torch.cat(weights).to(runtime.device,dtype=torch.float64)
            preserve_mask=torch.cat([torch.full((key['rows'],),key['preservation'],dtype=torch.bool) for key in keys]).to(runtime.device)
            per_state=[];reference_checks=[];geometry={}
            for mode in MODES:
                status('paired_fixed_head_predictions',mode=mode,completed_states=38)
                x=torch.cat(designs[mode]).to(runtime.device,dtype=torch.float64)
                for version in probes:
                    coefficient=probes[version][mode]['coefficient'].to(runtime.device)
                    if coefficient.dtype!=torch.float64 or coefficient.shape!=(x.shape[1],y.shape[1]):
                        raise ValueError('Wrong fixed coefficient dimensions/dtype')
                    predicted=x@coefficient
                    if not torch.isfinite(predicted).all():raise ValueError('Nonfinite fixed head prediction')
                    reference={r['state_id']:r for r in reports[version][mode]['states']}
                    offset=0
                    for key in keys:
                        n=key['rows'];value=float((predicted[offset:offset+n]-y[offset:offset+n]).square().mean()/key['denominator'])
                        row=dict(key,mode=mode,version=version,physical_relative_target_mse=value)
                        per_state.append(row)
                        if key['state_id'] in reference:
                            expected=reference[key['state_id']]['physical_relative_target_mse']
                            if not math.isclose(value,expected,rel_tol=1e-8,abs_tol=1e-10):
                                raise ValueError('Frozen scalar fit reproduction failed: '+str(row))
                            reference_checks.append(dict(state_id=key['state_id'],mode=mode,version=version,
                                expected=expected,observed=value,absolute_error=abs(value-expected)))
                        offset+=n
                    del predicted,coefficient
                write_json(OUT/'paired_predictions.json',per_state)
                write_json(OUT/'reference_checks.json',reference_checks)
                scale=probes['new_v58'][mode]['column_scale'].to(runtime.device)
                calculated_scale=(x*w).square().sum(0).sqrt().clamp_min(1e-12)
                torch.testing.assert_close(calculated_scale,scale,rtol=1e-10,atol=1e-12)
                z=x*w/scale
                p=z[preserve_mask];t=z[~preserve_mask];yt=(y*w)[~preserve_mask]
                del x,z,calculated_scale
                teacher_keys=[key for key in keys if not key['preservation']]
                result,artifact=preservation_geometry(p,t,yt,[key['state_id'] for key in teacher_keys],
                    [key['rows'] for key in teacher_keys],
                    notify=lambda phase:status('geometry_'+phase,mode=mode,completed_states=38))
                path=OUT/(mode+'_spectra.pt');torch.save(artifact,path)
                geometry[mode]=dict(result,artifact_sha256=file_hash(path),
                    column_scale_sha256=tensor_hash(probes['new_v58'][mode]['column_scale']))
                write_json(OUT/'geometry_results.json',geometry)
                del p,t,yt,artifact
            if len(reference_checks)!=88:raise ValueError('Expected 12 old plus76 new scalar reference checks')
        if base_hash!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}:
            raise ValueError('Base model tensor changed')
        if adapter_hash!={k:tensor_hash(v) for k,v in op.adapter_state().items()}:
            raise ValueError('Frozen adapter changed')
        if any(p.grad is not None for p in runtime.model.parameters()) or any(p.grad is not None for p in op.trainable_parameters()):
            raise ValueError('Unexpected gradients in read-only audit')
        for version,paths in probe_paths.items():
            for mode,path in paths.items():
                if file_hash(path)!=config['probe_sha256'][version][mode]:raise ValueError('Original coefficient bytes changed')
        if file_hash(checkpoint)!=checkpoint_sha:raise ValueError('Original checkpoint bytes changed')
        status('complete',states=38,paired_prediction_records=len(per_state),reference_scalar_checks=88,
            geometry_modes=2,subspace_projection_diagnostics=4,
            full_base_and_adapter_hashes_unchanged=True,base_and_adapter_no_grad=True,
            original_checkpoint_and_coefficients_unchanged=True,
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
            no_new_semantic_or_quality_measurement=True)
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
