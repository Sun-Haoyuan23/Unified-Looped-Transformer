"""Fixed final checkpoints, unchanged recurrence inference, same six TRAIN states."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from candidate_provenance import file_hash,freeze_config
from export_results import export
from launch_grounding_diagnostic import GPUS,write

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT=WORK/'stage3_grounded_repair'
OUT=ROOT/'process_recurrent_trainfit_v53'
TRAIN=ROOT/'process_recurrent_train_v52'
PYTHON='/home/shy_23/venvs/show_o2/bin/python'
ARMS=(('joint_control','joint_ilsd',0),('outer_process','outer_ilsd',4),
      ('inner_process','inner_ilsd',6),('joint_process','joint_ilsd',7))


def aligned_fits():
    # Pure post-inference CPU measurement, never used for action/checkpoint choice.
    import torch
    targets={q['state_id']:q for q in map(json.loads,(ROOT/'teacher_additive_reachability_v50/all_actions.jsonl').read_text().splitlines())
        if q['action']=='additive_teacher'}
    values=[]
    for arm,_,_ in ARMS:
        records=[json.loads(q) for q in (OUT/arm/'all_actions.jsonl').read_text().splitlines()]
        baselines={q['state_id']:q for q in records if q['action']=='baseline'}
        selected=[]
        for row in records:
            sid=row['state_id']
            if sid not in targets:continue
            target_row=targets[sid]
            for q in (row,target_row,baselines[sid]):
                if file_hash(q['final_latent'])!=q['final_latent_sha256']:raise ValueError('Aligned-fit tensor bytes changed')
            final=torch.load(row['final_latent'],map_location='cpu',weights_only=False)['latent'].float()
            target=torch.load(target_row['final_latent'],map_location='cpu',weights_only=False)['latent'].float()
            baseline=torch.load(baselines[sid]['final_latent'],map_location='cpu',weights_only=False)['latent'].float()
            selected.append(dict(arm=arm,state_id=sid,action=row['action'],
                target='v50 additive interface endpoint',
                normalized_relative_target_mse=float((final-target).square().mean()/(target-baseline).square().mean().clamp_min(1e-5))))
        write(OUT/arm/'aligned_endpoint_fit.json',selected)
        values.extend(selected)
    write(OUT/'aligned_endpoint_fit.json',values)


def main():
    OUT.mkdir()
    def status(phase,**kw):
        write(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,controller_started=False,**kw))
    try:
        if json.loads((TRAIN/'pipeline_status.json').read_text())['phase']!='training_complete':
            raise ValueError('Require all four completed final100 training arms')
        if not json.loads((TRAIN/'matched_initialization.json').read_text())['passed']:
            raise ValueError('Matched initialization/sampling proof missing')
        shutil.copy2(ROOT/'recurrent_trainfit_v47/cohort.json',OUT/'cohort.json')
        names=('run_process_trainfit.py','build_recurrent_trainfit_candidates.py','trainable_recurrent_state.py',
            'recurrent_training_protocol.py','runtime.py','state_space_revisit.py','evidence_encoder.py',
            'candidate_provenance.py','training_pairs.py','evaluate_oracle_pipeline.py','evaluate_oracle_pipeline_available.py')
        freeze_config(OUT/'config.json',dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            checkpoint_sha256={a:file_hash(TRAIN/a/'operator_step100.pt') for a,_,_ in ARMS},
            cohort_sha256=file_hash(OUT/'cohort.json'),expected_scoring_rows=144,
            arms=ARMS,checkpoint_selection='fixed final100 in every arm',
            inference_forward='identical v47 builder; process hooks and targets used only during training',
            legacy_fit_target='Original v36 teacher endpoints for continuity; aligned v50 fits exported separately',
            validation_images=0,controller=False,FLOPs=None))
        (OUT/'source').mkdir()
        for n in names:shutil.copy2(ROOT/n,OUT/'source'/n)
        status('generating',expected_scoring_rows=144)
        def worker(item):
            arm,study,gpu=item
            row=dict(arm=arm,study=study,gpu=gpu,status='checking_gpu');path=OUT/(arm+'.json');write(path,row)
            try:
                q=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used',
                    '--format=csv,noheader,nounits'],text=True).strip().split(',')
                if q[0].strip()!=GPUS[gpu] or int(q[1].strip())>=2000:
                    raise RuntimeError('GPU occupied; no other task interrupted')
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],TORCHDYNAMO_DISABLE='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    PYTHONWARNINGS='ignore',PYTHONPATH=str(WORK/'deps'))
                command=[PYTHON,str(ROOT/'build_recurrent_trainfit_candidates.py'),'--work',str(WORK),
                    '--states',str(ROOT/'states/repair_train_targets_v36'),'--teacher-root',str(ROOT/'repair_train_targets_v36'),
                    '--cohort',str(OUT/'cohort.json'),'--checkpoint',str(TRAIN/arm/'operator_step100.pt'),
                    '--study',study,'--output',str(OUT/arm)]
                with (OUT/(arm+'.log')).open('w') as log:
                    proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
                    row.update(pid=proc.pid,status='generating');write(path,row)
                    if proc.wait(timeout=1800):raise RuntimeError('TRAIN diagnostic generation failed')
                row.update(status='complete')
            except Exception as error:row.update(status='failed',error=str(error))
            write(path,row);return row
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            rows=list(pool.map(worker,ARMS))
        if any(q['status']!='complete' for q in rows):raise ValueError('One or more generation arms failed')
        status('evaluating',scoring_rows=144)
        official=ROOT/'evaluator_runtime_v1/venv'
        for arm,_,_ in ARMS:
            command=[PYTHON,str(ROOT/'evaluate_oracle_pipeline_available.py'),'--work',str(WORK),
                '--experiment',OUT.name+'/'+arm,'--wait-generation','--official-python',str(official/'bin/python'),
                '--official-model-config',str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
                '--official-model-path',str(WORK/'stage2b/geneval_models'),'--evaluator-verification',
                str(ROOT/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
            with (OUT/(arm+'_evaluation.log')).open('w') as log:
                subprocess.run(command,cwd=WORK,stdout=log,stderr=subprocess.STDOUT,check=True)
            export(OUT/arm)
            status('evaluating',last_complete_arm=arm,scoring_rows=144)
        aligned_fits()
        status('complete',scoring_rows=144,validation_images=0,
            next='Analyze matched joint control/process, semantics and quality, loop1/2, shuffled/opposite; no automatic val/controller')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
