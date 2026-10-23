"""Finite v50 decode then original official/preference evaluation, no training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from candidate_provenance import file_hash, freeze_config
from export_results import export
from launch_grounding_diagnostic import GPUS,write


def main():
    work=Path('/home/shy_23/iterative_visual_stage1_20261003')
    root=work/'stage3_grounded_repair'
    out=root/'teacher_additive_reachability_v50'
    out.mkdir()
    py='/home/shy_23/venvs/show_o2/bin/python'
    def status(phase,**kw):
        write(out/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,
            controller_started=False,**kw))
    try:
        for gpu in (0,4,6,7):
            result=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used',
                '--format=csv,noheader,nounits'],text=True).strip().split(',')
            if result[0].strip()!=GPUS[gpu] or int(result[1].strip())>=2000:
                raise RuntimeError('Required GPU occupied or identity changed; no other task interrupted')
        names=('run_additive_teacher_probe.py','build_additive_teacher_probe.py','runtime.py',
            'candidate_provenance.py','training_pairs.py','evaluate_oracle_pipeline.py',
            'evaluate_oracle_pipeline_available.py','score_repair_candidates.py',
            'evaluate_repair_pairwise.py','prepare_geneval_layout.py','export_results.py')
        freeze_config(out/'config.json',dict(source_sha256={n:file_hash(root/n) for n in names},
            process_status_sha256=file_hash(root/'teacher_process_targets_v49/pipeline_status.json'),
            purpose='Additive numerical teacher reachability, not learned recurrence',
            optimizer_updates=0,new_images=3,baseline_aliases=3,scoring_rows=6))
        (out/'source').mkdir()
        for n in names:shutil.copy2(root/n,out/'source'/n)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[0],TORCHDYNAMO_DISABLE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            PYTHONWARNINGS='ignore',PYTHONPATH=str(work/'deps'))
        status('decoding',new_images=3)
        with (out/'generation.log').open('w') as log:
            subprocess.run([py,str(root/'build_additive_teacher_probe.py'),'--work',str(work),
                '--output',str(out/'generated')],env=env,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
        shutil.copy2(out/'generated/generation_status.json',out/'generation_status.json')
        status('evaluating',scoring_rows=6,new_images=3,baseline_aliases=3)
        official=root/'evaluator_runtime_v1/venv'
        command=[py,str(root/'evaluate_oracle_pipeline_available.py'),'--work',str(work),
            '--experiment',out.name,'--wait-generation','--official-python',str(official/'bin/python'),
            '--official-model-config',str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
            '--official-model-path',str(work/'stage2b/geneval_models'),'--evaluator-verification',
            str(root/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
        with (out/'evaluation.log').open('w') as log:
            subprocess.run(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        export(out)
        # Audit baseline scoring drift explicitly; never silently relabel the
        # known original teacher scores or interpret aliases as new examples.
        old={(q['state_id'],q['action']):q for q in map(json.loads,(root/'repair_train_targets_v36/scores.jsonl').read_text().splitlines())}
        new={(q['state_id'],q['action']):q for q in map(json.loads,(out/'scores.jsonl').read_text().splitlines())}
        drift={sid:{m:new[sid,'baseline'][m]-old[sid,'baseline'][m] for m in ('pickscore','hps','siglip')}
            for sid,action in new if action=='baseline'}
        write(out/'baseline_score_drift.json',drift)
        status('complete',scoring_rows=6,new_images=3,baseline_aliases=3,
            interpretation='Numerical teacher interface diagnostic; not trained-loop repair')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
