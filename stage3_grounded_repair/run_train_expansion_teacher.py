"""Explicit64 TRAIN offline teachers after intermediate-only quality of geometry."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from candidate_provenance import file_hash,freeze_config
from train_geometry_gate import geometry_gate
from launch_grounding_diagnostic import GPUS,write

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'oracle_minimal_expansion_train32x2_v35';out.mkdir()
pilot=r/'train_pilot_expansion_v24';states=r/'states/train_pilot_expansion_v24';masks=r/'train_masks_expansion_v30'
diagnostic=r/'minimal_geometry_expansion_v34';audit=r/'train_expansion_identity_audit_v32.json'
py='/home/shy_23/venvs/show_o2/bin/python';gpus=(0,4,6,7)

def update(phase,**kw):write(out/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_steps=0,controller_started=False,**kw))

try:
    if json.loads((diagnostic/'diagnostic_status.json').read_text())['phase']!='complete':raise ValueError('Intermediate diagnosis incomplete')
    localized={q['state_id'] for q in json.loads((masks/'mask_manifest.json').read_text())['rows'] if q['eligible']}
    gate=geometry_gate(json.loads((diagnostic/'geometry_report.json').read_text()),localized)
    identities={q['state_id']:q['independent_object_instances'] for q in json.loads(audit.read_text())['rows']}
    trigger=sorted(sid for sid in localized if gate[sid] and identities[sid])
    if not trigger:raise ValueError('No valid intermediate teacher; do not generate identical candidates')
    names=('run_train_expansion_teacher.py','build_train_expansion_candidates.py','fm_rebridge.py','runtime.py',
           'candidate_provenance.py','train_geometry_gate.py')
    freeze_config(out/'config.json',dict(expected_states=64,actions_per_state=7,terminal_candidate_rows=448,
        trigger_states=trigger,split='train',terminal_outcomes_used_by_action=False,
        geometry_report_sha256=file_hash(diagnostic/'geometry_report.json'),identity_audit_sha256=file_hash(audit),
        source_sha256={n:file_hash(r/n) for n in names},optimizer_steps=0,
        alias_policy='Identical zero/untriggered states reuse strictly checked baseline; 448 scored rows are not448 independent draws'))
    (out/'source').mkdir()
    for n in names:shutil.copy2(r/n,out/'source'/n)
    update('generation',trigger_states=trigger,expected_rows=448)
    def worker(shard,gpu):
        status=dict(status='waiting_for_gpu',shard=shard,gpu=gpu)
        path=out/f'worker_{shard}.json';write(path,status)
        try:
            deadline=time.time()+1800
            while int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
                if time.time()>deadline:raise TimeoutError('GPU occupied; no other process interrupted')
                time.sleep(20)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(w/'deps'))
            command=[py,str(r/'build_train_expansion_candidates.py'),'--work',str(w),'--pilot',str(pilot),'--states',str(states),
                '--masks',str(masks),'--teacher-diagnostic',str(diagnostic),'--identity-audit',str(audit),
                '--expected-states','64','--output',str(out/f'shard{shard}'),'--shard',str(shard),'--num-shards','4']
            with (out/f'candidate_shard{shard}.log').open('w') as log:
                p=subprocess.Popen(command,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT)
                status.update(status='generation',pid=p.pid);write(path,status)
                if p.wait(timeout=1800):raise RuntimeError('Candidate worker failed')
            status.update(status='generation_complete')
        except Exception as e:status.update(status='failed',error=str(e))
        write(path,status);return status
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(lambda x:worker(*x),enumerate(gpus)))
    success=all(q['status']=='generation_complete' for q in result)
    write(out/'generation_status.json',dict(success=success,expected_images=448,workers=result,optimizer_steps=0))
    if not success:raise RuntimeError('Candidate generation failed; do not evaluate partial cohort')
    update('evaluation')
    official=r/'evaluator_runtime_v1/venv'
    command=[py,str(r/'evaluate_oracle_pipeline_available.py'),'--work',str(w),'--experiment',out.name,
        '--wait-generation','--official-python',str(official/'bin/python'),'--official-model-config',
        str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
        '--official-model-path',str(w/'stage2b/geneval_models'),'--evaluator-verification',
        str(r/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
    with (out/'evaluation.log').open('w') as log:subprocess.run(command,cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True)
    from export_results import export
    export(out)
    update('complete',terminal_candidate_rows=448,trigger_states=trigger,
        next='Strict TRAIN same-state official wrong-to-correct and both HPS/Pick nonnegative; no automatic training')
except Exception as error:update('failed',error=str(error));raise
