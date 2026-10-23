"""Finite dependency watcher: successful fixed training -> TRAIN diagnosis only."""
import json,subprocess,time
from pathlib import Path
from candidate_provenance import freeze_config,file_hash
from launch_grounding_diagnostic import write

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'rms_trainfit_wait_v42';out.mkdir()
names=('wait_rms_trainfit.py','run_rms_two_level_trainfit.py','build_rms_two_level_trainfit_candidates.py','rms_scaled_revisit.py')
freeze_config(out/'config.json',dict(source_sha256={n:file_hash(r/n) for n in names},
    cohort_sha256=file_hash(r/'two_level_trainfit_v39/cohort.json'),fixed_checkpoint_step=100,
    teacher_manifest_sha256=file_hash(r/'repair_train_targets_v36/paired_report.json'),
    val_images=0,controller=False))
try:
    write(out/'pipeline_status.json',dict(phase='waiting_for_fixed_training',controller=False))
    deadline=time.time()+3600
    while True:
        status=json.loads((r/'rms_two_level_train_v41/pipeline_status.json').read_text())
        if status['phase']=='failed':raise RuntimeError('RMS training failed; diagnosis will not run')
        if status['phase']=='training_complete':break
        if time.time()>deadline:raise TimeoutError('Fixed training did not complete in one hour')
        time.sleep(20)
    for n in names:
        if file_hash(r/n)!=json.loads((out/'config.json').read_text())['configuration']['source_sha256'][n]:
            raise ValueError('Frozen RMS diagnosis source changed while waiting')
    write(out/'pipeline_status.json',dict(phase='diagnosing',controller=False))
    with (out/'diagnostic.log').open('w') as log:
        proc=subprocess.Popen(['/home/shy_23/venvs/show_o2/bin/python',str(r/'run_rms_two_level_trainfit.py')],
                              cwd=r,stdout=log,stderr=subprocess.STDOUT)
        write(out/'worker.json',dict(pid=proc.pid,status='running'))
        if proc.wait(timeout=3600):raise RuntimeError('RMS TRAIN diagnostic failed')
    write(out/'pipeline_status.json',dict(phase='complete',controller=False))
except Exception as error:
    write(out/'pipeline_status.json',dict(phase='failed',error=str(error),controller=False));raise
