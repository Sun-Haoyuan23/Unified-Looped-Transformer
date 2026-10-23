"""Wait for this project's runtime installation, then score saved images only."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
from candidate_provenance import file_hash,freeze_config

p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('--launch-terminal',action='store_true');args=p.parse_args()
work=args.work.resolve();root=work/'stage3_grounded_repair'
runtime=root/'evaluator_runtime_v1';out=root/'minimal_teacher_geometry_dev6_v12'
status=root/'transfers/evaluator_recovery_queue_v12.json'
def update(phase,**kw):
    status.write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2))
try:
    update('waiting_for_runtime')
    deadline=time.time()+5400
    while True:
        s=json.loads((runtime/'setup_status.json').read_text())
        if s['phase']=='failed':raise RuntimeError('Runtime installation failed: '+s['error'])
        if s['phase']=='installed_not_yet_semantically_verified':break
        if time.time()>deadline:raise TimeoutError('Evaluator installation timed out')
        time.sleep(20)
    env=os.environ.copy();env.update(PYTHONPATH=str(work/'deps'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=runtime/'venv/bin/python'
    code="from PIL import Image; import torch,mmcv,mmdet; from mmcv.ops import roi_align; from pathlib import Path; print(Path(mmdet.__file__).parent/'.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py')"
    probe=subprocess.run([str(python),'-c',code],env=env,check=True,capture_output=True,text=True,timeout=120)
    config=Path(probe.stdout.strip().splitlines()[-1])
    if not config.is_file():raise RuntimeError('Official model configuration missing')
    update('waiting_for_free_gpu0',official_python=str(python),model_config=str(config))
    deadline=time.time()+1800
    while True:
        q=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
        if int(q.stdout.strip())<2000:break
        if time.time()>deadline:raise TimeoutError('GPU0 occupied; no process interrupted')
        time.sleep(20)
    env['CUDA_VISIBLE_DEVICES']='GPU-4a177c46-c73f-d963-8d61-b7fd90196038'
    command=[sys.executable,str(root/'recover_geometry_evaluation.py'),'--work',str(work),'--output',str(out),
        '--eval-python',str(python),'--model-config',str(config),'--model-path',str(work/'stage2b/geneval_models'),
        '--compatibility-source',str(root/'oracle_fm_rebridge_dev12x2_v11/official_results.jsonl'),'--recovery-tag','v1']
    update('scoring_saved_images',command=command)
    with (root/'transfers/evaluator_recovery_v12.log').open('w') as log:
        subprocess.run(command,env=env,cwd=work,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=3000)
    update('complete_without_generation',scored_intermediate_images=36,anchor_images=144,optimizer_steps=0)
    if args.launch_terminal:
        report=json.loads((out/'geometry_report.json').read_text())
        lookup={(r['state_id'],r['action']):r for r in report['rows']}
        audit_path=root/'minimal_teacher_identity_audit_v13.json'
        audit=json.loads(audit_path.read_text())
        if audit['terminal_outcomes_used'] is not False:raise ValueError('Identity audit cannot use endpoints')
        eligible=[r['state_id'] for r in audit['rows'] if r['independent_object_instances'] and
            lookup[r['state_id'],'target_pixels']['correct'] and lookup[r['state_id'],'target_roundtrip']['correct']]
        if not eligible:
            update('complete_no_valid_minimal_geometry',intermediate_images=36,anchor_images=144);sys.exit(0)
        experiment='oracle_minimal_teacher_dev12x2_v13';terminal=root/experiment
        if terminal.exists():raise RuntimeError('Terminal directory already exists; do not overwrite')
        terminal.mkdir();snap=terminal/'source';snap.mkdir()
        names=['build_minimal_fm_candidates.py','launch_minimal_fm_oracle.py','evaluate_oracle_pipeline.py',
            'queue_evaluator_recovery.py','recover_geometry_evaluation.py','setup_official_evaluator.py',
            'minimal_pixel_teacher.py','diagnose_minimal_teacher.py','runtime.py','fm_rebridge.py',
            'build_pixel_teacher_candidates.py','candidate_provenance.py','minimal_teacher_identity_audit_v13.json']
        for name in names:shutil.copy2(root/name,snap/name)
        freeze_config(terminal/'launch_config.json',dict(eligible_intermediate_states=eligible,
            teacher_geometry_sha256=file_hash(out/'geometry_report.json'),identity_audit_sha256=file_hash(audit_path),
            source_sha256={n:file_hash(root/n) for n in names},actions=6,states=24,expected_images=144,
            optimizer_steps=0,terminal_outcome_used_by_action=False,
            note='Offline minimal-change teacher continuation; not shared recurrence success'))
        generation=[sys.executable,str(root/'launch_minimal_fm_oracle.py'),'--work',str(work),'--experiment',experiment,
            '--teacher-diagnostic',str(out),'--identity-audit',str(audit_path),'--gpus','0,2,3,4,6,7']
        evaluation=[sys.executable,str(root/'evaluate_oracle_pipeline.py'),'--work',str(work),'--experiment',experiment,
            '--wait-generation','--official-python',str(python),'--official-model-config',str(config),
            '--official-model-path',str(work/'stage2b/geneval_models'),
            '--evaluator-verification',str(out/'evaluation_recovery_v1/compatibility_report.json')]
        processes={}
        for name,command in [('generation',generation),('evaluation',evaluation)]:
            with (terminal/(name+'_launcher.log')).open('w') as log:
                process=subprocess.Popen(command,env=env,cwd=work,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=dict(pid=process.pid,command=command)
        (terminal/'launch.json').write_text(json.dumps(processes,indent=2))
        update('terminal_probe_launched',experiment=experiment,eligible_intermediate_states=eligible,processes=processes)
except Exception as error:
    update('failed',error=str(error));raise
