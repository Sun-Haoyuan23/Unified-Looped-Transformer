"""Train-only clean export then official intermediate masks; no optimizer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from launch_grounding_diagnostic import GPUS,write

work=Path(sys.argv[1]).resolve();root=work/'stage3_grounded_repair'
out=root/'train_middle_minimal_v15';out.mkdir(exist_ok=True);status=out/'pipeline_status.json'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[0],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(work/'deps'))
try:
    write(status,dict(phase='waiting_for_gpu0',time=time.time(),optimizer_steps=0))
    deadline=time.time()+1800
    while True:
        q=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
        if int(q.stdout.strip())<2000:break
        if time.time()>deadline:raise TimeoutError('GPU0 occupied; no process interrupted')
        time.sleep(20)
    evaluation=root/'evaluator_runtime_v1/venv'
    commands=[('clean_export',['/home/shy_23/venvs/show_o2/bin/python',str(root/'export_train_middle.py'),'--work',str(work)]),
        ('instance_masks',[str(evaluation/'bin/python'),str(root/'export_train_grounding_masks.py'),'--work',str(work),
            '--source',str(out/'clean_records.jsonl'),'--output',str(root/'train_masks_minimal_v15'),
            '--model-config',str(evaluation/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
            '--weights',str(work/'stage2b/geneval_models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth')])]
    for phase,command in commands:
        write(status,dict(phase=phase,time=time.time(),command=command,optimizer_steps=0))
        with (out/(phase+'.log')).open('w') as log:
            subprocess.run(command,env=env,cwd=work,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    manifest=json.loads((root/'train_masks_minimal_v15/mask_manifest.json').read_text())
    write(status,dict(phase='complete',time=time.time(),states=48,localized=manifest['n_eligible'],
        optimizer_steps=0,next='Train-only minimum geometry/VAE and intermediate identity audit before same-state terminal teacher scoring'))
except Exception as error:
    write(status,dict(phase='failed',time=time.time(),error=str(error),optimizer_steps=0));raise
