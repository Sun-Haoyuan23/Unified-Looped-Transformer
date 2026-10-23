"""Recover middle export's compiler environment; never recapture64 states."""
import json,os,pathlib,shutil,subprocess,time
from candidate_provenance import file_hash,freeze_config
from launch_grounding_diagnostic import GPUS

w=pathlib.Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
parent=r/'train_expansion_middle_v30';out=parent/'recovery_v1';out.mkdir()
pilot=r/'train_pilot_expansion_v24';middle=r/'train_middle_expansion_v30';masks=r/'train_masks_expansion_v30'
status=out/'pipeline_status.json';py='/home/shy_23/venvs/show_o2/bin/python'
def update(phase,**kw):
    item=dict(phase=phase,time=time.time(),optimizer_steps=0,**kw);status.write_text(json.dumps(item,indent=2))
    (parent/'recovery_status.json').write_text(json.dumps(item,indent=2))
def execute(cmd,label):
    deadline=time.time()+1800
    while int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
        if time.time()>deadline:raise TimeoutError('GPU occupied; never interrupt others')
        time.sleep(20)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[0],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCHDYNAMO_DISABLE='1',PYTHONPATH=str(w/'deps'))
    with (out/(label+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/(label+'.json')).write_text(json.dumps(dict(pid=p.pid,gpu=0,status='running')))
        if p.wait(timeout=1800):raise RuntimeError(label+' failed')
    (out/(label+'.json')).write_text(json.dumps(dict(pid=p.pid,gpu=0,status='complete')))
try:
    cap=json.loads((pilot/'capture_status.json').read_text())
    if not cap['success'] or cap['captured_states']!=64:raise ValueError('Require completed64 capture')
    if (middle/'clean_records.jsonl').exists():raise ValueError('Partial exported records require explicit review')
    failure=json.loads((parent/'pipeline_status.json').read_text())
    if failure['phase']!='failed' or failure['error']!='middle failed':raise ValueError('Unexpected recovery target')
    freeze_config(out/'config.json',dict(reason='Missing TORCHDYNAMO_DISABLE in original middle launcher triggered functorch.compile import',
        environment_addition={'TORCHDYNAMO_DISABLE':'1'},capture_sha256=file_hash(pilot/'capture_status.json'),
        unchanged_middle_source_sha256=file_hash(r/'export_train_middle_expansion.py'),
        unchanged_mask_source_sha256=file_hash(r/'export_train_grounding_masks_expansion.py'),
        source_sha256=file_hash(__file__),optimizer_steps=0))
    shutil.copy2(__file__,out/'resume_train_expansion_middle.py')
    update('export_intermediate')
    execute([py,str(r/'export_train_middle_expansion.py'),'--work',str(w),'--pilot',str(pilot),
        '--states',str(r/'states/train_pilot_expansion_v24'),'--output',str(middle),'--expected-states','64'],'middle')
    update('localize_intermediate')
    official=r/'evaluator_runtime_v1/venv'
    execute([str(official/'bin/python'),str(r/'export_train_grounding_masks_expansion.py'),'--work',str(w),
        '--source',str(middle/'clean_records.jsonl'),'--output',str(masks),'--expected-states','64',
        '--pilot-prompts',str(pilot/'train_prompts.json'),'--model-config',
        str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
        '--weights',str(w/'stage2b/geneval_models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth')],'masks')
    manifest=json.loads((masks/'mask_manifest.json').read_text())
    update('complete',states=64,localized=manifest['n_eligible'],train_teachers_qualified=0,
        next='Intermediate identity audit; generic64 geometry and same-state terminal quality scoring')
except Exception as error:update('failed',error=str(error));raise
