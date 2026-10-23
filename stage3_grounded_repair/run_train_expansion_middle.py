"""New balanced TRAIN capture -> intermediate decode -> masks. No optimizer."""
import json,os,pathlib,shutil,subprocess,time
from candidate_provenance import file_hash,freeze_config
from launch_grounding_diagnostic import GPUS

w=pathlib.Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'train_expansion_middle_v30';out.mkdir()
status=out/'pipeline_status.json';py='/home/shy_23/venvs/show_o2/bin/python'
pilot=r/'train_pilot_expansion_v24';middle=r/'train_middle_expansion_v30';masks=r/'train_masks_expansion_v30'
def update(phase,**kw):status.write_text(json.dumps(dict(phase=phase,time=time.time(),optimizer_steps=0,**kw),indent=2))
def execute(cmd,label,gpu=None):
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(w/'deps'))
    if gpu is not None:
        deadline=time.time()+1800
        while int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
            if time.time()>deadline:raise TimeoutError('GPU occupied; never stop others')
            time.sleep(20)
        env['CUDA_VISIBLE_DEVICES']=GPUS[gpu]
    with (out/(label+'.log')).open('w') as stream:
        proc=subprocess.Popen(cmd,cwd=w,env=env,stdout=stream,stderr=subprocess.STDOUT)
        (out/(label+'.json')).write_text(json.dumps(dict(pid=proc.pid,gpu=gpu,status='running')))
        if proc.wait(timeout=3600):raise RuntimeError(label+' failed')
    (out/(label+'.json')).write_text(json.dumps(dict(pid=proc.pid,gpu=gpu,status='complete')))
try:
    if pilot.exists():raise ValueError('Existing capture requires inspection; never duplicate launcher')
    names=('run_train_expansion_middle.py','export_train_middle_expansion.py','export_train_grounding_masks_expansion.py',
           'launch_train_capture_expansion.py','prepare_train_expansion.py','test_train_expansion.py','runtime.py','capture_repair_states.py')
    freeze_config(out/'config.json',dict(prompts=32,states=64,split='train',index=35,seeds=[101,202],
        excludes_old_train24=True,source_sha256={n:file_hash(r/n) for n in names},terminal_selection=False,optimizer_steps=0))
    (out/'source').mkdir()
    for n in names:shutil.copy2(r/n,out/'source'/n)
    with (out/'unit_tests.log').open('w') as stream:
        subprocess.run([py,'-m','unittest','test_train_expansion.py'],cwd=r,stdout=stream,stderr=subprocess.STDOUT,check=True)
    update('capture_new_train')
    execute([py,str(r/'launch_train_capture_expansion.py'),'--work',str(w),'--gpus','0,4,6,7','--count','32'],'capture')
    update('export_intermediate')
    execute([py,str(r/'export_train_middle_expansion.py'),'--work',str(w),'--pilot',str(pilot),
        '--states',str(r/'states/train_pilot_expansion_v24'),'--output',str(middle),'--expected-states','64'],'middle',0)
    update('localize_intermediate')
    official=r/'evaluator_runtime_v1/venv'
    execute([str(official/'bin/python'),str(r/'export_train_grounding_masks_expansion.py'),'--work',str(w),
        '--source',str(middle/'clean_records.jsonl'),'--output',str(masks),'--expected-states','64',
        '--pilot-prompts',str(pilot/'train_prompts.json'),'--model-config',
        str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
        '--weights',str(w/'stage2b/geneval_models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth')],'masks',0)
    manifest=json.loads((masks/'mask_manifest.json').read_text())
    update('complete',states=64,localized=manifest['n_eligible'],train_teachers_qualified=0,
        next='Audit intermediate object identities; new-count scene/VAE geometry then same-state terminal quality scoring')
except Exception as error:update('failed',error=str(error));raise
