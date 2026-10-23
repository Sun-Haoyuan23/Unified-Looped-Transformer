"""Distinguish train fit from val transfer; audit BF16 modulation. No optimizer."""
import concurrent.futures,json,os,pathlib,subprocess,time
from launch_grounding_diagnostic import GPUS
from candidate_provenance import file_hash

w=pathlib.Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'token_trainfit_v27';out.mkdir()
state_dir=r/'states/trainfit_v27';state_dir.mkdir()
python='/home/shy_23/venvs/show_o2/bin/python';status=out/'pipeline_status.json'
ids={'train_extra_00021_seed101_i35','train_extra_00021_seed202_i35'}
source={p.stem:p.resolve() for p in (r/'states/train_pilot_minimal_v14').rglob('*_i35.pt')}
if not ids<=set(source):raise ValueError('Trainfit states missing')
for sid in sorted(ids):(state_dir/(sid+'.pt')).symlink_to(source[sid])
(out/'state_source_hashes.json').write_text(json.dumps({sid:file_hash(source[sid]) for sid in sorted(ids)},indent=2))
def update(phase,**kw):status.write_text(json.dumps(dict(phase=phase,time=time.time(),optimizer_steps=0,controller_started=False,**kw),indent=2))
def generate(core,gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCHDYNAMO_DISABLE='1',PYTHONPATH=str(w/'deps'))
    directory=out/f'core{core}';directory.mkdir()
    cmd=[python,str(r/'build_token_trainfit_candidates.py'),'--work',str(w),'--states',str(state_dir),'--output',str(directory),
         '--checkpoint',str(r/f'token_capacity_and_val_v25/core{core}/operator_step50.pt'),'--core-index',str(core),'--level','2',
         '--actions','baseline,conditional_zero,conditional_revisit,shuffled_evidence,opposite_evidence']
    # Same prompt's two seeds cannot provide an independent shuffled prompt.
    # Use correct/zero/opposite on both states; val already has valid shuffling.
    cmd[-1]='baseline,conditional_zero,conditional_revisit,opposite_evidence'
    with (directory/'generation.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT)
        (directory/'worker.json').write_text(json.dumps(dict(pid=p.pid,gpu=gpu,status='running')))
        if p.wait(timeout=1200):raise RuntimeError('Trainfit generation failed')
    (directory/'generation_status.json').write_text(json.dumps(dict(success=True,expected_images=8,optimizer_steps=0)))
try:
    update('generating',states=2,cores=[8,2])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(generate,8,4),pool.submit(generate,2,6)]
        for job in jobs:job.result()
    for core in (8,2):
        update('evaluating',core=core)
        official=r/'evaluator_runtime_v1/venv';directory=out/f'core{core}'
        cmd=[python,str(r/'evaluate_oracle_pipeline_available.py'),'--work',str(w),'--experiment',f'token_trainfit_v27/core{core}',
             '--wait-generation','--official-python',str(official/'bin/python'),'--official-model-config',
             str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
             '--official-model-path',str(w/'stage2b/geneval_models'),'--evaluator-verification',
             str(r/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
        with (directory/'evaluation.log').open('w') as log:subprocess.run(cmd,cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True)
        from export_results import export
        export(directory)
    update('complete',images=16,next='Compare train semantic fit, terminal distance and quantization with val; no controller')
except Exception as error:update('failed',error=str(error));raise
