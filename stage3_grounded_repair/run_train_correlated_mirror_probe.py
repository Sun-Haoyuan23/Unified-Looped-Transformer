"""Finite TRAIN-only intermediate diagnostic -> paired terminal scoring.

No optimizer update, no development teacher, no endpoint-based action selection.
"""
import json,os,subprocess,time,shutil
from pathlib import Path
from candidate_provenance import file_hash
from train_geometry_gate import geometry_gate

work=Path('/home/shy_23/iterative_visual_stage1_20261003')
root=work/'stage3_grounded_repair'
diagnostic=root/'native_mirror_geometry_train10_v21'
out=root/'oracle_correlated_mirror_train24x2_v22'
python='/home/shy_23/venvs/show_o2/bin/python'
out.mkdir(parents=True,exist_ok=True)
status=out/'pipeline_status.json'

def update(phase,**kw):
    status.write_text(json.dumps(dict(phase=phase,time=time.time(),optimizer_steps=0,**kw),indent=2))

try:
    if (diagnostic/'records.jsonl').exists() or (out/'generation_status.json').exists():
        raise ValueError('Use a new experiment directory')
    update('intermediate_diagnostic')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='GPU-4a177c46-c73f-d963-8d61-b7fd90196038',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               PYTHONPATH=str(work/'deps'),PYTHONWARNINGS='ignore')
    with (out/'intermediate_launcher.log').open('w') as log:
        subprocess.run([python,str(root/'diagnose_train_native_mirror.py'),'--work',str(work),
                        '--output',str(diagnostic)],env=env,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
    teachers=json.loads((diagnostic/'teacher_manifest.json').read_text())['rows']
    gate=geometry_gate(json.loads((diagnostic/'geometry_report.json').read_text()),{r['state_id'] for r in teachers})
    audit=json.loads((root/'train_identity_audit_v17.json').read_text())
    ids={r['state_id']:r['independent_object_instances'] for r in audit['rows']}
    eligible=sorted(sid for sid,value in gate.items() if value and ids[sid])
    if not eligible:
        update('complete_without_terminal_candidates',triggered_states=[])
    else:
        source=out/'source';source.mkdir()
        names=('correlated_scene_mirror.py','test_correlated_scene_mirror.py','scene_mirror_teacher.py','test_scene_mirror_teacher.py','diagnose_train_native_mirror.py',
               'build_train_correlated_mirror_candidates.py','launch_train_correlated_mirror.py','run_train_correlated_mirror_probe.py',
               'train_identity_audit_v17.json','train_geometry_gate.py','runtime.py','fm_rebridge.py',
               'candidate_provenance.py','evaluate_oracle_pipeline.py','evaluate_oracle_pipeline_available.py',
               'training_pairs.py')
        for name in names:shutil.copy2(root/name,source/name)
        (out/'source_snapshot.json').write_text(json.dumps({n:file_hash(source/n) for n in names},indent=2))
        update('terminal_generation',triggered_states=eligible,expected_images=288)
        with (out/'generation_launcher.log').open('w') as log:
            subprocess.run([python,str(root/'launch_train_correlated_mirror.py'),'--work',str(work),
                            '--experiment',out.name,'--teacher-diagnostic',str(diagnostic),
                            '--identity-audit',str(root/'train_identity_audit_v17.json'),'--gpus','0,4,6,7'],
                            cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
        update('terminal_evaluation',triggered_states=eligible)
        official=root/'evaluator_runtime_v1/venv'
        command=[python,str(root/'evaluate_oracle_pipeline_available.py'),'--work',str(work),
                 '--experiment',out.name,'--wait-generation','--official-python',str(official/'bin/python'),
                 '--official-model-config',str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
                 '--official-model-path',str(work/'stage2b/geneval_models'),
                 '--evaluator-verification',str(root/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
        with (out/'evaluation_launcher.log').open('w') as log:
            subprocess.run(command,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
        update('complete',triggered_states=eligible,expected_images=288)
except Exception as error:
    update('failed',error=str(error));raise
