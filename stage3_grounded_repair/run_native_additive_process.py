"""Finite v69 process export and actual additive endpoint quality verification."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from candidate_provenance import file_hash,freeze_config
from evaluate_repair_pairwise import read_jsonl,evaluate
from export_results import export
from run_native_teacher_interface import WORK,ROOT,PYTHON,GPU_POOL,check_gpu,environment
from launch_grounding_diagnostic import write

OUT=ROOT/'native_teacher_additive_v69'


def main():
    if OUT.exists():raise ValueError('v69 exists: no overwrite or duplicate run')
    OUT.mkdir()
    def status(phase,**kw):
        write(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,
            controller_started=False,validation_images=0,**kw))
    try:
        status('validating_sources')
        v68=ROOT/'native_teacher_interface_v68'
        if json.loads((v68/'pipeline_status.json').read_text())['phase']!='complete':
            raise ValueError('v68 not complete')
        manifest=json.loads((v68/'artifact_manifest.json').read_text())
        for n,h in manifest.items():
            if file_hash(v68/n)!=h:raise ValueError('v68 frozen artifact changed: '+n)
        prior_config=json.loads((v68/'config.json').read_text())['configuration']
        for n,h in prior_config['source_sha256'].items():
            if file_hash(ROOT/n)!=h:raise ValueError('v68 verified replay/runtime source changed: '+n)
        proof=json.loads((ROOT/'v68_portable_verification.json').read_text())
        if not proof['passed'] or not proof['original_canonical_paired_report_exact']:
            raise ValueError('Require full local portable v68 verification')
        qualified=json.loads((v68/'qualified_native_teachers.json').read_text())['candidates']
        ids=sorted(r['state_id'] for r in qualified)
        if ids!=['train_extra_00006_seed101_i35','train_extra_00010_seed101_i35','train_extra_00054_seed101_i35']:
            raise ValueError('Frozen native qualified cohort differs')
        names=('run_native_additive_process.py','build_native_additive_process.py','native_teacher_interface.py',
            'build_native_teacher_candidates.py','run_native_teacher_interface.py','runtime.py',
            'state_space_revisit.py','evidence_encoder.py','candidate_provenance.py','launch_grounding_diagnostic.py',
            'evaluate_oracle_pipeline.py','evaluate_oracle_pipeline_available.py','evaluate_repair_pairwise.py',
            'score_repair_candidates.py','prepare_geneval_layout.py','export_results.py')
        freeze_config(OUT/'config.json',dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            v68_manifest_sha256=file_hash(v68/'artifact_manifest.json'),
            v68_portable_verification_sha256=file_hash(ROOT/'v68_portable_verification.json'),
            v68_qualified_sha256=file_hash(v68/'qualified_native_teachers.json'),
            protocol_sha256=file_hash(ROOT/'STAGE3_NATIVE_ADDITIVE_PROTOCOL_20261008.md'),
            source_state_ids=ids,split='train',index=35,actions=['baseline','native_additive_teacher'],
            physical_interface='original + float32(native_bridge_input - original)',
            quality_reference='Original canonical source baseline, finite HPS/Pick >=0 and official wrong-to-correct',
            scoring_rows=6,new_images=3,baseline_aliases=3,optimizer_updates=0,
            purpose='Native additive reachability and process target export; not learned loop success'))
        (OUT/'source').mkdir()
        for n in names:shutil.copy2(ROOT/n,OUT/'source'/n)
        shutil.copy2(ROOT/'STAGE3_NATIVE_ADDITIVE_PROTOCOL_20261008.md',OUT/'protocol.md')
        check_gpu(0);status('exporting_and_generating',source_state_ids=ids)
        with (OUT/'generation.log').open('w') as log:
            subprocess.run([PYTHON,str(ROOT/'build_native_additive_process.py')],cwd=WORK,env=environment(0),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        if not json.loads((OUT/'generation_status.json').read_text())['success']:
            raise ValueError('Additive interface failed')
        status('evaluating',scoring_rows=6)
        for gpu in GPU_POOL:check_gpu(gpu)
        official=ROOT/'evaluator_runtime_v1/venv'
        cmd=[PYTHON,str(ROOT/'evaluate_oracle_pipeline_available.py'),'--work',str(WORK),'--experiment',OUT.name,
            '--wait-generation','--official-python',str(official/'bin/python'),'--official-model-config',
            str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
            '--official-model-path',str(WORK/'stage2b/geneval_models'),'--evaluator-verification',
            str(ROOT/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
        with (OUT/'evaluation.log').open('w') as log:
            subprocess.run(cmd,cwd=WORK,env=environment(0),stdout=log,stderr=subprocess.STDOUT,check=True)
        export(OUT)
        records=read_jsonl(OUT/'all_actions.jsonl');scores=read_jsonl(OUT/'scores.jsonl');official_rows=read_jsonl(OUT/'official_results_keyed.jsonl')
        old_scores={(r['state_id'],r['action']):r for r in read_jsonl(ROOT/'repair_train_targets_v36/scores.jsonl')}
        old_official={(r['state_id'],r['action']):r for r in read_jsonl(ROOT/'repair_train_targets_v36/official_results_keyed.jsonl')}
        canonical_scores=[old_scores[r['state_id'],'baseline'] if r['action']=='baseline' else r for r in scores]
        canonical_official=[old_official[r['state_id'],'baseline'] if r['action']=='baseline' else r for r in official_rows]
        report=evaluate(records,canonical_official,canonical_scores)
        write(OUT/'canonical_paired_report.json',report)
        for name,values in [('canonical_scores.jsonl',canonical_scores),('canonical_official_results_keyed.jsonl',canonical_official)]:
            (OUT/name).write_text(''.join(json.dumps(r)+'\n' for r in values))
        write(OUT/'baseline_score_drift.json',[dict(state_id=r['state_id'],**{m:r[m]-old_scores[r['state_id'],'baseline'][m] for m in ('pickscore','hps','siglip')}) for r in scores if r['action']=='baseline'])
        accepted=[r for r in report['pairs'] if r['classification']=='positive_repair']
        write(OUT/'qualified_process_targets.json',dict(candidates=accepted,n_qualified=len(accepted),
            source_state_ids=[r['state_id'] for r in accepted],is_learned_loop_success=False,
            existing_training_corpus_unchanged=True))
        status('complete',scoring_rows=6,qualified_additive_teachers=len(accepted),
            independent_qualified_prompts=len({r['prompt_id'] for r in accepted}),learned_loop_success=False,
            next='Archive/local verify and freeze expanded TRAIN process corpus; no automatic training/val/controller')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
