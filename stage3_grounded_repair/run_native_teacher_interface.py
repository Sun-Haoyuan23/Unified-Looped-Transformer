"""Finite v68 TRAIN native-interface audit, generation and frozen evaluation."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from candidate_provenance import file_hash, freeze_config
from export_results import export
from launch_grounding_diagnostic import GPUS, write
from native_teacher_interface import ACTIONS

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT=WORK/'stage3_grounded_repair'
OUT=ROOT/'native_teacher_interface_v68'
PYTHON='/home/shy_23/venvs/show_o2/bin/python'
GPU_POOL=(0,4,6,7)
EXPECTED_IDS=('train_extra_00003_seed101_i35','train_extra_00006_seed101_i35',
    'train_extra_00009_seed202_i35','train_extra_00010_seed101_i35',
    'train_extra_00019_seed101_i35','train_extra_00054_seed101_i35')


def environment(gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],TORCHDYNAMO_DISABLE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONWARNINGS='ignore',PYTHONPATH=str(WORK/'deps'))
    return env


def check_gpu(gpu):
    q=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    if q[0].strip()!=GPUS[gpu] or int(q[1].strip())>=2000:
        raise ValueError('GPU occupied or identity mismatch; no other process interrupted')


def canonical_analysis():
    """Keep fresh evaluation report AND separately gate against ORIGINAL baseline."""
    from evaluate_repair_pairwise import read_jsonl,evaluate
    records=read_jsonl(OUT/'all_actions.jsonl')
    old_scores={(r['state_id'],r['action']):r for r in read_jsonl(ROOT/'repair_train_targets_v36/scores.jsonl')}
    old_official={(r['state_id'],r['action']):r for r in read_jsonl(ROOT/'repair_train_targets_v36/official_results_keyed.jsonl')}
    scores=read_jsonl(OUT/'scores.jsonl');official=read_jsonl(OUT/'official_results_keyed.jsonl')
    drift=[]
    for r in scores:
        if r['action']=='baseline':
            old=old_scores[r['state_id'],'baseline']
            drift.append(dict(state_id=r['state_id'],**{m:r[m]-old[m] for m in ('pickscore','hps','siglip')}))
    write(OUT/'baseline_score_drift.json',drift)
    canonical_scores=[old_scores[r['state_id'],'baseline'] if r['action']=='baseline' else r for r in scores]
    canonical_official=[old_official[r['state_id'],'baseline'] if r['action']=='baseline' else r for r in official]
    report=evaluate(records,canonical_official,canonical_scores)
    write(OUT/'canonical_paired_report.json',report)
    (OUT/'canonical_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in canonical_scores))
    (OUT/'canonical_official_results_keyed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in canonical_official))
    accepted=[p for p in report['pairs'] if p['action']=='native_final_rebridge' and p['classification']=='positive_repair']
    pairs={r['source_state_id']:r for r in json.loads((OUT/'pairs.json').read_text())}
    audits=[r for path in sorted(OUT.glob('shard*/interface_audit.json')) for r in json.loads(path.read_text())]
    by_id={r['state_id']:r for r in audits}
    candidates=[]
    lookup={(r['state_id'],r['action']):r for r in records}
    for p in accepted:
        sid=p['state_id'];row=lookup[sid,p['action']]
        candidates.append(dict(p,source_pair=pairs[sid],action_record=row,
            source_state_mmu_detected_error=by_id[sid]['source_mmu_detected_error'],
            interface_audit=by_id[sid],retains_source_inferred_noise=True,
            is_offline_TRAIN_teacher=True,is_learned_loop_success=False))
    write(OUT/'qualified_native_teachers.json',dict(candidates=candidates,n_qualified=len(candidates),
        gate='Official wrong-to-correct AND finite HPS/Pick delta >=0 against ORIGINAL canonical source baseline',
        direct_and_additive_are_cross_seed_upper_bounds=True,controller_started=False,
        old_training_corpus_unchanged=True))
    write(OUT/'interface_audit.json',audits)
    return report,candidates


def main():
    if OUT.exists():raise ValueError('v68 already exists; do not overwrite or repeat')
    OUT.mkdir()
    def status(phase,**kw):
        write(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,
            controller_started=False,validation_images=0,**kw))
    try:
        status('validating_sources')
        v67=ROOT/'native_teacher_pair_audit_v67'
        manifest=json.loads((v67/'artifact_manifest.json').read_text())
        # Metadata artifact manifest is a basename -> SHA mapping.
        for name,digest in manifest.items():
            if file_hash(v67/name)!=digest:raise ValueError('v67 metadata artifact changed: '+name)
        old_config=json.loads((v67/'config.json').read_text())
        for name,digest in old_config['canonical_source_files_sha256'].items():
            if file_hash(ROOT/'repair_train_targets_v36'/name)!=digest:
                raise ValueError('Canonical input changed: '+name)
        pairs=sorted([r for r in json.loads((v67/'native_pairs.json').read_text()) if r['native_parent_quality_gate']],
                     key=lambda r:r['source_state_id'])
        if tuple(r['source_state_id'] for r in pairs)!=EXPECTED_IDS:
            raise ValueError('Frozen six-candidate selection differs')
        for r in pairs:
            if r['split']!='train' or r['source_baseline_correct'] or not r['parent_baseline_correct']:
                raise ValueError('Invalid native TRAIN pair')
        names=('run_native_teacher_interface.py','build_native_teacher_candidates.py','native_teacher_interface.py',
            'test_native_teacher_interface.py','runtime.py','state_space_revisit.py','evidence_encoder.py',
            'candidate_provenance.py','launch_grounding_diagnostic.py','evaluate_oracle_pipeline.py',
            'evaluate_oracle_pipeline_available.py','score_repair_candidates.py','evaluate_repair_pairwise.py',
            'prepare_geneval_layout.py','export_results.py')
        transport_paths=('transport/path.py','transport/transport.py','configs/showo2_1.5b_demo_432x432.yaml')
        freeze_config(OUT/'config.json',dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            canonical_actions_sha256=old_config['canonical_source_files_sha256']['all_actions.jsonl'],
            canonical_inputs_sha256=old_config['canonical_source_files_sha256'],
            native_pairs_sha256=file_hash(v67/'native_pairs.json'),v67_manifest_sha256=file_hash(v67/'artifact_manifest.json'),
            protocol_sha256=file_hash(ROOT/'STAGE3_NATIVE_TEACHER_PROTOCOL_20261008.md'),
            official_transport_sha256={n:file_hash(WORK/'Show-o/show-o2'/n) for n in transport_paths},
            source_state_ids=list(EXPECTED_IDS),actions=list(ACTIONS),split='train',index=35,
            formula='epsilon_hat = original - grid_t * fresh_complete_velocity; bridge=(1-grid_t)*epsilon_hat+grid_t*final',
            quality_reference='ORIGINAL canonical source baseline',scoring_rows=36,baseline_aliases=6,
            actual_terminal_rollouts=30,actual_new_decodes=30,
            duplicate_content_controls='checked_zero and parent_direct must strictly reproduce existing endpoints/pixels',
            optimizer_updates=0,purpose='Offline native TRAIN teacher interface, not learned recurrence',
            only_native_final_rebridge_can_expand_teacher_pool=True))
        (OUT/'source').mkdir()
        for n in names:shutil.copy2(ROOT/n,OUT/'source'/n)
        (OUT/'official_transport_source').mkdir()
        for n in transport_paths:shutil.copy2(WORK/'Show-o/show-o2'/n,OUT/'official_transport_source'/Path(n).name)
        shutil.copy2(ROOT/'STAGE3_NATIVE_TEACHER_PROTOCOL_20261008.md',OUT/'protocol.md')
        write(OUT/'pairs.json',pairs)
        from build_native_teacher_candidates import load_pair,read_rows
        baselines={r['state_id']:r for r in read_rows(ROOT/'repair_train_targets_v36/all_actions.jsonl') if r['action']=='baseline'}
        initial=[]
        for pair in pairs:
            values=load_pair(pair,baselines)
            initial.append(dict(state_id=pair['source_state_id'],parent_state_id=pair['native_parent_state_id'],
                actual_source_and_parent_state_and_final_hashes_passed=True,
                source_state_fields=sorted(values[0][0]),same_grid_cfg_prompt_train=True,
                cfg_physical_branches_equal=True,canonical_final_equals_capture=True))
        write(OUT/'metadata_preflight.json',initial)
        with (OUT/'unit_tests.log').open('w') as log:
            subprocess.run([PYTHON,'-m','unittest','test_native_teacher_interface','-v'],cwd=ROOT,
                env=environment(0),stdout=log,stderr=subprocess.STDOUT,check=True)
        status('generating',expected_scoring_rows=36,actual_terminal_rollouts=30)
        def worker(item):
            shard,gpu=item;row=dict(shard=shard,gpu_physical=gpu,gpu_uuid=GPUS[gpu],status='checking_gpu')
            path=OUT/('worker_'+str(shard)+'.json');write(path,row)
            try:
                check_gpu(gpu)
                with (OUT/('generation_shard'+str(shard)+'.log')).open('w') as log:
                    proc=subprocess.Popen([PYTHON,str(ROOT/'build_native_teacher_candidates.py'),'--work',str(WORK),
                        '--experiment',str(OUT),'--shard',str(shard),'--num-shards','4'],cwd=WORK,
                        env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
                    row.update(status='running',pid=proc.pid);write(path,row)
                    if proc.wait(timeout=3600):raise ValueError('Native interface worker failed')
                row.update(status='complete')
            except Exception as error:row.update(status='failed',error=str(error))
            write(path,row);return row
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            workers=list(pool.map(worker,enumerate(GPU_POOL)))
        if any(r['status']!='complete' for r in workers):raise ValueError('Generation failed; no automatic retry or scoring')
        statuses=[json.loads(p.read_text()) for p in sorted(OUT.glob('shard*/generation_status.json'))]
        if len(statuses)!=4 or sum(r['expected_images'] for r in statuses)!=36:
            raise ValueError('Incomplete native candidate coverage')
        write(OUT/'generation_status.json',dict(success=True,expected_images=36,actual_terminal_rollouts=30,
            actual_new_decodes=30,baseline_aliases=6,optimizer_updates=0,
            unique_new_velocity_calls=426,remaining_velocity_calls=420,bridge_construction_calls=6))
        status('evaluating',scoring_rows=36)
        for gpu in GPU_POOL:check_gpu(gpu)
        official=ROOT/'evaluator_runtime_v1/venv'
        command=[PYTHON,str(ROOT/'evaluate_oracle_pipeline_available.py'),'--work',str(WORK),
            '--experiment',OUT.name,'--wait-generation','--official-python',str(official/'bin/python'),
            '--official-model-config',str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
            '--official-model-path',str(WORK/'stage2b/geneval_models'),'--evaluator-verification',
            str(ROOT/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
        with (OUT/'evaluation.log').open('w') as log:
            subprocess.run(command,cwd=WORK,env=environment(0),stdout=log,stderr=subprocess.STDOUT,check=True)
        export(OUT)
        report,candidates=canonical_analysis()
        status('complete',scoring_rows=36,qualified_native_final_rebridge_teachers=len(candidates),
            independent_qualified_prompts=len({r['prompt_id'] for r in candidates}),
            learned_loop_success=False,old_training_corpus_unchanged=True,
            next='Archive and verify; audit native process mapping before any new training; no automatic val/controller')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
