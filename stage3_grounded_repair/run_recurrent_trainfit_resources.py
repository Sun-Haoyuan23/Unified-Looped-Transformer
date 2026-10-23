"""Freeze six TRAIN states before inference, then score fixed three-arm checkpoints."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from candidate_provenance import file_hash, freeze_config
from training_pairs import selected_pairs, preservation_states, validate_training_pairs
from launch_grounding_diagnostic import GPUS, write

w = Path('/home/shy_23/iterative_visual_stage1_20261003')
r = w/'stage3_grounded_repair'
out = r/'recurrent_trainfit_v47'
py = '/home/shy_23/venvs/show_o2/bin/python'


def update(phase, **kwargs):
    write(out/'pipeline_status.json', dict(phase=phase, time=time.time(), controller_started=False, **kwargs))


def main():
    out.mkdir()
    try:
        training = r/'recurrent_train_v46'; teacher = r/'repair_train_targets_v36'
        summary = json.loads((training/'pipeline_status.json').read_text())
        matched = json.loads((training/'matched_initialization.json').read_text())
        if summary['phase'] != 'training_complete' or summary['optimizer_updates'] != 400 or not matched['passed']:
            raise ValueError('Require all three fixed-step100 arms and matched initialization')
        report = json.loads((teacher/'paired_report.json').read_text())
        records = [json.loads(l) for l in (teacher/'all_actions.jsonl').read_text().splitlines()]
        gate = validate_training_pairs(report, records)
        pairs = selected_pairs(report, records); correct = preservation_states(report, records)
        if gate['selected_teacher_states'] != 3 or gate['preservation_states'] != 35:
            raise ValueError('Strict teacher/preservation cohort changed')
        by = {q['state_id']:q for q in records if q['action'] == 'baseline'}
        # All successful teachers; prefer a correct paired seed, then sorted unseen prompts.
        preserve = []
        for sid in sorted(pairs):
            options = [s for s in correct if by[s]['prompt_id'] == by[sid]['prompt_id'] and s not in preserve]
            if options: preserve.append(options[0])
        used = {by[s]['prompt_id'] for s in list(pairs) + preserve}
        for sid in correct:
            if len(preserve) == 3: break
            if by[sid]['prompt_id'] not in used:
                preserve.append(sid); used.add(by[sid]['prompt_id'])
        if len(preserve) != 3: raise ValueError('Three real baseline-correct diagnostic states missing')
        states = r/'states/repair_train_targets_v36'; ids = sorted(list(pairs) + preserve)
        write(out/'cohort.json', dict(split='train', teacher_state_ids=sorted(pairs), preservation_state_ids=preserve,
            state_ids=ids, state_sha256={sid:file_hash(states/(sid+'.pt')) for sid in ids},
            teacher_report_sha256=file_hash(teacher/'paired_report.json'),
            selection='all3 qualified TRAIN teachers; correct paired seeds then first sorted distinct preservation prompts; frozen before generation',
            inference_external_masks=False, inference_pixel_teacher=False, independent_val=False))
        if file_hash(out/'cohort.json') != file_hash(r/'two_level_trainfit_v39/cohort.json'):
            raise ValueError('RMS diagnosis must use the exact same previously frozen six-state cohort')
        names = ('run_recurrent_trainfit_resources.py','evaluate_oracle_pipeline_two_available.py','run_recurrent_trainfit.py','build_recurrent_trainfit_candidates.py','trainable_recurrent_state.py','recurrent_training_protocol.py','state_space_revisit.py','runtime.py','evidence_encoder.py','training_pairs.py','candidate_provenance.py')
        freeze_config(out/'config.json', dict(training_gate=gate, cohort_sha256=file_hash(out/'cohort.json'),
            checkpoint_sha256={a:file_hash(training/a/'operator_step100.pt') for a in ('outer_ilsd','inner_ilsd','joint_ilsd','joint_terminal')},
            source_sha256={n:file_hash(r/n) for n in names}, expected_images=144,
            checkpoints='fixed final step100 in every arm', val_images=0, controller=False,
            comparison='same TRAIN cohort/objective/common initialization; costs differ, not equal-FLOPs proof'))
        (out/'source').mkdir()
        for n in names: shutil.copy2(r/n, out/'source'/n)
        update('generating', expected_images=144)
        def worker(arm, gpu):
            status = dict(arm=arm, gpu=gpu, status='waiting_for_gpu')
            path = out/(arm+'.json'); write(path, status)
            try:
                deadline = time.time()+6*3600
                while int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], text=True).strip()) >= 2000:
                    if time.time() > deadline: raise TimeoutError('GPU occupied; do not interrupt others')
                    time.sleep(20)
                env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], HF_HUB_OFFLINE='1',
                    TRANSFORMERS_OFFLINE='1', TORCHDYNAMO_DISABLE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    PYTHONWARNINGS='ignore', PYTHONPATH=str(w/'deps'))
                command = [py, str(r/'build_recurrent_trainfit_candidates.py'), '--work', str(w), '--states', str(states),
                    '--teacher-root', str(teacher), '--cohort', str(out/'cohort.json'), '--checkpoint',
                    str(training/arm/'operator_step100.pt'), '--study', arm, '--output', str(out/arm)]
                with (out/(arm+'.log')).open('w') as log:
                    p = subprocess.Popen(command, cwd=w, env=env, stdout=log, stderr=subprocess.STDOUT)
                    status.update(pid=p.pid, status='running'); write(path, status)
                    if p.wait(timeout=1800): raise RuntimeError(arm+' generation failed')
                status.update(status='complete')
            except Exception as error: status.update(status='failed', error=str(error))
            write(path, status); return status
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda q:worker(*q), (('outer_ilsd',6),('inner_ilsd',7),('joint_ilsd',6),('joint_terminal',7))))
        if any(q['status'] != 'complete' for q in results): raise ValueError('One or more fixed-arm generations failed')
        official = r/'evaluator_runtime_v1/venv'
        update('evaluating', terminal_images=144)
        # One arm at a time uses the same four-GPU score resource pool.
        for arm in ('outer_ilsd','inner_ilsd','joint_ilsd','joint_terminal'):
            command = [py, str(r/'evaluate_oracle_pipeline_two_available.py'), '--work', str(w), '--experiment',
                'recurrent_trainfit_v47/'+arm, '--wait-generation', '--official-python', str(official/'bin/python'),
                '--official-model-config', str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
                '--official-model-path', str(w/'stage2b/geneval_models'), '--evaluator-verification',
                str(r/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
            with (out/(arm+'_evaluation.log')).open('w') as log:
                subprocess.run(command, cwd=w, stdout=log, stderr=subprocess.STDOUT, check=True)
            from export_results import export
            export(out/arm)
            update('evaluating', last_completed_arm=arm, terminal_images=144)
        update('complete', terminal_images=144, optimizer_updates=0, val_images=0,
               next='Review semantic and quality TRAIN fit, preservation and shuffled/opposite controls before independent val/controller')
    except Exception as error:
        update('failed', error=str(error)); raise


if __name__ == '__main__':
    main()
