"""Finite preparation -> real-model preflight -> 66 endpoints -> four scores."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
from candidate_provenance import file_hash
from export_results import export
from launch_grounding_diagnostic import GPUS, write

WORK = Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT = WORK/'stage3_grounded_repair'
OUT = ROOT/'analytic_readout_trainfit_v56'
PYTHON = '/home/shy_23/venvs/show_o2/bin/python'
ARMS = (('loop2', 0), ('no_loop', 4))


def check_gpu(gpu):
    values = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if values[0].strip() != GPUS[gpu] or int(values[1].strip()) >= 2000:
        raise RuntimeError('GPU occupied/identity mismatch; no other task interrupted')


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], TORCHDYNAMO_DISABLE='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONWARNINGS='ignore', PYTHONPATH=str(WORK/'deps'))
    return env


def aligned_fits():
    """Post-inference TRAIN analysis; targets never enter generator/action choice."""
    import torch
    targets = {row['state_id']: row for row in map(json.loads,
        (ROOT/'teacher_additive_reachability_v50/all_actions.jsonl').read_text().splitlines())
        if row['action'] == 'additive_teacher'}
    cohort = json.loads((OUT/'cohort.json').read_text())
    results = []
    for mode, _ in ARMS:
        records = [json.loads(line) for line in (OUT/mode/'all_actions.jsonl').read_text().splitlines()]
        baselines = {row['state_id']: row for row in records if row['action'] == 'baseline'}
        for row in records:
            sid = row['state_id']; preserve = sid not in cohort['teacher_state_ids']
            source = baselines[sid] if preserve else targets[sid]
            for item in (row, source, baselines[sid]):
                if file_hash(item['final_latent']) != item['final_latent_sha256']:
                    raise ValueError('Posthoc target/final latent bytes changed')
            final = torch.load(row['final_latent'], map_location='cpu', weights_only=False)['latent'].float()
            target = torch.load(source['final_latent'], map_location='cpu', weights_only=False)['latent'].float()
            baseline = torch.load(baselines[sid]['final_latent'], map_location='cpu', weights_only=False)['latent'].float()
            scale = baseline.square().mean() if preserve else (target-baseline).square().mean()
            results.append(dict(mode=mode, state_id=sid, action=row['action'], preservation=preserve,
                target='original baseline' if preserve else 'v50 additive teacher endpoint',
                normalized_relative_target_mse=float((final-target).square().mean()/scale.clamp_min(1e-5))))
    write(OUT/'aligned_endpoint_fit.json', results)


def main():
    if OUT.exists():
        raise ValueError('v56 already exists; do not duplicate or overwrite a frozen trial')
    def status(phase, **kw):
        write(OUT/'pipeline_status.json', dict(phase=phase, time=time.time(), optimizer_updates=0,
            controller_started=False, validation_images=0, **kw))
    try:
        # Tests are meaningful only for conversion and actual zero-core routing.
        env = environment(0)
        test = subprocess.run([PYTHON, '-m', 'unittest', 'test_analytic_readout_operator', '-v'],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        subprocess.run([PYTHON, str(ROOT/'prepare_analytic_readout.py'), '--work', str(WORK),
            '--output', str(OUT)], cwd=WORK, env=env, check=True)
        (OUT/'unit_tests.log').write_text(test.stdout)
        status('validating_fp32_and_zero')
        check_gpu(0)
        with (OUT/'validation.log').open('w') as log:
            subprocess.run([PYTHON, str(ROOT/'validate_analytic_readout.py'), '--work', str(WORK),
                '--experiment', str(OUT)], cwd=WORK, env=environment(0), stdout=log,
                stderr=subprocess.STDOUT, check=True, timeout=1200)
        if json.loads((OUT/'validation_status.json').read_text())['phase'] != 'complete':
            raise ValueError('Actual conversion/zero endpoint validation did not pass')
        status('generating', expected_scoring_rows=66)
        def worker(item):
            mode, gpu = item
            row = dict(mode=mode, gpu=gpu, status='checking_gpu'); path=OUT/(mode+'.json'); write(path, row)
            try:
                check_gpu(gpu)
                command = [PYTHON, str(ROOT/'build_analytic_readout_candidates.py'), '--work', str(WORK),
                    '--experiment', str(OUT), '--output', str(OUT/mode), '--mode', mode]
                with (OUT/(mode+'.log')).open('w') as log:
                    process = subprocess.Popen(command, cwd=WORK, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
                    row.update(pid=process.pid, status='generating'); write(path, row)
                    if process.wait(timeout=2400):
                        raise RuntimeError('Fixed analytical readout generation failed')
                row.update(status='complete')
            except Exception as error:
                row.update(status='failed', error=str(error))
            write(path, row)
            return row
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(worker, ARMS))
        if any(row['status'] != 'complete' for row in rows):
            raise ValueError('One or more generation arms failed; no automatic retry')
        status('evaluating', scoring_rows=66)
        official = ROOT/'evaluator_runtime_v1/venv'
        for mode, _ in ARMS:
            command = [PYTHON, str(ROOT/'evaluate_oracle_pipeline_available.py'), '--work', str(WORK),
                '--experiment', OUT.name+'/'+mode, '--wait-generation', '--official-python', str(official/'bin/python'),
                '--official-model-config', str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
                '--official-model-path', str(WORK/'stage2b/geneval_models'), '--evaluator-verification',
                str(ROOT/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
            with (OUT/(mode+'_evaluation.log')).open('w') as log:
                subprocess.run(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, check=True)
            export(OUT/mode)
            status('evaluating', last_complete_arm=mode, scoring_rows=66)
        aligned_fits()
        status('complete', scoring_rows=66, analytical_fits_deployed=2,
            next='Compare real loops/no_loop, teachers/preservation, evidence and budget; no automatic val/controller')
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
