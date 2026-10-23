"""Wait for exact-fork generation then run official semantics and independent scores.

No training starts here. This pipeline emits an explicit scientific gate report.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from score_repair_candidates import load_action_records,merge_score_rows,read_jsonl,write_jsonl
from prepare_geneval_layout import prepare
from evaluate_repair_pairwise import evaluate
from candidate_provenance import file_hash, freeze_config

GPUS=['GPU-4a177c46-c73f-d963-8d61-b7fd90196038',
      'GPU-d0ecf02c-9378-cb53-a381-8e4bd7fe6362',
      'GPU-388c574b-3a93-c8dd-774a-52cfda4f2e5d',
      'GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--experiment',required=True)
    p.add_argument('--wait-generation',action='store_true')
    p.add_argument('--official-python',type=Path)
    p.add_argument('--official-model-config',type=Path)
    p.add_argument('--official-model-path',type=Path)
    p.add_argument('--evaluator-verification',type=Path)
    args=p.parse_args()
    root=args.work/'stage3_grounded_repair'
    experiment=root/args.experiment
    experiment.mkdir(parents=True,exist_ok=True)
    status_path=experiment/'evaluation_status.json'
    def status(phase,**kw):
        status_path.write_text(json.dumps({'phase':phase,'time':time.time(),**kw},indent=2))
    try:
        overrides=[args.official_python,args.official_model_config,args.official_model_path,args.evaluator_verification]
        if any(overrides) and not all(overrides):
            raise ValueError('Recovered evaluator requires all explicit paths and a verified compatibility report')
        if all(overrides):
            verification=json.loads(args.evaluator_verification.read_text())
            if verification.get('n_anchors')!=144 or verification.get('all_correctness_and_reasons_exact') is not True:
                raise ValueError('All144 original anchors must be verified before terminal evaluation')
            recovery=json.loads((args.evaluator_verification.parent/'config.json').read_text())['configuration']
            if str(args.official_python)!=recovery['eval_python'] or file_hash(args.official_model_config)!=recovery['model_config_sha256']:
                raise ValueError('Evaluator runtime/config differ from compatibility verification')
            checkpoint=args.official_model_path/'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth'
            if file_hash(checkpoint)!=recovery['checkpoint_sha256'] or file_hash(args.work/'stage2b/geneval/evaluation/evaluate_images.py')!=recovery['official_evaluator_sha256']:
                raise ValueError('Evaluator source/weights differ from compatibility verification')
            freeze_config(experiment/'evaluation_config.json',dict(
                source_sha256=file_hash(__file__),verification_sha256=file_hash(args.evaluator_verification),
                recovery_config_sha256=file_hash(args.evaluator_verification.parent/'config.json'),
                official_python=str(args.official_python),model_config_sha256=recovery['model_config_sha256'],
                detector_sha256=recovery['checkpoint_sha256']))
        if args.wait_generation:
            status('waiting_for_generation')
            deadline=time.time()+7200
            while not (experiment/'generation_status.json').exists():
                if time.time()>deadline:raise TimeoutError('Generation did not finish within two hours')
                time.sleep(20)
            result=json.loads((experiment/'generation_status.json').read_text())
            if not result['success']:raise RuntimeError('Failed generation worker; evaluation not started')
        records=load_action_records([experiment])
        write_jsonl(experiment/'all_actions.jsonl',records)
        prepare(records,experiment/'official_layout')
        status('evaluating',images=len(records))
        base=os.environ.copy();base.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TORCHDYNAMO_DISABLE='1',PYTHONWARNINGS='ignore',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            PYTHONPATH=str(args.work/'deps'))
        jobs=[('official',GPUS[0],[str(args.official_python or '/home/shy_23/calm_geneval_20260930/eval_venv/bin/python'),
            str(args.work/'stage2b/geneval/evaluation/evaluate_images.py'),
            str(experiment/'official_layout'),'--outfile',str(experiment/'official_results.jsonl'),
            '--model-path',str(args.official_model_path or '/home/shy_23/calm_geneval_20260930/detector'),
            '--model-config',str(args.official_model_config or '/home/shy_23/calm_geneval_20260930/eval_venv/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py')])]
        for metric,gpu in zip(['pickscore','hps','siglip'],GPUS[1:]):
            jobs.append((metric,gpu,['/home/shy_23/venvs/show_o2/bin/python',
                str(root/'score_repair_candidates.py'),'--records',str(experiment),
                '--work',str(args.work),'--metrics',metric,'--batch-size','4',
                '--output',str(experiment/f'{metric}_scores.jsonl')]))
        def run(job):
            name,gpu,cmd=job
            env=base.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            with (experiment/f'eval_{name}.log').open('w') as output:
                proc=subprocess.run(cmd,cwd=args.work,env=env,stdout=output,stderr=subprocess.STDOUT)
            if proc.returncode:raise RuntimeError(f'{name} evaluation failed; see log')
            return name
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            completed=list(pool.map(run,jobs))
        scores=merge_score_rows(records,[read_jsonl(experiment/f'{m}_scores.jsonl')
            for m in ['pickscore','hps','siglip']],require_metrics=('pickscore','hps','siglip'))
        write_jsonl(experiment/'scores.jsonl',scores)
        result=evaluate(records,read_jsonl(experiment/'official_results.jsonl'),scores)
        (experiment/'paired_report.json').write_text(json.dumps(result,indent=2))
        write_jsonl(experiment/'paired_results.jsonl',result['pairs'])
        guarded=result['offline_oracle']['quality_guarded_semantic']
        status('complete',images=len(records),prompts=result['n_prompts'],
            states=result['n_states'],controller_started=False,
            next='Review H1 repairability; independent repairer validation required before controller',
            quality_guarded_semantic_oracle=guarded)
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
