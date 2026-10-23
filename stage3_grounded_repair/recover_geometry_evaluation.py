"""Score saved intermediate images after an evaluator-environment recovery.

Never regenerates images. Check the frozen detector hash and reevaluate ALL
previous144 v11 images as a compatibility anchor before accepting v12 scores.
Preserve the original failure status/log and freeze a separate recovery config.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from candidate_provenance import file_hash, freeze_config

ACTIONS = ('original_clean','original_roundtrip','target_pixels','target_roundtrip',
           'opposite_pixels','opposite_roundtrip')


def compare_semantics(original, recovered, path_to_original):
    expected = {r['filename']:r for r in original}
    if len(expected) != len(original):
        raise ValueError('Duplicate original anchor result')
    seen = set()
    for row in recovered:
        filename = path_to_original[str(Path(row['filename']).resolve())]
        if filename in seen:
            raise ValueError('Duplicate recovered anchor result')
        seen.add(filename)
        reference = expected[filename]
        if row['prompt'] != reference['prompt'] or bool(row['correct']) != bool(reference['correct']) or row['reason'] != reference['reason']:
            raise ValueError('Recovered evaluator changes original anchor semantics: '+filename)
    if seen != set(expected):
        raise ValueError('Missing recovered anchor evaluations')
    return {'n_anchors':len(seen),'all_correctness_and_reasons_exact':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--eval-python',type=Path,required=True)
    p.add_argument('--model-config',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--compatibility-source',type=Path,required=True)
    p.add_argument('--recovery-tag',default='v1')
    args=p.parse_args();root=args.work/'stage3_grounded_repair';out=args.output
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.recovery_tag):
        raise ValueError('Recovery tag must be a simple experiment identifier')
    recovery=out/('evaluation_recovery_'+args.recovery_tag);recovery.mkdir(exist_ok=True)
    if (recovery/'config.json').exists():raise ValueError('Use a new recovery script/directory on configuration changes')
    for name in ('diagnostic_status.json','run.log','detector.log'):
        if (out/name).exists():shutil.copy2(out/name,recovery/('pre_recovery_'+name))
    def update(phase,**kw):
        (recovery/'status.json').write_text(json.dumps({'phase':phase,'time':time.time(),**kw},indent=2))
    try:
        checkpoint=args.model_path/'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth'
        frozen=json.loads((root/'object_masks_dev12x2_v6/config.json').read_text())
        expected=frozen['configuration']['detector_checkpoint_sha256']
        if file_hash(checkpoint)!=expected:raise ValueError('Detector weights differ from the frozen original')
        script=args.work/'stage2b/geneval/evaluation/evaluate_images.py'
        env=os.environ.copy();env.update(PYTHONPATH=str(args.work/'deps'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore')
        # Preserve the proven official import order (PIL before mmcv).
        probe="from PIL import Image; import torch,mmcv,mmdet,json; print(json.dumps({'torch':torch.__version__,'mmcv':mmcv.__version__,'mmdet':mmdet.__version__}))"
        version=subprocess.run([str(args.eval_python),'-c',probe],env=env,capture_output=True,text=True,check=True,timeout=120)
        freeze_config(recovery/'config.json',{'recovery_source_sha256':file_hash(__file__),
            'original_generation_config_sha256':file_hash(out/'config.json'),'original_records_sha256':file_hash(out/'records.jsonl'),
            'official_evaluator_sha256':file_hash(script),'model_config_sha256':file_hash(args.model_config),
            'checkpoint_sha256':expected,'eval_python':str(args.eval_python),'versions':version.stdout.strip(),
            'compatibility_source_sha256':file_hash(args.compatibility_source),'no_generation':True,'optimizer_steps':0})
        def run(layout,result,log):
            command=[str(args.eval_python),str(script),str(layout),'--outfile',str(result),
                     '--model-path',str(args.model_path),'--model-config',str(args.model_config)]
            with log.open('w') as stream:
                proc=subprocess.run(command,env=env,cwd=args.work,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
            if proc.returncode:raise RuntimeError('Official evaluator failed; inspect '+str(log))
            return list(map(json.loads,result.read_text().splitlines()))
        original=list(map(json.loads,args.compatibility_source.read_text().splitlines()))
        if len(original)!=144:raise ValueError('Expected all144 completed v11 anchor images')
        anchors=recovery/'compatibility_layout';mapping={}
        for index,row in enumerate(sorted(original,key=lambda r:r['filename'])):
            source=Path(row['filename'])
            if not source.is_file():raise ValueError('Missing original compatibility image')
            directory=anchors/f'{index:06d}';(directory/'samples').mkdir(parents=True)
            image_path=directory/'samples/0000.png';os.symlink(source.resolve(),image_path)
            metadata=json.loads(row['metadata']) if isinstance(row['metadata'],str) else row['metadata']
            (directory/'metadata.jsonl').write_text(json.dumps(metadata)+'\n')
            mapping[str(image_path.resolve())]=row['filename']
        update('compatibility_evaluation',images=len(original))
        anchor_results=run(anchors,recovery/'compatibility_results.jsonl',recovery/'compatibility.log')
        verified=compare_semantics(original,anchor_results,mapping)
        (recovery/'compatibility_report.json').write_text(json.dumps(verified,indent=2))
        records=list(map(json.loads,(out/'records.jsonl').read_text().splitlines()))
        if len(records)!=36 or len({(r['state_id'],r['action']) for r in records})!=36:
            raise ValueError('Expected exactly36 unique six-state saved diagnostic images')
        for row in records:
            if file_hash(row['image_path'])!=row['image_sha256']:raise ValueError('Changed intermediate image')
        update('intermediate_evaluation',images=len(records))
        official=run(out/'official_layout',out/'official_results.jsonl',recovery/'detector.log')
        lookup={str(Path(r['image_path']).resolve()):r for r in records};keyed=[]
        for row in official:
            record=lookup.pop(str(Path(row['filename']).resolve()))
            if row['prompt']!=record['prompt']:raise ValueError('Wrong intermediate prompt')
            keyed.append(dict(row,state_id=record['state_id'],action=record['action']))
        if lookup:raise ValueError('Missing intermediate results')
        (out/'official_results_keyed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in keyed))
        report={'interpretation':'Intermediate geometry/encoder diagnostic only, not terminal/shared-loop success',
            'states':6,'prompts':len({r['prompt_id'] for r in records}),
            'by_action':{a:{'n':sum(r['action']==a for r in keyed),
                'correct':sum(bool(r['correct']) for r in keyed if r['action']==a)} for a in ACTIONS},
            'rows':keyed,'optimizer_steps':0,'evaluator_recovery':verified}
        (out/'geometry_report.json').write_text(json.dumps(report,indent=2))
        (out/'diagnostic_status.json').write_text(json.dumps({'phase':'complete','time':time.time(),'images':36,
            'states':6,'by_action':report['by_action'],'optimizer_steps':0,'recovered_without_regeneration':True},indent=2))
        update('complete',**verified,images=36)
    except Exception as error:
        update('failed',error=str(error));raise


if __name__=='__main__':main()
