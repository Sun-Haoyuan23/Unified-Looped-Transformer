"""Score partial predicted-clean images with the existing official detector.

These scores diagnose object visibility and location at a flow time. They are
not terminal GenEval gains, successful repair teachers, or an understanding
benchmark. All 96 saved development states are retained without outcome selection.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time
from candidate_provenance import file_hash,freeze_config


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows,official):
    if len(rows)!=len(official):raise ValueError('Incomplete official diagnostic')
    indexed={str(Path(row['filename']).resolve()):row for row in official}
    output=[]
    for row in rows:
        result=indexed[str(Path(row['clean_image']).resolve())]
        if result['prompt']!=row['prompt']:raise ValueError('Wrong prompt scored')
        boxes=json.loads(result['details']) if isinstance(result['details'],str) else result['details']
        entry={key:row[key] for key in ('state_id','prompt_id','seed','flow_index','true_t','original_evidence','reflected_evidence')}
        entry.update(official_clean_correct=bool(result['correct']),official_reason=result['reason'],objects={})
        for role in ('subject','reference'):
            item=row['objects'][role];name=item['object'];found=boxes.get(name,[])
            obj={'name':name,'mmu_present_margin':item['present'],'detected':bool(found),
                 'mmu_horizontal_margin':item['left_minus_right'],'mmu_vertical_margin':item['top_minus_bottom']}
            if found:
                box=found[0];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
                obj.update(box=box,normalized_center=[cx/432,cy/432],
                    horizontal_half_agreement=(item['left_minus_right']>0)==(cx<216),
                    vertical_half_agreement=(item['top_minus_bottom']>0)==(cy<216))
            entry['objects'][role]=obj
        output.append(entry)
    by_time={}
    for index in sorted({row['flow_index'] for row in output}):
        selected=[r for r in output if r['flow_index']==index]
        objects=[o for r in selected for o in r['objects'].values()]
        detected=[o for o in objects if o['detected']]
        by_time[index]={'n_states':len(selected),'n_prompts':len({r['prompt_id'] for r in selected}),
            'official_clean_correct':sum(r['official_clean_correct'] for r in selected),
            'both_objects_detected':sum(all(o['detected'] for o in r['objects'].values()) for r in selected),
            'objects_detected':len(detected),'objects_requested':len(objects),
            'horizontal_half_agreement':sum(o['horizontal_half_agreement'] for o in detected)/len(detected) if detected else None,
            'vertical_half_agreement':sum(o['vertical_half_agreement'] for o in detected)/len(detected) if detected else None,
            'mirror_opposite_sign_count':sum(r['original_evidence']*r['reflected_evidence']<0 for r in selected),
            'mirror_mean_abs_sum':sum(abs(r['original_evidence']+r['reflected_evidence']) for r in selected)/len(selected)}
    return {'interpretation':'Architecture-development grounding diagnostic only; detector is an external oracle and partial-image scores are not terminal improvements',
        'n_states':len(output),'n_prompts':len({r['prompt_id'] for r in output}),
        'optimizer_steps':0,'by_flow_index':by_time,'rows':output,
        'limitations':['MMU half-image questions are not instance segmentation','Latent reflection is not exact pixel reflection',
                      'Mirror sign reversal is descriptive and does not define ground truth on overlapping or missing objects',
                      'Repeated flow times share prompt and seed; 12 independent prompt clusters only']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True)
    args=p.parse_args();root=args.diagnostic;root.mkdir(parents=True,exist_ok=True)
    status=root/'evaluation_status.json'
    def update(phase,**extra):status.write_text(json.dumps({'phase':phase,'time':time.time(),**extra},indent=2))
    try:
        update('waiting_for_grounding');deadline=time.time()+2400
        while not (root/'diagnostic_status.json').exists():
            if time.time()>deadline:raise TimeoutError('Grounding workers did not finish')
            time.sleep(20)
        state=json.loads((root/'diagnostic_status.json').read_text())
        if not state['success']:raise RuntimeError('Grounding worker failed; do not score incomplete output')
        rows=sorted([r for path in root.glob('grounding_shard*.jsonl') for r in read_rows(path)],key=lambda r:r['state_id'])
        if len(rows)!=96 or len({r['state_id'] for r in rows})!=96:raise ValueError('Expected all 96 saved states')
        for row in rows:
            if file_hash(row['clean_image'])!=row['clean_image_sha256']:raise ValueError('Partial image changed')
        (root/'all_grounding.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        freeze_config(root/'evaluation_config.json',{'source_sha256':file_hash(__file__),
            'all_grounding_sha256':file_hash(root/'all_grounding.jsonl'),'expected_states':96})
        layout=root/'official_layout';layout.mkdir(exist_ok=True)
        for i,row in enumerate(rows):
            source=Path(row['clean_image']);folder=layout/f'{i:06d}';(folder/'samples').mkdir(parents=True,exist_ok=True)
            link=folder/'samples/0000.png'
            if not link.exists():link.symlink_to(source.resolve())
            if link.resolve()!=source.resolve():raise ValueError('Changed layout symlink')
            (folder/'metadata.jsonl').write_bytes((source.parent.parent/'metadata.jsonl').read_bytes())
        update('waiting_for_detector_gpu',images=len(rows));deadline=time.time()+1200
        while True:
            q=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if int(q.stdout.strip())<2000:break
            if time.time()>deadline:raise TimeoutError('Detector GPU occupied; no process killed')
            time.sleep(20)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-4a177c46-c73f-d963-8d61-b7fd90196038',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        cmd=['/home/shy_23/calm_geneval_20260930/eval_venv/bin/python',str(args.work/'stage2b/geneval/evaluation/evaluate_images.py'),
            str(layout),'--outfile',str(root/'official_results.jsonl'),'--model-path','/home/shy_23/calm_geneval_20260930/detector',
            '--model-config','/home/shy_23/calm_geneval_20260930/eval_venv/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py']
        update('evaluating',images=len(rows))
        with (root/'detector.log').open('w') as out:
            proc=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=1800)
        if proc.returncode:raise RuntimeError('Official detector failed')
        report=summarize(rows,read_rows(root/'official_results.jsonl'))
        (root/'grounding_report.json').write_text(json.dumps(report,indent=2))
        update('complete',images=len(rows),prompts=report['n_prompts'],optimizer_steps=0)
    except Exception as error:update('failed',error=str(error));raise


if __name__=='__main__':main()
