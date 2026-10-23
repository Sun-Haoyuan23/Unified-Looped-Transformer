"""Check the object relocation BEFORE any remaining flow can undo it.

18 intermediate images from all6 pre-existing eligible development states.
Diagnostic only; no final labels, optimization, semantic teacher admission or
claim of shared-loop success. Distinguish broken transport from flow reversion.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from runtime import Runtime
from object_transport import transport_field
from candidate_provenance import file_hash,freeze_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.work/'stage3_grounded_repair';out=args.output;out.mkdir(parents=True,exist_ok=True)
    status=out/'diagnostic_status.json'
    def update(phase,**kw):status.write_text(json.dumps({'phase':phase,'time':time.time(),**kw},indent=2))
    try:
        manifest=root/'object_masks_dev12x2_v6/mask_manifest.json'
        selected=[r for r in json.loads(manifest.read_text())['rows'] if r['eligible']]
        freeze_config(out/'config.json',{'source_sha256':{n:file_hash(root/n) for n in ('runtime.py','object_transport.py','diagnose_transport_geometry.py')},
            'mask_manifest_sha256':file_hash(manifest),'selection':'all pre-existing eligible intermediate states; no terminal label selection',
            'n_states':len(selected),'optimizer_steps':0})
        update('generating_diagnostic',states=len(selected));runtime=Runtime(args.work);rows=[]
        paths={q.stem:q for q in (root/'states/oracle_dev12x2_v1').rglob('*_i35.pt')}
        for record in selected:
            state=torch.load(paths[record['state_id']],map_location='cpu',weights_only=False)
            if state['state_fingerprint']!=record['state_fingerprint'] or file_hash(record['mask_path'])!=record['mask_sha256']:raise ValueError('State/mask identity changed')
            with np.load(record['mask_path'],allow_pickle=False) as archive:
                masks={role:torch.from_numpy(archive[role].copy()).to(runtime.device) for role in ('subject','reference')}
            clean=state['latent'][:1].to(runtime.device).float()+(1-state['true_t'])*state['baseline_prediction'][:1].to(runtime.device)
            for action in ('original_clean','target_moved_clean','opposite_moved_clean'):
                image_field=clean if action=='original_clean' else transport_field(clean,masks,record['boxes'],state['evidence']['relation'],opposite=action=='opposite_moved_clean')
                directory=out/'official_layout'/f'{len(rows):06d}';(directory/'samples').mkdir(parents=True)
                image_path=directory/'samples/0000.png';runtime.decode(image_field).save(image_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                rows.append({'state_id':state['state_id'],'prompt_id':state['prompt_id'],'prompt':state['prompt'],'action':action,
                    'image_path':str(image_path),'image_sha256':file_hash(image_path),'mask_sha256':record['mask_sha256']})
        (out/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        del runtime;gc.collect();torch.cuda.empty_cache()
        update('evaluating',images=len(rows))
        cmd=['/home/shy_23/calm_geneval_20260930/eval_venv/bin/python',str(args.work/'stage2b/geneval/evaluation/evaluate_images.py'),
            str(out/'official_layout'),'--outfile',str(out/'official_results.jsonl'),'--model-path','/home/shy_23/calm_geneval_20260930/detector',
            '--model-config','/home/shy_23/calm_geneval_20260930/eval_venv/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py']
        with (out/'detector.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
        if r.returncode:raise RuntimeError('Official diagnostic evaluator failed')
        official=[json.loads(line) for line in (out/'official_results.jsonl').read_text().splitlines()]
        lookup={str(Path(r['image_path']).resolve()):r for r in rows}
        if len(official)!=len(rows):raise ValueError('Missing diagnostic image scores')
        keyed=[]
        for result in official:
            record=lookup[str(Path(result['filename']).resolve())]
            if result['prompt']!=record['prompt']:raise ValueError('Wrong diagnostic prompt')
            keyed.append(dict(result,state_id=record['state_id'],action=record['action']))
        (out/'official_results_keyed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in keyed))
        report={'interpretation':'Intermediate geometry check, NOT terminal repair/quality or final shared-loop success',
            'states':len(selected),'prompts':len({r['prompt_id'] for r in selected}),
            'by_action':{a:{'n':len([r for r in keyed if r['action']==a]),'correct':sum(bool(r['correct']) for r in keyed if r['action']==a)} for a in ('original_clean','target_moved_clean','opposite_moved_clean')},
            'rows':keyed}
        (out/'geometry_report.json').write_text(json.dumps(report,indent=2))
        update('complete',images=len(rows),states=len(selected),prompts=report['prompts'],optimizer_steps=0)
    except Exception as error:update('failed',error=str(error));raise


if __name__=='__main__':main()
