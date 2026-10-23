"""External detector masks for OFFLINE repairability oracle, never terminal labels."""
import argparse
import json
from pathlib import Path
import numpy as np
# Match official GenEval import order. Importing mmcv/cv2 first selects the
# system C++ runtime, which is too old for this server's inherited Pillow.
from PIL import Image
import torch
from mmdet.apis import init_detector,inference_detector
from candidate_provenance import file_hash,freeze_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.work/'stage3_grounded_repair';args.output.mkdir(parents=True,exist_ok=True)
    source=root/'grounding_dev12x2_v4/all_grounding.jsonl'
    names=(args.work/'stage2b/geneval/evaluation/object_names.txt').read_text().splitlines()
    cfg='/home/shy_23/calm_geneval_20260930/eval_venv/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    weights='/home/shy_23/calm_geneval_20260930/detector/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth'
    freeze_config(args.output/'config.json',{'source_sha256':file_hash(__file__),'grounding_rows_sha256':file_hash(source),
        'detector_checkpoint_sha256':file_hash(weights),'flow_index':35,'threshold':.3,'error_gate':'original MMU evidence < 0',
        'terminal_outcome_used':False})
    detector=init_detector(cfg,weights,device='cuda:0');output=[]
    rows=sorted([json.loads(line) for line in source.read_text().splitlines() if json.loads(line)['flow_index']==35],key=lambda r:r['state_id'])
    if len(rows)!=24:raise ValueError('Expected 24 late check states')
    for row in rows:
        record={'state_id':row['state_id'],'state_fingerprint':row['state_fingerprint'],'prompt_id':row['prompt_id'],
            'clean_image_sha256':row['clean_image_sha256'],'original_evidence':row['original_evidence'],
            'split':row['split'],'eligible':False,'mask_path':None,'mask_sha256':None,'boxes':{}}
        if file_hash(row['clean_image'])!=row['clean_image_sha256']:raise ValueError('Changed partial image')
        if row['original_evidence']>=0:
            record['reason']='MMU did not flag error';output.append(record);continue
        result=inference_detector(detector,row['clean_image'])
        if not isinstance(result,tuple) or len(result)<2:raise ValueError('Detector has no instance masks')
        boxes,segm=result;arrays={}
        for role in ('subject','reference'):
            name=row['objects'][role]['object'];index=names.index(name)
            if len(boxes[index])==0:break
            best=int(np.argmax(boxes[index][:,4]));box=boxes[index][best]
            if float(box[4])<=.3:break
            mask=np.asarray(segm[index][best],dtype=np.float32)
            if mask.shape!=(432,432) or not mask.any():raise ValueError('Invalid instance mask')
            arrays[role]=mask;record['boxes'][role]=box.tolist()
        if len(arrays)==2:
            path=args.output/(row['state_id']+'.npz');np.savez_compressed(path,**arrays)
            record.update(eligible=True,mask_path=str(path.resolve()),mask_sha256=file_hash(path),reason='Both objects localized in intermediate clean prediction')
        else:record['reason']='One or both objects not detected; preserve original continuation'
        output.append(record)
    (args.output/'mask_manifest.json').write_text(json.dumps({'rows':output,'n_states':24,'n_eligible':sum(r['eligible'] for r in output),
        'source':'external offline detector on predicted-clean intermediate image; no final outcome used'},indent=2))
    print(json.dumps({'states':24,'eligible':sum(r['eligible'] for r in output)}),flush=True)


if __name__=='__main__':main()
