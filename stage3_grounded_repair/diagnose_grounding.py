"""Development-only grounding diagnostics; no optimizer or controller.

Reflect predicted-clean latent along the requested relation axis, query object
presence and four half-image positions with the frozen MMU path, and export
clean images for the official detector. Reflection is of the latent, not an
exact pixel reflection. No final outcome is used to construct these inputs.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from runtime import Runtime, tensor_hash
from evidence_encoder import EvidenceReader
from candidate_provenance import file_hash, freeze_config


def reflection_axis(relation):
    if relation in ('left of', 'right of'): return -1
    if relation in ('above', 'below'): return -2
    raise ValueError('Unsupported relation')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--states',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    args=p.parse_args()
    paths=sorted(q for root in args.states for q in root.rglob('*.pt'))
    if not paths: raise ValueError('No states')
    args.output.mkdir(parents=True,exist_ok=True)
    log=args.output/f'grounding_shard{args.shard}.jsonl'
    config={'states':[str(q.resolve()) for q in args.states], 'shard':args.shard,
            'num_shards':args.num_shards,'source_sha256':{name:file_hash(Path(__file__).parent/name)
                for name in ('diagnose_grounding.py','runtime.py','evidence_encoder.py')},
            'protocol':'all saved architecture-development states; latent-axis reflection; no final-outcome conditioning',
            'optimizer_steps':0}
    fingerprint=freeze_config(args.output/f'config_shard{args.shard}.json',config,log.exists())
    done={}
    if log.exists():
        for line in log.read_text().splitlines():
            row=json.loads(line)
            if row['configuration_sha256']!=fingerprint:raise ValueError('Changed diagnostic configuration')
            done[row['state_id']]=row
    runtime=Runtime(args.work)
    reader=EvidenceReader(runtime)
    with log.open('a',encoding='utf-8') as stream:
        for path in paths[args.shard::args.num_shards]:
            source_hash=file_hash(path)
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['split']!='architecture_dev':raise ValueError('Diagnostic restricted to architecture development')
            if state['state_id'] in done:
                if done[state['state_id']]['state_file_sha256']!=source_hash:raise ValueError('Changed input state')
                continue
            if tensor_hash(state['latent'])!=state['state_fingerprint']:raise ValueError('Corrupt state')
            if tensor_hash(state['prefix_cache'])!=state['prefix_cache_hash']:raise ValueError('Corrupt prefix')
            if not state['zero_intervention_test']['passed']:raise ValueError('Unmatched state')
            started=time.perf_counter()
            clean=state['latent'][:1].to(runtime.device).float()+(1-state['true_t'])*state['baseline_prediction'][:1].to(runtime.device)
            with torch.no_grad():
                original=reader.read(clean,state['prompt'])
                reflected=reader.read(clean.flip(reflection_axis(original['relation'])),state['prompt'])
                visual=reader.base._visual_features(clean)
                object_scores={}
                for role in ('subject','reference'):
                    name=original[role]
                    scores={'present':reader._question(visual,f'Is there a {name} in this image? Answer only Yes or No.')[0]}
                    for side in ('left','right','top','bottom'):
                        scores[side]=reader._question(visual,f'Is the {name} in the {side} half of this image? Answer only Yes or No.')[0]
                    scores['left_minus_right']=scores['left']-scores['right']
                    scores['top_minus_bottom']=scores['top']-scores['bottom']
                    object_scores[role]={'object':name,**scores}
            difference=abs(original['evidence']-state['evidence']['evidence'])
            if difference>.02:raise RuntimeError(f'MMU evidence replay mismatch {difference}')
            directory=args.output/'clean_images'/state['state_id']
            (directory/'samples').mkdir(parents=True,exist_ok=True)
            image_path=directory/'samples/0000.png'
            runtime.decode(clean).save(image_path)
            metadata=state['geneval_metadata']
            if metadata['prompt']!=state['prompt']:raise ValueError('Metadata prompt differs')
            (directory/'metadata.jsonl').write_text(json.dumps(metadata)+'\n')
            row={key:state[key] for key in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t','state_fingerprint')}
            row.update(configuration_sha256=fingerprint,state_file_sha256=source_hash,
                original_evidence=original['evidence'], reflected_evidence=reflected['evidence'],
                target_margin=original['target_margin'],opposite_margin=original['opposite_margin'],
                reflected_target_margin=reflected['target_margin'],reflected_opposite_margin=reflected['opposite_margin'],
                reflection_axis=reflection_axis(original['relation']),objects=object_scores,
                clean_image=str(image_path),clean_image_sha256=file_hash(image_path),
                evidence_replay_abs_diff=difference,mmu_queries=14,
                wall_time_s=time.perf_counter()-started,
                interpretation='Development grounding diagnosis, not a repair success or independent understanding benchmark')
            stream.write(json.dumps(row)+'\n');stream.flush()
            print(json.dumps({'state_id':state['state_id'],'original':row['original_evidence'],
                              'reflected':row['reflected_evidence']}),flush=True)


if __name__=='__main__':main()
