"""Predeclared routing vs projection-interface probe; 24 dev states at index20.

Original core2 and natural-input core8 are a single architectural contrast,
not a block-position sweep. Four-step window, same prompt/seed, no new weights.
This interface probes repairability and is not evidence of learned self-repair.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from runtime import Runtime,replay,tensor_hash,rng_restore
from routed_shared_revisit import RoutedSharedRevisit,find_prompt_token_spans
from candidate_provenance import file_hash,freeze_config,verify_completed

ACTIONS=('baseline','conditional_zero','routed_core2','opposite_routed_core2','routed_core8','opposite_routed_core8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--num-shards',type=int,default=7)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in args.states.rglob('*_i20.pt'))
    if len(files)!=24:raise ValueError('Expected exactly 24 index20 states (12 dev prompts x2seed)')
    log=args.output/f'actions_shard{args.shard}.jsonl';source=Path(__file__).parent
    config={'actions':ACTIONS,'window':4,'flow_index':20,'routing_strength':1.0,'prior_floor':.01,
        'cores':[2,8],'input_after_block':7,'states':str(args.states.resolve()),
        'shard':args.shard,'num_shards':args.num_shards,'optimizer_steps':0,
        'source_sha256':{name:file_hash(source/name) for name in ('runtime.py','routed_shared_revisit.py','build_routed_candidates.py','grounded_actions.py','candidate_provenance.py')},
        'captured_state_sha256':{str(p):file_hash(p) for p in files[args.shard::args.num_shards]}}
    fingerprint=freeze_config(args.output/f'config_shard{args.shard}.json',config,log.exists())
    done=set()
    if log.exists():
        for line in log.read_text().splitlines():
            row=json.loads(line);verify_completed(row,fingerprint);done.add((row['state_id'],row['action']))
    runtime=Runtime(args.work)
    with log.open('a',encoding='utf-8') as stream:
        for path in files[args.shard::args.num_shards]:
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['split']!='architecture_dev' or not state['zero_intervention_test']['passed']:raise ValueError('Invalid dev capture')
            if tensor_hash(state['latent'])!=state['state_fingerprint']:raise ValueError('Corrupt latent')
            kwargs=runtime.inputs(state['prompt']);offset=int(kwargs['modality_positions'][0,0,0])
            spans=find_prompt_token_spans(runtime.tokenizer,state['prompt'],state['evidence'],kwargs['text_tokens'],offset)
            for action in ACTIONS:
                if (state['state_id'],action) in done:continue
                core=2 if action.endswith('core2') else 8
                repair=None if action=='baseline' else RoutedSharedRevisit(runtime.original[core],spans,state['evidence']['relation'],
                    runtime.h,runtime.w,strength=0 if action=='conditional_zero' else 1.0,opposite=action.startswith('opposite'))
                rng_restore(state['rng']);torch.cuda.synchronize();started=time.perf_counter()
                with torch.no_grad():final,cost=replay(runtime,state,repair=repair,window=4)
                image=runtime.decode(final);torch.cuda.synchronize();cost['wall_time_s']=time.perf_counter()-started
                if action in ('baseline','conditional_zero') and not torch.equal(final,state['baseline_final_latent'].to(runtime.device)):
                    raise RuntimeError('Zero routing failed exact replay')
                if repair and action!='conditional_zero' and repair.changed_mask_elements==0:raise RuntimeError('Routing did not alter any attention element')
                directory=args.output/action/state['state_id'];(directory/'samples').mkdir(parents=True,exist_ok=True)
                image_path=directory/'samples/0000.png';image.save(image_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                final_path=directory/'final_latent.pt';torch.save({'latent':final.detach().cpu(),'action':action,'state_id':state['state_id']},final_path)
                row={key:state[key] for key in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t','state_fingerprint','zero_intervention_test','runtime')}
                row.update(action=action,image_path=str(image_path),final_latent=str(final_path),
                    configuration_sha256=fingerprint,image_path_sha256=file_hash(image_path),final_latent_sha256=file_hash(final_path),
                    object_token_spans=spans,changed_mask_elements=repair.changed_mask_elements if repair else 0,
                    core_index=core if repair else None,cost=cost,repair_window=4,condition_type='object text-key routing',
                    unique_parameters=state['runtime']['model_unique_parameters'],added_trainable_parameters=0,FLOPs=None,
                    action_note='Deterministic structural binding oracle; not trained MMU-conditioned repair or controller',
                    cost_notes={'prefix_cache_reused':True,'evidence_precomputed':True,'evidence_queries_if_deployed':2,'compute_matching':'core2/core8/opposite controls have identical counts; baseline has fewer calls'})
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps({'state_id':state['state_id'],'action':action,'status':'ok','mask_changes':row['changed_mask_elements']}),flush=True)


if __name__=='__main__':main()
