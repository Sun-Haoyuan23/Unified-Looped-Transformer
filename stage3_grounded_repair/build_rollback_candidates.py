"""Offline causal late-check / early-repair plasticity probe, not final loop.

Check index35, rewind SAME trajectory to20, and use ONLY its intermediate
predicted-clean latent + detector masks. Check-derived clean-only control
separates moving objects from importing later visual information. All24 states
retained, original MMU<0 plus both-present gate, no final outcome consulted.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from runtime import Runtime,replay,tensor_hash,rng_restore
from rollback_evidence import checked_then_rewound
from object_transport import transport_field
from candidate_provenance import file_hash,freeze_config,verify_completed

ACTIONS=('baseline','checked_zero','checked_clean_only','checked_target_transport','checked_opposite_transport')


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--states',type=Path,required=True)
    p.add_argument('--checked-states',type=Path,required=True);p.add_argument('--mask-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--num-shards',type=int,default=6)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    early=sorted(args.states.rglob('*_i20.pt'));late={q.stem:q for q in args.checked_states.rglob('*_i35.pt')}
    if len(early)!=24 or len(late)!=24:raise ValueError('Expected24 matched early and late development states')
    masks_by_state={r['state_id']:r for r in json.loads(args.mask_manifest.read_text())['rows']}
    log=args.output/f'actions_shard{args.shard}.jsonl';source=Path(__file__).parent
    config={'actions':ACTIONS,'rewind_index':20,'check_index':35,'window':1,'strength':1.,
        'shard':args.shard,'num_shards':args.num_shards,'mask_manifest_sha256':file_hash(args.mask_manifest),
        'source_sha256':{n:file_hash(source/n) for n in ('runtime.py','rollback_evidence.py','object_transport.py','build_rollback_candidates.py','candidate_provenance.py')},
        'states_sha256':{str(q):file_hash(q) for q in early[args.shard::args.num_shards]},
        'checked_states_sha256':{str(q):file_hash(q) for q in late.values()},
        'optimizer_steps':0,'terminal_outcome_used_by_action':False,
        'note':'External masks + saved intermediate clean are offline oracle; extra discarded history is charged'}
    fingerprint=freeze_config(args.output/f'config_shard{args.shard}.json',config,log.exists());done=set()
    if log.exists():
        for line in log.read_text().splitlines():
            row=json.loads(line);verify_completed(row,fingerprint);done.add((row['state_id'],row['action']))
    runtime=Runtime(args.work)
    with log.open('a',encoding='utf-8') as stream:
        for path in early[args.shard::args.num_shards]:
            state=torch.load(path,map_location='cpu',weights_only=False)
            checked_id=state['state_id'].rsplit('_i',1)[0]+'_i35'
            checked=torch.load(late[checked_id],map_location='cpu',weights_only=False)
            rewound=checked_then_rewound(state,checked);record=masks_by_state[checked_id]
            if state['split']!='architecture_dev' or record['state_fingerprint']!=checked['state_fingerprint']:raise ValueError('Invalid checked mask identity')
            eligible=record['eligible']
            masks={}
            if eligible:
                if checked['evidence']['evidence']>=0 or file_hash(record['mask_path'])!=record['mask_sha256']:raise ValueError('Changed check gate or mask')
                with np.load(record['mask_path'],allow_pickle=False) as archive:
                    masks={role:torch.from_numpy(archive[role].copy()).to(runtime.device) for role in ('subject','reference')}
            checked_clean=checked['latent'][:1].to(runtime.device).float()+(1-checked['true_t'])*checked['baseline_prediction'][:1].to(runtime.device)
            for action in ACTIONS:
                if (state['state_id'],action) in done:continue
                desired=None
                if eligible and action=='checked_clean_only':desired=checked_clean
                elif eligible and action in ('checked_target_transport','checked_opposite_transport'):
                    desired=transport_field(checked_clean,masks,record['boxes'],checked['evidence']['relation'],opposite=action=='checked_opposite_transport')
                modifier=None
                if desired is not None:
                    interval=state['time_grid'][21]-state['time_grid'][20]
                    def modifier(rt,x,t,velocity,target=desired,interval=interval):
                        early_clean=x[:1].float()+(1-t)*velocity[:1]
                        correction=t*(target-early_clean)/interval.to(x.device)
                        return velocity+torch.cat((correction,correction)),0,0
                rng_restore(state['rng']);torch.cuda.synchronize();started=time.perf_counter()
                with torch.no_grad():final,cost=replay(runtime,state,window=1,velocity_modifier=modifier)
                image=runtime.decode(final);torch.cuda.synchronize();cost['wall_time_s']=time.perf_counter()-started
                if (desired is None) and not torch.equal(final,state['baseline_final_latent'].to(runtime.device)):raise RuntimeError('Zero/untriggered rewind continuation was not exact')
                cost.update(full_network_velocity_calls=49 if action=='baseline' else 65,
                            full_visual_block_calls=490 if action=='baseline' else 642)
                directory=args.output/action/state['state_id'];(directory/'samples').mkdir(parents=True,exist_ok=True)
                image_path=directory/'samples/0000.png';image.save(image_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                final_path=directory/'final_latent.pt';torch.save({'latent':final.detach().cpu(),'state_id':state['state_id'],'action':action},final_path)
                row={key:state[key] for key in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t','state_fingerprint','zero_intervention_test','runtime')}
                row.update(action=action,configuration_sha256=fingerprint,image_path=str(image_path),final_latent=str(final_path),
                    image_path_sha256=file_hash(image_path),final_latent_sha256=file_hash(final_path),cost=cost,
                    intervention_eligible=eligible,intervention_applied=desired is not None,
                    checked_state_id=checked_id,checked_state_fingerprint=checked['state_fingerprint'],
                    rollback=rewound['rollback'] if action!='baseline' else None,mask_sha256=record['mask_sha256'],
                    added_trainable_parameters=0,unique_parameters=state['runtime']['model_unique_parameters'],FLOPs=None,
                    action_note='offline late-check/early-repair teacher plasticity, not shared-loop success',
                    cost_notes={'prefix_cache_reused':True,'late_evidence_precomputed':True,'detector_precomputed':True,
                        'cost_scope':'wall_time is remaining continuation only; full call counts include late checking and discarded history',
                        'mmu_queries_if_deployed':0 if action=='baseline' else 2,'detector_evaluations_if_deployed':0 if action=='baseline' else 1})
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps({'state_id':state['state_id'],'action':action,'eligible':eligible,'status':'ok'}),flush=True)


if __name__=='__main__':main()
