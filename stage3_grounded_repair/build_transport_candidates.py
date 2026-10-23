"""Causal intermediate-object relocation oracle and shared native-core repair.

All24 index35 development states retained. Only negative-MMU-evidence states
with two localized intermediate objects are changed. No final label gates the
intervention. Latent transport is an offline teacher; shared transport is an
untrained structural repair probe, not learned self-understanding feedback.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from runtime import Runtime,replay,tensor_hash,rng_restore
from object_transport import ObjectTransportRevisit,transport_field
from candidate_provenance import file_hash,freeze_config,verify_completed

ACTIONS=('baseline','conditional_zero','latent_transport','opposite_latent_transport','shared_transport','opposite_shared_transport')


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--states',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mask-manifest',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--num-shards',type=int,default=6)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files=sorted(args.states.rglob('*_i35.pt'))
    if len(files)!=24:raise ValueError('Expected exactly24 saved index35 states')
    manifest=json.loads(args.mask_manifest.read_text());records={r['state_id']:r for r in manifest['rows']}
    if len(records)!=24:raise ValueError('Incomplete masks manifest')
    source=Path(__file__).parent;log=args.output/f'actions_shard{args.shard}.jsonl'
    config={'actions':ACTIONS,'shared_window':4,'latent_window':1,'flow_index':35,'core':8,'strength':1.,
        'shard':args.shard,'num_shards':args.num_shards,'mask_manifest_sha256':file_hash(args.mask_manifest),
        'source_sha256':{n:file_hash(source/n) for n in ('runtime.py','object_transport.py','build_transport_candidates.py','candidate_provenance.py')},
        'captured_state_sha256':{str(q):file_hash(q) for q in files[args.shard::args.num_shards]},
        'gate':'existing MMU evidence <0 AND both objects present in intermediate detector output; no terminal outcome used',
        'optimizer_steps':0,'note':'External segmentation masks are OFFLINE oracle information; not final deployable understanding path'}
    fingerprint=freeze_config(args.output/f'config_shard{args.shard}.json',config,log.exists());done=set()
    if log.exists():
        for line in log.read_text().splitlines():
            row=json.loads(line);verify_completed(row,fingerprint);done.add((row['state_id'],row['action']))
    runtime=Runtime(args.work)
    with log.open('a',encoding='utf-8') as stream:
        for path in files[args.shard::args.num_shards]:
            state=torch.load(path,map_location='cpu',weights_only=False);record=records[state['state_id']]
            if state['split']!='architecture_dev' or state['state_fingerprint']!=record['state_fingerprint']:raise ValueError('Mask identity mismatch')
            if tensor_hash(state['latent'])!=state['state_fingerprint'] or not state['zero_intervention_test']['passed']:raise ValueError('Invalid state')
            if abs(state['evidence']['evidence']-record['original_evidence'])>.02:raise ValueError('Mask evidence mismatch')
            masks={}
            if record['eligible']:
                if state['evidence']['evidence']>=0:raise ValueError('Noncausal or changed gate')
                if file_hash(record['mask_path'])!=record['mask_sha256']:raise ValueError('Mask bytes changed')
                with np.load(record['mask_path'],allow_pickle=False) as archive:
                    masks={role:torch.from_numpy(archive[role].copy()).to(runtime.device) for role in ('subject','reference')}
            for action in ACTIONS:
                if (state['state_id'],action) in done:continue
                opposite=action.startswith('opposite');shared='shared_transport' in action
                repair,modifier=None,None;window=4 if shared or action=='conditional_zero' else 1
                if action=='conditional_zero' or (shared and record['eligible']):
                    repair=ObjectTransportRevisit(runtime.original[8],masks,record['boxes'],state['evidence']['relation'],runtime.h,runtime.w,
                        strength=0 if action=='conditional_zero' else 1.,opposite=opposite)
                if 'latent_transport' in action and record['eligible']:
                    interval=state['time_grid'][state['flow_index']+1]-state['time_grid'][state['flow_index']]
                    def modifier(rt,x,t,velocity,wrong=opposite,interval=interval):
                        clean=x[:1].float()+(1-t)*velocity[:1]
                        moved=transport_field(clean,masks,record['boxes'],state['evidence']['relation'],opposite=wrong)
                        # Shift ONLY estimated clean content; preserve the current
                        # noise residual. Euler applies t*(moved-clean) once.
                        delta=torch.cat((moved-clean,moved-clean))*t/interval.to(x.device)
                        return velocity+delta,0,0
                rng_restore(state['rng']);torch.cuda.synchronize();started=time.perf_counter()
                with torch.no_grad():final,cost=replay(runtime,state,repair=repair,window=window,velocity_modifier=modifier)
                image=runtime.decode(final);torch.cuda.synchronize();cost['wall_time_s']=time.perf_counter()-started
                should_be_exact=action in ('baseline','conditional_zero') or not record['eligible']
                if should_be_exact and not torch.equal(final,state['baseline_final_latent'].to(runtime.device)):raise RuntimeError('Zero/untriggered transport was not exact')
                directory=args.output/action/state['state_id'];(directory/'samples').mkdir(parents=True,exist_ok=True)
                image_path=directory/'samples/0000.png';image.save(image_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                final_path=directory/'final_latent.pt';torch.save({'latent':final.detach().cpu(),'state_id':state['state_id'],'action':action},final_path)
                row={key:state[key] for key in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t','state_fingerprint','zero_intervention_test','runtime')}
                row.update(action=action,configuration_sha256=fingerprint,image_path=str(image_path),final_latent=str(final_path),
                    image_path_sha256=file_hash(image_path),final_latent_sha256=file_hash(final_path),cost=cost,
                    intervention_eligible=record['eligible'],intervention_applied=record['eligible'] and action not in ('baseline','conditional_zero'),
                    gate_reason=record['reason'],mask_sha256=record['mask_sha256'],repair_window=window,
                    added_trainable_parameters=0,unique_parameters=state['runtime']['model_unique_parameters'],FLOPs=None,
                    action_note='offline detector-grounded structural probe; not trained repair or final MMU-only method',
                    cost_notes={'prefix_cache_reused':True,'evidence_precomputed':True,'external_detector_precomputed':True,
                        'detector_evaluations_if_deployed':1,'mmu_queries_if_deployed':2,
                        'compute_matching':'latent impulse and shared visits have different costs; report separately'})
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps({'state_id':state['state_id'],'action':action,'eligible':record['eligible'],'status':'ok'}),flush=True)


if __name__=='__main__':main()
