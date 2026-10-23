"""Fork exact stored states; actions use one canonical continuation implementation."""
import argparse
import json
import time
from pathlib import Path
import torch
from runtime import Runtime,replay,tensor_hash,rng_restore,write_json
from evidence_encoder import condition_tensor
from evidence_conditioned_revisit import make_operator
from candidate_provenance import file_hash,freeze_config,shuffled_source,verify_completed


def prompt_repair(prompt,evidence,opposite=False):
    from mmu_evidence_reader import OPPOSITE
    relation=evidence['relation']
    if opposite: relation=OPPOSITE[relation]
    return (prompt+'. The '+evidence['subject']+' must be clearly '+relation+' the '+
            evidence['reference']+'. Keep both objects visible and clearly separated.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--states',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--actions',default='baseline,generic_revisit,extra_ode,conditional_zero,semantic_probe,opposite_probe,prompt_repair,opposite_prompt')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--level',type=int,default=1)
    p.add_argument('--window',type=int,default=4)
    p.add_argument('--gain',type=float,default=.1)
    args=p.parse_args()
    r=Runtime(args.work)
    op=make_operator(r,level=args.level,checkpoint=args.checkpoint)
    actions=args.actions.split(',')
    allowed={'baseline','generic_revisit','extra_ode','conditional_zero','conditional_revisit',
             'shuffled_evidence','opposite_evidence','semantic_probe','opposite_probe',
             'prompt_repair','opposite_prompt','layout_plan','opposite_layout',
             'regional_flow','opposite_regional','spatial_probe','opposite_spatial'}
    if not set(actions)<=allowed: raise ValueError('Unknown action')
    if args.checkpoint is None and any(a in actions for a in ['conditional_revisit','shuffled_evidence','opposite_evidence']):
        raise ValueError('Learned evidence ablations require a trained repair checkpoint')
    files=sorted(args.states.glob('*.pt'))
    if not files: raise RuntimeError('No captured states')
    args.output.mkdir(parents=True,exist_ok=True)
    log=args.output/f'actions_shard{args.shard}.jsonl'
    source=Path(__file__).parent
    config={'actions':actions,'window':args.window,'gain':args.gain,'level':args.level,
        'checkpoint_sha256':file_hash(args.checkpoint) if args.checkpoint else None,
        'states':str(args.states.resolve()),'shard':args.shard,'num_shards':args.num_shards,
        'code_sha256':{name:file_hash(source/name) for name in ('runtime.py',
            'evidence_encoder.py','evidence_conditioned_revisit.py',
            'build_repair_candidates.py','candidate_provenance.py')}}
    if any(a in actions for a in ('layout_plan','opposite_layout','regional_flow',
                                 'opposite_regional','spatial_probe','opposite_spatial')):
        config['code_sha256']['grounded_actions.py']=file_hash(source/'grounded_actions.py')
    fingerprint=freeze_config(args.output/f'config_shard{args.shard}.json',config,log.exists())
    done=set()
    done_states={}
    if log.exists():
        for line in log.read_text().splitlines():
            row=json.loads(line);verify_completed(row,fingerprint)
            done.add((row['state_id'],row['action']))
            previous=done_states.setdefault(row['state_id'],row['state_fingerprint'])
            if previous!=row['state_fingerprint']:raise RuntimeError('Mixed saved state fingerprints')
    state_metadata={}
    state_paths={}
    if 'shuffled_evidence' in actions:
        for path in files:
            item=torch.load(path,map_location='cpu',weights_only=False)
            state_metadata[item['state_id']]={key:item[key] for key in ('split','flow_index','prompt_id')}
            state_paths[item['state_id']]=path
    with log.open('a',encoding='utf-8') as stream:
        for offset,state_path in enumerate(files):
            if offset%args.num_shards!=args.shard: continue
            state=torch.load(state_path,map_location='cpu',weights_only=False)
            zero=state['zero_intervention_test']
            if not zero['passed']: raise RuntimeError('Reject state without exact zero check')
            if tensor_hash(state['latent'])!=state['state_fingerprint']: raise RuntimeError('State corrupted')
            if state['state_id'] in done_states and done_states[state['state_id']]!=state['state_fingerprint']:
                raise RuntimeError('Captured state changed since candidate generation')
            for action in actions:
                if (state['state_id'],action) in done: continue
                op.mode='learned'
                op.generic_gain=args.gain
                op.condition=condition_tensor(state['evidence'],state['true_t'],r.device)
                repair,subdivisions,override,modifier=None,1,None,None
                if action=='generic_revisit': op.mode='generic';repair=op
                elif action=='extra_ode': subdivisions=2
                elif action in ('semantic_probe','opposite_probe','spatial_probe','opposite_spatial'):
                    op.mode='probe';repair=op
                    delta=state['evidence']['target_hidden'].to(r.device).float()-state['evidence']['opposite_hidden'].to(r.device).float()
                    with torch.no_grad():
                        projection=r.model.diff_proj(delta.to(r.dtype)) if hasattr(r.model,'diff_proj') else delta
                    projection=projection.float()
                    op.probe_beta=projection/(projection.square().mean().sqrt()+1e-6)
                    if action=='opposite_probe': op.probe_beta=-op.probe_beta
                    if action in ('spatial_probe','opposite_spatial'):
                        from grounded_actions import spatial_field
                        support,_=spatial_field(state['evidence']['relation'],r.h,r.w,r.device)
                        field=(2*support-1).reshape(-1)
                        field=field/(field.square().mean().sqrt()+1e-6)
                        if action=='opposite_spatial':field=-field
                        tokens=torch.zeros((state['prefix_cache'].shape[1],projection.numel()),device=r.device)
                        positions=r.inputs(state['prompt'])['modality_positions']
                        start,length=positions[0,0].tolist()
                        if length-1!=field.numel():raise RuntimeError('Spatial token grid differs from model positions')
                        tokens[start+1:start+length]=field[:,None]*op.probe_beta[None,:]
                        op.probe_beta=tokens
                elif action in ('conditional_zero','conditional_revisit','shuffled_evidence','opposite_evidence'):
                    repair=op
                    if action=='conditional_zero' and args.checkpoint is not None:
                        raise ValueError('Zero ablation uses an explicitly untrained adapter')
                    if action=='opposite_evidence':
                        swapped=dict(state['evidence'])
                        for a,b in [('target_margin','opposite_margin'),('target_hidden','opposite_hidden')]:
                            swapped[a],swapped[b]=swapped[b],swapped[a]
                        swapped['evidence']=-swapped['evidence']
                        op.condition=condition_tensor(swapped,state['true_t'],r.device)
                    elif action=='shuffled_evidence':
                        source_id=shuffled_source(state_metadata,state['state_id'])
                        other=torch.load(state_paths[source_id],map_location='cpu',weights_only=False)
                        op.condition=condition_tensor(other['evidence'],state['true_t'],r.device)
                elif action in ('prompt_repair','opposite_prompt'):
                    override=prompt_repair(state['prompt'],state['evidence'],action=='opposite_prompt')
                elif action in ('layout_plan','opposite_layout'):
                    from grounded_actions import concise_layout_prompt
                    override=concise_layout_prompt(state['evidence'],action=='opposite_layout')
                elif action in ('regional_flow','opposite_regional'):
                    from grounded_actions import regional_velocity
                    def modifier(rt,x,t,velocity,ev=state['evidence'],wrong=action=='opposite_regional'):
                        return regional_velocity(rt,x,t,velocity,ev,opposite=wrong)
                rng_restore(state['rng'])
                torch.cuda.synchronize();started=time.perf_counter()
                with torch.no_grad():
                    final,cost=replay(r,state,repair=repair,extra_subdivisions=subdivisions,
                                      prompt_override=override,window=args.window,velocity_modifier=modifier)
                image=r.decode(final)
                torch.cuda.synchronize();cost['wall_time_s']=time.perf_counter()-started
                if action in ('baseline','conditional_zero') and not torch.equal(final,state['baseline_final_latent'].to(r.device)):
                    raise RuntimeError('Candidate zero intervention is not exact')
                sample_dir=args.output/action/state['state_id']
                sample_dir.mkdir(parents=True,exist_ok=True)
                image_path=sample_dir/'samples/0000.png';image_path.parent.mkdir(exist_ok=True)
                image.save(image_path)
                if state.get('geneval_metadata') is not None:
                    (sample_dir/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                final_path=sample_dir/'final_latent.pt'
                torch.save({'latent':final.detach().cpu(),'action':action,'state_id':state['state_id']},final_path)
                row={k:state[k] for k in ('state_id','prompt_id','prompt','seed','split','flow_index',
                    'true_t','state_fingerprint','zero_intervention_test')}
                row.update({'action':action,'image_path':str(image_path),'final_latent':str(final_path),
                    'configuration_sha256':fingerprint,
                    'image_path_sha256':file_hash(image_path),'final_latent_sha256':file_hash(final_path),
                    'shuffled_source_state':source_id if action=='shuffled_evidence' else None,
                    'cost_notes':{'prefix_cache_reused':override is None,
                         'compute_matching':'not established; report full measured costs',
                         'evidence_precomputed':True,'evidence_queries_if_deployed':2 if
                             action in ('semantic_probe','opposite_probe','conditional_revisit','shuffled_evidence','opposite_evidence','prompt_repair','opposite_prompt') else 0},
                    'cost':cost,'repair_window':args.window,'condition_type':op.encoder.level,
                    'action_note':'offline teacher interface, not final loop method' if override or modifier else
                                  'untrained deterministic MMU-semantic probe' if 'probe' in action else None,
                    'checkpoint':str(args.checkpoint) if args.checkpoint else None,
                    'unique_parameters':sum(p.numel() for p in r.model.parameters())+
                                        (sum(p.numel() for p in op.trainable_parameters()) if
                                         action in ('conditional_zero','conditional_revisit','shuffled_evidence','opposite_evidence') else 0),
                    'added_trainable_parameters':sum(p.numel() for p in op.trainable_parameters()) if
                        action in ('conditional_zero','conditional_revisit','shuffled_evidence','opposite_evidence') else 0,
                    'FLOPs':None,'runtime':state['runtime']})
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps({'state_id':state['state_id'],'action':action,'status':'ok',
                                  'wall_time_s':round(cost['wall_time_s'],3)}),flush=True)


if __name__=='__main__': main()
