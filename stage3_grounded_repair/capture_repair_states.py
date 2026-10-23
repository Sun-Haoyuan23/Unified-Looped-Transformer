"""Capture exact float32 fork states, actual grid, RNG, MMU and prefix cache."""
import argparse
import json
import time
from pathlib import Path
import torch
from runtime import Runtime,replay,tensor_hash,write_json,rng_snapshot,rng_restore
from evidence_encoder import EvidenceReader,condition_tensor
from evidence_conditioned_revisit import make_operator
from candidate_provenance import file_hash,freeze_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',default='30,35')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--max-tasks',type=int,default=0)
    p.add_argument('--split',default='architecture_dev')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).parent
    config={'tasks_sha256':file_hash(args.tasks),'indices':args.indices,
        'shard':args.shard,'num_shards':args.num_shards,'max_tasks':args.max_tasks,
        'default_split':args.split,'code_sha256':{name:file_hash(source/name)
            for name in ('capture_repair_states.py','runtime.py','evidence_encoder.py',
                         'evidence_conditioned_revisit.py','candidate_provenance.py')}}
    configuration_hash=freeze_config(args.output/f'capture_config_shard{args.shard}.json',
        config,bool(list(args.output.glob('*.pt'))))
    text=args.tasks.read_text(encoding='utf-8')
    tasks=json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.splitlines() if l.strip()]
    jobs=[(task,seed) for task in tasks for seed in task.get('seeds',[task.get('seed',101)])]
    jobs=jobs[args.shard::args.num_shards]
    if args.max_tasks: jobs=jobs[:args.max_tasks]
    r=Runtime(args.work)
    reader=EvidenceReader(r)
    operator=make_operator(r)
    indices={int(i) for i in args.indices.split(',')}
    if not indices.issubset(set(range(len(r.grid)-1))): raise ValueError('Invalid fork index')
    global_metadata=r.metadata()
    write_json(args.output/f'runtime_shard{args.shard}.json',global_metadata)
    logs=args.output/f'capture_shard{args.shard}.jsonl'
    with logs.open('a',encoding='utf-8') as stream:
        for task,seed in jobs:
            sid=task.get('prompt_id',task.get('id'))
            state_ids=[f'{sid}_seed{seed}_i{index:02d}' for index in sorted(indices)]
            if all((args.output/(s+'.pt')).exists() for s in state_ids):
                for state_id in state_ids:
                    stored=torch.load(args.output/(state_id+'.pt'),map_location='cpu',weights_only=False)
                    if (stored.get('configuration_sha256')!=configuration_hash or
                        stored['prompt']!=task['prompt'] or stored['seed']!=seed or
                        not stored['zero_intervention_test']['passed'] or
                        tensor_hash(stored['latent'])!=stored['state_fingerprint'] or
                        tensor_hash(stored['prefix_cache'])!=stored['prefix_cache_hash']):
                        raise RuntimeError('Invalid completed capture; use a new experiment directory')
                continue
            prefix=[]
            index=0
            captured={}
            def hook(_m,_a,result): prefix.append(result[0].detach().clone())
            def record(x,t,**kwargs):
                nonlocal index
                h=r.original[7].register_forward_hook(hook) if index in indices else None
                if h is not None:
                    prefix.clear(); latent=x.detach().clone(); rng=rng_snapshot()
                velocity=r.model.t2i_generate(x,t.clone(),**kwargs)
                if h is not None:
                    h.remove()
                    if len(prefix)!=1: raise RuntimeError('Ambiguous prefix cache')
                    captured[index]={'latent':latent.cpu(),'prefix_cache':prefix[0].cpu(),
                        'baseline_prediction':velocity.detach().float().cpu(),
                        'rng':rng,'true_t':float(t[0])}
                index+=1
                return velocity
            started=time.perf_counter()
            with torch.no_grad():
                final=r.sample_fn(r.noise(seed),record,**r.inputs(task['prompt']))[-1]
            baseline_image=r.decode(final)
            image_path=args.output/f'{sid}_seed{seed}_baseline.png'
            baseline_image.save(image_path)
            for flow_index,part in captured.items():
                state_id=f'{sid}_seed{seed}_i{flow_index:02d}'
                state={'state_id':state_id,'prompt_id':sid,'prompt':task['prompt'],
                    'configuration_sha256':configuration_hash,
                    'seed':seed,'flow_index':flow_index,'split':task.get('split',args.split),
                    'time_grid':r.grid.detach().cpu(),'prefix_cache_version':'after-original-block-7-v1',
                    'CFG':r.cfg,'solver_state':{'method':'euler','index':flow_index},
                    'baseline_final_latent':final.detach().cpu(),
                    'baseline_final_image':str(image_path),'runtime':global_metadata,
                    'geneval_metadata':task.get('geneval_metadata',task.get('metadata')),**part}
                state['prefix_cache_hash']=tensor_hash(state['prefix_cache'])
                clean=part['latent'][:1].to(r.device).float()+(1-part['true_t'])*part['baseline_prediction'][:1].to(r.device)
                evidence_started=time.perf_counter()
                with torch.no_grad(): state['evidence']=reader.read(clean,task['prompt'])
                torch.cuda.synchronize()
                state['evidence_runtime_s']=time.perf_counter()-evidence_started
                rng_restore(state['rng'])
                with torch.no_grad(): replayed,_=replay(r,state)
                operator.condition=condition_tensor(state['evidence'],state['true_t'],r.device)
                rng_restore(state['rng'])
                with torch.no_grad(): zero,_=replay(r,state,repair=operator)
                replay_image=r.decode(replayed); zero_image=r.decode(zero)
                import numpy as np
                diff=int(np.abs(np.asarray(baseline_image).astype('int16')-
                                np.asarray(replay_image).astype('int16')).max())
                zero_diff=int(np.abs(np.asarray(baseline_image).astype('int16')-
                                     np.asarray(zero_image).astype('int16')).max())
                exact=torch.equal(replayed,final) and torch.equal(zero,final)
                state['zero_intervention_test']={'passed':bool(exact and diff==0 and zero_diff==0),
                    'latent_exact':exact,'pixel_max_abs_diff':max(diff,zero_diff),
                    'baseline_replay_latent_max_abs_diff':float((replayed-final).abs().max()),
                    'adapter_zero_latent_max_abs_diff':float((zero-final).abs().max())}
                if not state['zero_intervention_test']['passed']:
                    write_json(args.output/f'{state_id}_ZERO_FAILED.json',state['zero_intervention_test'])
                    raise RuntimeError(f'Zero fork mismatch: {state["zero_intervention_test"]}')
                state['state_fingerprint']=tensor_hash(state['latent'])
                torch.save(state,args.output/f'{state_id}.pt')
                row={k:state[k] for k in ('state_id','prompt_id','seed','split','flow_index','true_t',
                                           'state_fingerprint','zero_intervention_test')}
                row['evidence']={k:v for k,v in state['evidence'].items() if 'hidden' not in k}
                row['wall_time_s']=time.perf_counter()-started
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps({'state_id':state_id,'status':'zero_exact',
                                   'true_t':state['true_t'],'evidence':state['evidence']['evidence']}),flush=True)


if __name__=='__main__': main()
