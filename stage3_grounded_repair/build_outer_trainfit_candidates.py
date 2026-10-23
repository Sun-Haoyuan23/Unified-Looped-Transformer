"""TRAIN capacity diagnostic for outer shared recurrence; no external teachers in inference."""
import argparse
import json
import time
from pathlib import Path
import torch
from runtime import Runtime,rng_restore,tensor_hash,write_json
from evidence_encoder import condition_tensor
from outer_shared_revisit import make_operator,install,replay_outer
from candidate_provenance import file_hash,freeze_config,shuffled_source
from training_pairs import selected_pairs


def main():
    p=argparse.ArgumentParser()
    for name in ('work','states','evidence-states','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    root=args.work/'stage3_grounded_repair'
    args.output.mkdir()
    teacher_root=root/'oracle_encoded_correlation_train24x2_v23'
    pairs=selected_pairs(json.loads((teacher_root/'paired_report.json').read_text()),
                         list(map(json.loads,(teacher_root/'all_actions.jsonl').read_text().splitlines())))
    paths=sorted(args.states.glob('*_i35.pt'))
    if len(paths)!=2:raise ValueError('Require two pre-existing fixed TRAIN diagnostic states')
    all_evidence={q.stem:q for q in args.evidence_states.glob('shard*/*_i35.pt')}
    if len(all_evidence)!=48:raise ValueError('Frozen evidence cohort changed')
    metadata={}
    for sid,path in all_evidence.items():
        item=torch.load(path,map_location='cpu',weights_only=False)
        if item['split']!='train' or item['flow_index']!=35:raise ValueError('Evidence split/time leak')
        metadata[sid]={k:item[k] for k in ('split','flow_index','prompt_id')}
    runtime=Runtime(args.work)
    op=make_operator(runtime)
    zero=make_operator(runtime)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if (payload['step']!=50 or payload['level']!=2 or payload['width']!=128 or
            payload['metadata']['operator_kind']!='outer_tokenwise'):
        raise ValueError('Fixed outer architecture/checkpoint mismatch')
    expected={k for k in op.state_dict() if not k.startswith('shared_core.')}
    if set(payload['operator'])!=expected:raise ValueError('Incomplete outer checkpoint')
    op.load_state_dict(payload['operator'],strict=False)
    actions=('baseline','conditional_zero','conditional_revisit','shuffled_evidence','opposite_evidence')
    frozen=freeze_config(args.output/'config_shard0.json',dict(
        actions=actions,split='train',window=4,checkpoint_sha256=file_hash(args.checkpoint),
        teacher_report_sha256=file_hash(teacher_root/'paired_report.json'),
        state_sha256={q.stem:file_hash(q) for q in paths},
        evidence_state_sha256={k:file_hash(v) for k,v in all_evidence.items()},
        source_sha256={n:file_hash(root/n) for n in ('build_outer_trainfit_candidates.py','outer_shared_revisit.py',
            'runtime.py','candidate_provenance.py','evidence_encoder.py','training_pairs.py')},
        checkpoint_selection='fixed final step50; TRAIN fit only, no val or controller'))
    with (args.output/'actions_shard0.jsonl').open('w') as log:
        for path in paths:
            state=torch.load(path,map_location='cpu',weights_only=False)
            sid=state['state_id']
            if (state['split']!='train' or state['flow_index']!=35 or not state['zero_intervention_test']['passed'] or
                    tensor_hash(state['latent'])!=state['state_fingerprint']):raise ValueError('Invalid TRAIN input')
            baseline=state['baseline_final_latent'].to(runtime.device).float()
            target=baseline
            if sid in pairs:
                teacher=pairs[sid]
                if file_hash(teacher['final_latent'])!=teacher['final_latent_sha256']:raise ValueError('Changed teacher')
                saved=torch.load(teacher['final_latent'],map_location=runtime.device,weights_only=False)
                if saved['state_id']!=sid or saved['action']!=teacher['action']:raise ValueError('Wrong teacher state')
                target=saved['latent'].float()
            for action in actions:
                current=zero if action in ('baseline','conditional_zero') else op
                evidence=state['evidence'];source_id=None
                if action=='shuffled_evidence':
                    source_id=shuffled_source(metadata,sid)
                    evidence=torch.load(all_evidence[source_id],map_location='cpu',weights_only=False)['evidence']
                elif action=='opposite_evidence':
                    evidence=dict(evidence)
                    for a,b in (('target_margin','opposite_margin'),('target_hidden','opposite_hidden')):
                        evidence[a],evidence[b]=evidence[b],evidence[a]
                    evidence['evidence']=-evidence['evidence']
                current.condition=condition_tensor(evidence,state['true_t'],runtime.device)
                rng_restore(state['rng']);torch.cuda.synchronize();started=time.perf_counter()
                with torch.no_grad(),install(runtime,current):
                    final,cost=replay_outer(runtime,state,current,window=0 if action=='baseline' else 4)
                if action in ('baseline','conditional_zero') and not torch.equal(final,baseline):
                    raise ValueError('Outer baseline/explicit untrained zero not strictly identical')
                image=runtime.decode(final);torch.cuda.synchronize()
                cost['wall_time_s']=time.perf_counter()-started
                cost['semantic_encoder_layer_calls_nominal']=14*len(runtime.model.und_trans.layers)
                directory=args.output/action/sid;(directory/'samples').mkdir(parents=True)
                image_path=directory/'samples/0000.png';image.save(image_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                final_path=directory/'final_latent.pt';torch.save(dict(latent=final.cpu(),state_id=sid,action=action),final_path)
                scale=baseline.square().mean().clamp_min(1e-5)
                row={k:state[k] for k in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t',
                                        'state_fingerprint','zero_intervention_test','runtime')}
                row.update(action=action,configuration_sha256=frozen,image_path=str(image_path),final_latent=str(final_path),
                    image_path_sha256=file_hash(image_path),final_latent_sha256=file_hash(final_path),
                    cost=cost,cost_notes=dict(scope='remaining14 full velocities and decode, not end-to-end',
                        old_inner_prefix_reused=False,old_KV_reused=False,MMU_checks_if_deployed=0 if action=='baseline' else 2,
                        first_checked_velocity_recomputed=True,FLOPs_measured=False),
                    trainfit_teacher_distance=float((final-target).square().mean()/scale),
                    trainfit_baseline_teacher_distance=float((baseline-target).square().mean()/scale),
                    shuffled_source_state=source_id,checkpoint=str(args.checkpoint),
                    added_trainable_parameters=0 if action=='baseline' else sum(q.numel() for q in current.trainable_parameters()),
                    FLOPs=None)
                log.write(json.dumps(row)+'\n');log.flush()
    write_json(args.output/'generation_status.json',dict(success=True,expected_images=10,optimizer_steps=0))


if __name__=='__main__':main()
