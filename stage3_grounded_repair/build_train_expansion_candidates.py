"""Explicit64 TRAIN teacher probe; actions use intermediate information only.

Same fixed original seed and index35. Minimum placement is an offline teacher.
Identical untriggered/zero candidates reuse a verified same-state baseline;
record aliases and measured cost explicitly, never count aliases as fresh draws.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from runtime import Runtime,replay,rng_restore,tensor_hash
from candidate_provenance import file_hash,freeze_config,verify_completed
from train_geometry_gate import geometry_gate
from fm_rebridge import continue_rebridged

ACTIONS=('baseline','checked_zero','checked_original_roundtrip','checked_seed_target',
         'checked_inferred_target','checked_seed_opposite','checked_inferred_opposite')


def main():
    p=argparse.ArgumentParser()
    for name in ('work','pilot','states','masks','teacher-diagnostic','identity-audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-states',type=int,required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--num-shards',type=int,required=True)
    args=p.parse_args();root=args.work/'stage3_grounded_repair';out=args.output;out.mkdir(parents=True)
    paths=sorted(args.states.glob('shard*/*_i35.pt'))
    tasks={q['prompt_id']:q for q in json.loads((args.pilot/'train_prompts.json').read_text())}
    masks={q['state_id']:q for q in json.loads((args.masks/'mask_manifest.json').read_text())['rows']}
    if (len(paths)!=args.expected_states or len({q.stem for q in paths})!=args.expected_states or
            set(masks)!={q.stem for q in paths} or any(q['split']!='train' for q in tasks.values()) or
            any(q['split']!='train' for q in masks.values()) or sum(len(q['seeds']) for q in tasks.values())!=args.expected_states):
        raise ValueError('Invalid explicit frozen TRAIN cohort')
    selected={sid for sid,q in masks.items() if q['eligible']}
    diagnostic=args.teacher_diagnostic
    teachers={q['state_id']:q for q in json.loads((diagnostic/'teacher_manifest.json').read_text())['rows']}
    if set(teachers)!=selected:raise ValueError('Teacher/mask cohort mismatch')
    gate=geometry_gate(json.loads((diagnostic/'geometry_report.json').read_text()),selected)
    audit=json.loads(args.identity_audit.read_text());identities={q['state_id']:q for q in audit['rows']}
    if (audit['terminal_outcomes_used'] is not False or audit['split']!='train' or
            len(identities)!=len(audit['rows']) or set(identities)!=selected):
        raise ValueError('Incomplete intermediate-only TRAIN identity audit')
    originals={q['state_id']:q for q in map(json.loads,(diagnostic/'records.jsonl').read_text().splitlines())
               if q['action']=='original_clean'}
    for sid,q in identities.items():
        if (q['state_fingerprint']!=masks[sid]['state_fingerprint'] or
                q['source_mask_sha256']!=masks[sid]['mask_sha256'] or
                q['original_clean_image_sha256']!=originals[sid]['image_sha256'] or
                type(q['independent_object_instances']) is not bool):raise ValueError('Changed identity provenance')
    frozen=freeze_config(out/f'config_shard{args.shard}.json',dict(
        actions=ACTIONS,expected_states=args.expected_states,shard=args.shard,num_shards=args.num_shards,split='train',
        check_index=35,repair_index=35,original_seed_fixed=True,terminal_outcomes_used_by_action=False,
        geometry='unchanged fixed minimum placement; no scan',
        teacher_manifest_sha256=file_hash(diagnostic/'teacher_manifest.json'),
        teacher_sha256={sid:q['teacher_sha256'] for sid,q in teachers.items()},
        geometry_report_sha256=file_hash(diagnostic/'geometry_report.json'),
        identity_audit_sha256=file_hash(args.identity_audit),mask_manifest_sha256=file_hash(args.masks/'mask_manifest.json'),
        pilot_prompts_sha256=file_hash(args.pilot/'train_prompts.json'),
        states_sha256={q.stem:file_hash(q) for q in paths[args.shard::args.num_shards]},
        source_sha256={n:file_hash(root/n) for n in ('build_train_expansion_candidates.py','runtime.py',
            'candidate_provenance.py','fm_rebridge.py','train_geometry_gate.py')},
        alias_policy='exact same-state zero/untriggered outputs reuse one verified baseline; no seed or checkpoint selection'))
    rt=Runtime(args.work);rows=[]
    with (out/f'actions_shard{args.shard}.jsonl').open('w') as log:
        for path in paths[args.shard::args.num_shards]:
            state=torch.load(path,map_location='cpu',weights_only=False);sid=state['state_id'];mask=masks[sid]
            if (state['split']!='train' or state['flow_index']!=35 or
                    state['prompt']!=tasks[state['prompt_id']]['prompt'] or state['seed'] not in tasks[state['prompt_id']]['seeds'] or
                    not state['zero_intervention_test']['passed'] or tensor_hash(state['latent'])!=state['state_fingerprint'] or
                    mask['state_fingerprint']!=state['state_fingerprint']):raise ValueError('Invalid TRAIN state')
            eligible=bool(mask['eligible'] and gate.get(sid,False) and identities.get(sid,{}).get('independent_object_instances',False))
            payload=None
            if eligible:
                teacher=teachers[sid]
                if state['evidence']['evidence']>=0 or file_hash(teacher['teacher_path'])!=teacher['teacher_sha256']:
                    raise ValueError('Changed intermediate teacher')
                payload=torch.load(teacher['teacher_path'],map_location='cpu',weights_only=False)
                if payload['state_fingerprint']!=state['state_fingerprint'] or payload['mask_sha256']!=mask['mask_sha256']:
                    raise ValueError('Teacher state/mask mismatch')
            baseline_final=None;baseline_image=None;baseline_cost=None
            for action in ACTIONS:
                intervention=eligible and action not in ('baseline','checked_zero')
                alias=action!='baseline' and not intervention
                rng_restore(state['rng']);torch.cuda.synchronize();start=time.perf_counter();bridge=None
                if alias:
                    final=baseline_final;image=baseline_image;cost=copy.deepcopy(baseline_cost)
                    cost['actual_new_velocity_calls']=0;cost['actual_new_decodes']=0;cost['actual_generation_wall_time_s']=0.
                else:
                    with torch.no_grad():
                        if intervention:
                            direction='original' if action=='checked_original_roundtrip' else 'opposite' if 'opposite' in action else 'target'
                            noise_source='seed' if 'seed_' in action else 'inferred'
                            final,cost,bridge=continue_rebridged(rt,state,payload['latents'][direction].to(rt.device),noise_source)
                        else:final,cost=replay(rt,state)
                    image=rt.decode(final);torch.cuda.synchronize()
                    cost.update(actual_new_velocity_calls=cost['network_velocity_calls'],actual_new_decodes=1,
                                actual_generation_wall_time_s=time.perf_counter()-start)
                    if action=='baseline':
                        if not torch.equal(final,state['baseline_final_latent'].to(rt.device)):raise ValueError('Zero baseline not exact')
                        baseline_final=final;baseline_image=image;baseline_cost=copy.deepcopy(cost)
                cost['wall_time_s']=None if alias else cost['actual_generation_wall_time_s']
                cost['nominal_deployed_full_velocity_calls']=50 if intervention else 49
                directory=out/action/sid;(directory/'samples').mkdir(parents=True)
                image_path=directory/'samples/0000.png';image.save(image_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                final_path=directory/'final_latent.pt';torch.save(dict(latent=final.cpu(),state_id=sid,action=action),final_path)
                row={k:state[k] for k in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t',
                    'state_fingerprint','zero_intervention_test','runtime')}
                row.update(action=action,configuration_sha256=frozen,image_path=str(image_path),final_latent=str(final_path),
                    image_path_sha256=file_hash(image_path),final_latent_sha256=file_hash(final_path),
                    original_mask_eligible=mask['eligible'],intermediate_geometry_valid=gate.get(sid,False),
                    intermediate_identity_valid=identities.get(sid,{}).get('independent_object_instances',False),
                    intervention_eligible=eligible,intervention_applied=intervention,
                    teacher_sha256=teachers[sid]['teacher_sha256'] if eligible else None,bridge=bridge,cost=cost,
                    baseline_alias=alias,alias_source_action='baseline' if alias else None,
                    added_trainable_parameters=0,unique_parameters=state['runtime']['model_unique_parameters'],FLOPs=None,
                    cost_notes=dict(scope='offline remaining flow and decode; aliases cost zero only for dataset collection',
                        deployed_MMUs=0 if action=='baseline' else 2,external_detector_VAE_composition_additional=True,
                        new_first_full_velocity=intervention,prefix_reused=not intervention),
                    action_note='Offline minimum geometry teacher; not learned shared recurrence')
                verify_completed(row,frozen);log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
                print(json.dumps(dict(state_id=sid,action=action,applied=intervention,alias=alias)),flush=True)


if __name__=='__main__':main()
