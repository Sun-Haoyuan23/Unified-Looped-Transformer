"""Late-check causal teacher probe: local inferred residual vs true seed noise.

Index35 in both arms, TRAIN only, minimum-change intermediate geometry and identity gate, no endpoint
labels. No early rewind, no strength/time sweep. Pixel-to-VAE remains offline.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from runtime import Runtime, replay, rng_restore
from candidate_provenance import file_hash, freeze_config, verify_completed
from train_geometry_gate import geometry_gate
from fm_rebridge import continue_rebridged

ACTIONS = ('baseline', 'checked_zero', 'checked_inferred_target',
           'checked_seed_original', 'checked_seed_target', 'checked_seed_opposite')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--teacher-diagnostic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--identity-audit', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=6)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    states = sorted(args.states.rglob('*_i35.pt'))
    if len(states) != 48:
        raise ValueError('Expected all48 frozen TRAIN index35 states')
    root = args.work / 'stage3_grounded_repair'
    masks = {r['state_id']: r for r in json.loads((root/'train_masks_minimal_v15/mask_manifest.json').read_text())['rows']}
    diagnostic = args.teacher_diagnostic
    teachers = {r['state_id']: r for r in json.loads((diagnostic/'teacher_manifest.json').read_text())['rows']}
    gate = geometry_gate(json.loads((diagnostic/'geometry_report.json').read_text()),set(teachers))
    tasks={r['prompt_id']:r for r in json.loads((root/'train_pilot_minimal_v14/train_prompts.json').read_text())}
    if set(teachers)!={sid for sid,row in masks.items() if row['eligible']} or any(row['split']!='train' for row in masks.values()):raise ValueError('Changed train mask cohort')
    audit=json.loads(args.identity_audit.read_text())
    if audit.get('terminal_outcomes_used') is not False:
        raise ValueError('Identity audit must use intermediate evidence only')
    identities={r['state_id']:r for r in audit['rows']}
    if set(identities)!=set(teachers) or len(audit['rows'])!=len(identities) or audit['split']!='train':
        raise ValueError('Incomplete or duplicate TRAIN intermediate identity audit')
    diagnostic_records=list(map(json.loads,(diagnostic/'records.jsonl').read_text().splitlines()))
    originals={r['state_id']:r for r in diagnostic_records if r['action']=='original_clean'}
    for sid,row in identities.items():
        if row['state_fingerprint']!=masks[sid]['state_fingerprint'] or row['source_mask_sha256']!=masks[sid]['mask_sha256'] or row['original_clean_image_sha256']!=originals[sid]['image_sha256']:
            raise ValueError('Identity audit does not match the original intermediate state')
        if type(row['independent_object_instances']) is not bool:
            raise ValueError('Identity gate must be an explicit Boolean')
    source = Path(__file__).parent
    log = args.output/f'actions_shard{args.shard}.jsonl'
    config = {'actions': ACTIONS, 'check_index': 35, 'repair_index': 35, 'window': 1,
              'source_sha256': {n:file_hash(source/n) for n in
                               ('runtime.py','fm_rebridge.py','build_train_minimal_candidates.py',
                                'train_geometry_gate.py','candidate_provenance.py')},
              'identity_audit_sha256': file_hash(args.identity_audit),
              'diagnostic_records_sha256':file_hash(diagnostic/'records.jsonl'),
              'geometry_report_sha256': file_hash(diagnostic/'geometry_report.json'),
              'teacher_manifest_sha256': file_hash(diagnostic/'teacher_manifest.json'),
              'teacher_sha256': {sid:r['teacher_sha256'] for sid,r in teachers.items()},
              'mask_manifest_sha256':file_hash(root/'train_masks_minimal_v15/mask_manifest.json'),
              'states_sha256':{str(q):file_hash(q) for q in states[args.shard::args.num_shards]},
              'shard':args.shard,'num_shards':args.num_shards,'pilot_prompts_sha256':file_hash(root/'train_pilot_minimal_v14/train_prompts.json'),'split':'train','optimizer_steps':0,
              'terminal_outcome_used_by_action':False,
              'note':'Fixed late-check probe. Recompute first velocity on all rebridged inputs. Same original seed; no new noise/seed selection.'}
    fingerprint = freeze_config(args.output/f'config_shard{args.shard}.json', config, log.exists())
    done = set()
    if log.exists():
        for row in map(json.loads,log.read_text().splitlines()):
            verify_completed(row,fingerprint);done.add((row['state_id'],row['action']))
    runtime = Runtime(args.work)
    with log.open('a') as stream:
        for path in states[args.shard::args.num_shards]:
            state = torch.load(path,map_location='cpu',weights_only=False)
            sid = state['state_id'];record = masks[sid]
            if state['split'] != 'train' or state['flow_index']!=35 or state['prompt']!=tasks[state['prompt_id']]['prompt'] or state['seed'] not in tasks[state['prompt_id']]['seeds'] or not state['zero_intervention_test']['passed'] or record['state_fingerprint'] != state['state_fingerprint']:
                raise ValueError('Invalid state/mask identity')
            identity_valid=bool(identities.get(sid,{}).get('independent_object_instances',False))
            eligible = bool(record['eligible'] and gate.get(sid,False) and identity_valid)
            payload = None
            if eligible:
                teacher = teachers[sid]
                if state['evidence']['evidence'] >= 0 or file_hash(teacher['teacher_path']) != teacher['teacher_sha256']:
                    raise ValueError('Changed check/teacher')
                payload = torch.load(teacher['teacher_path'],map_location='cpu',weights_only=False)
                if payload['state_fingerprint'] != state['state_fingerprint'] or payload['mask_sha256'] != record['mask_sha256']:
                    raise ValueError('Teacher from another state')
            for action in ACTIONS:
                if (sid,action) in done: continue
                desired = None
                if eligible and action not in ('baseline','checked_zero'):
                    direction = {'checked_inferred_target':'target','checked_seed_original':'original',
                                 'checked_seed_target':'target','checked_seed_opposite':'opposite'}[action]
                    desired = payload['latents'][direction].to(runtime.device)
                rng_restore(state['rng']);torch.cuda.synchronize();started=time.perf_counter()
                bridge_info = None
                with torch.no_grad():
                    if desired is not None:
                        final,cost,bridge_info = continue_rebridged(runtime,state,desired,
                            noise_source='inferred' if action=='checked_inferred_target' else 'seed')
                    else:
                        final,cost = replay(runtime,state,window=1)
                image = runtime.decode(final);torch.cuda.synchronize()
                cost['wall_time_s'] = time.perf_counter()-started
                if desired is None and not torch.equal(final,state['baseline_final_latent'].to(runtime.device)):
                    raise RuntimeError('Zero/untriggered continuation not exact')
                # Reach/check35 computes velocities0..35 (36 calls), then 14
                # remaining calls. Every edited arm, inferred or seed residual,
                # recomputes this velocity on the NEW full input (equal cost).
                rebridged = bridge_info is not None
                cost.update(full_network_velocity_calls=50 if rebridged else 49,
                            full_visual_block_calls=500 if rebridged else 490)
                directory=args.output/action/sid;(directory/'samples').mkdir(parents=True,exist_ok=True)
                image_path=directory/'samples/0000.png';image.save(image_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                final_path=directory/'final_latent.pt';torch.save({'latent':final.cpu(),'state_id':sid,'action':action},final_path)
                row={k:state[k] for k in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t',
                                         'state_fingerprint','zero_intervention_test','runtime')}
                row.update(action=action,configuration_sha256=fingerprint,image_path=str(image_path),final_latent=str(final_path),
                           image_path_sha256=file_hash(image_path),final_latent_sha256=file_hash(final_path),cost=cost,
                           original_mask_eligible=record['eligible'],intermediate_geometry_valid=gate.get(sid,False),
                           intermediate_identity_valid=identity_valid,intervention_eligible=eligible,intervention_applied=desired is not None,
                           teacher_sha256=teachers[sid]['teacher_sha256'] if eligible else None,bridge=bridge_info,
                           added_trainable_parameters=0,unique_parameters=state['runtime']['model_unique_parameters'],FLOPs=None,
                           action_note='Offline Linear FM teacher rebridge, not shared-loop success',
                           cost_notes={'prefix_cache_reused':not rebridged,'late_evidence_precomputed':True,'teacher_geometry_precomputed':True,
                                       'cost_scope':'wall_time is remaining flow and final decode only; MMU, VAE, detector and composition separately charged',
                                       'mmu_queries_for_late_check':0 if action=='baseline' else 2,
                                       'geometry_gate_detector_calls':3 if eligible and action!='baseline' else 0,
                                       'teacher_encoding_calls':int(desired is not None),
                                       'teacher_roundtrip_decode_calls':int(desired is not None)})
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps({'state_id':sid,'action':action,'applied':desired is not None,'status':'ok'}),flush=True)


if __name__=='__main__':main()
