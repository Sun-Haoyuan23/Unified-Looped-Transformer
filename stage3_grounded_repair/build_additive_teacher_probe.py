"""Decode v49 additive endpoints; audit numerical reachability of TRAIN teachers.

Only three new candidate images. Baselines are byte-identical aliases. Original
successful teacher endpoints are never regenerated, rescored or relabeled.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch
from candidate_provenance import file_hash, freeze_config
from runtime import Runtime, tensor_hash, write_json
from training_pairs import selected_pairs, validate_training_pairs


def main():
    p=argparse.ArgumentParser()
    for name in ('work','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir()
    root=args.work/'stage3_grounded_repair'
    process=root/'teacher_process_targets_v49'
    teacher=root/'repair_train_targets_v36'
    proof=json.loads((process/'pipeline_status.json').read_text())
    if proof['phase']!='complete' or not proof['all_direct_endpoints_exact'] or not proof['all_original_bridge_inputs_exact']:
        raise ValueError('Require strict original teacher process reconstruction')
    records=[json.loads(s) for s in (teacher/'all_actions.jsonl').read_text().splitlines()]
    report=json.loads((teacher/'paired_report.json').read_text())
    gate=validate_training_pairs(report,records)
    pairs=selected_pairs(report,records)
    if gate['selected_teacher_states']!=3 or gate['preservation_states']!=35:
        raise ValueError('Strict TRAIN gate changed')
    baselines={q['state_id']:q for q in records if q['action']=='baseline'}
    manifest={q['state_id']:q for q in map(json.loads,(process/'process_manifest.jsonl').read_text().splitlines())}
    source=Path(__file__).parent
    frozen=freeze_config(args.output/'config_shard0.json',dict(
        source_sha256={n:file_hash(source/n) for n in ('build_additive_teacher_probe.py','runtime.py','candidate_provenance.py','training_pairs.py')},
        process_manifest_sha256=file_hash(process/'process_manifest.jsonl'),
        teacher_report_sha256=file_hash(teacher/'paired_report.json'),
        cohort=sorted(pairs),split='train',actions=['baseline','additive_teacher'],
        artifact_sha256={sid:manifest[sid]['artifact_sha256'] for sid in pairs},
        optimizer_updates=0,new_images=3,baseline_aliases=3,
        intervention='Known qualified teacher input reconstructed by original + float32(teacher-original)',
        purpose='Read-only numerical interface ceiling; not learned-loop success'))
    runtime=Runtime(args.work)
    with (args.output/'actions_shard0.jsonl').open('w') as log,torch.no_grad():
        for sid in sorted(pairs):
            row=manifest[sid]
            if not row['terminal_exact'] or row['cohort']!='teacher' or file_hash(row['artifact_path'])!=row['artifact_sha256']:
                raise ValueError('Process target artifact changed')
            data=torch.load(row['artifact_path'],map_location='cpu',weights_only=False)
            state=torch.load(root/'states/repair_train_targets_v36'/(sid+'.pt'),map_location='cpu',weights_only=False)
            if data['state_id']!=sid or data['split']!='train' or data['flow_index']!=35 or data['source_state_fingerprint']!=state['state_fingerprint']:
                raise ValueError('TRAIN process identity changed')
            if tensor_hash(data['target_final'])!=row['target_terminal_sha256']:
                raise ValueError('Verified canonical target tensor changed')
            for action in ('baseline','additive_teacher'):
                directory=args.output/action/sid
                (directory/'samples').mkdir(parents=True)
                image_path=directory/'samples/0000.png'
                final_path=directory/'final_latent.pt'
                started=time.perf_counter()
                if action=='baseline':
                    old=baselines[sid]
                    if file_hash(old['image_path'])!=old['image_path_sha256'] or file_hash(old['final_latent'])!=old['final_latent_sha256']:
                        raise ValueError('Original baseline bytes changed')
                    shutil.copy2(old['image_path'],image_path)
                    shutil.copy2(old['final_latent'],final_path)
                else:
                    final=data['additive_final'].to(runtime.device)
                    runtime.decode(final).save(image_path)
                    torch.save(dict(latent=final.cpu(),state_id=sid,action=action),final_path)
                (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
                result={k:state[k] for k in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t',
                    'state_fingerprint','zero_intervention_test','runtime')}
                result.update(action=action,configuration_sha256=frozen,
                    image_path=str(image_path),image_path_sha256=file_hash(image_path),
                    final_latent=str(final_path),final_latent_sha256=file_hash(final_path),
                    cost=dict(actual_new_velocity_calls=0,actual_new_decodes=int(action!='baseline'),
                        prior_v49_additive_remaining_velocity_calls=0 if action=='baseline' else 14,
                        wall_time_s=time.perf_counter()-started,FLOPs=None),
                    cost_notes=dict(baseline_alias=action=='baseline',old_KV_used=False,
                        old_inner_prefix_used=False,offline_teacher=True,new_terminal_sampling=False),
                    arithmetic_audit={k:row[k] for k in ('additive_input_exact','additive_input_max_abs_diff',
                        'additive_terminal_exact','additive_terminal_relative_l2')},
                    learned_recurrence=False,optimizer_updates=0,FLOPs=None)
                log.write(json.dumps(result)+'\n');log.flush()
    write_json(args.output/'generation_status.json',dict(success=True,expected_images=6,
        actual_new_images=3,baseline_aliases=3,optimizer_updates=0,split='train'))


if __name__=='__main__':main()
