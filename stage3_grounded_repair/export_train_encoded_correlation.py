"""Export completed TRAIN pilot results, causal groups, contact sheet and source hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import shutil
from export_results import export
from candidate_provenance import file_hash

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);args=p.parse_args()
root=args.work/'stage3_grounded_repair';out=root/'oracle_encoded_correlation_train24x2_v23'
if json.loads((out/'evaluation_status.json').read_text())['phase']!='complete':
    raise ValueError('Do not export incomplete/missing measurements as final results')
export(out)
records=list(map(json.loads,(out/'all_actions.jsonl').read_text().splitlines()))
report=json.loads((out/'paired_report.json').read_text());pairs=report['pairs']
triggered={r['state_id'] for r in records if r['intervention_eligible']}
localized={r['state_id'] for r in records if r['original_mask_eligible']}
groups={'all48':{r['state_id'] for r in records},'localized'+str(len(localized)):localized,'triggered'+str(len(triggered)):triggered}
summary={}
for name,ids in groups.items():
    summary[name]={}
    for action in report['candidate_actions']:
        rows=[r for r in pairs if r['state_id'] in ids and r['action']==action]
        summary[name][action]=dict(states=len(rows),prompts=len({r['prompt_id'] for r in rows}),
            wrong_to_correct=sum(r['semantic_flip']==1 for r in rows),correct_to_wrong=sum(r['semantic_flip']==-1 for r in rows),
            qualified_repairs=sum(r['semantic_flip']==1 and r['hps_delta']>=0 and r['pick_delta']>=0 for r in rows),
            delta={m:statistics.mean(r[m+'_delta'] for r in rows) for m in ('pick','hps','siglip')},
            mean_full_velocity_calls=statistics.mean(r['candidate_cost']['full_network_velocity_calls'] for r in rows),
            mean_full_visual_blocks=statistics.mean(r['candidate_cost']['full_visual_block_calls'] for r in rows))
(out/'triggered_report.json').write_text(json.dumps(summary,indent=2))
flips=[r for r in pairs if r['semantic_flip']==1]
(out/'repair_gate_report.json').write_text(json.dumps(dict(
    action_level_flips=len(flips),independent_corrected_states=len({r['state_id'] for r in flips}),
    qualified_teacher_states=sorted({r['state_id'] for r in flips if r['hps_delta']>=0 and r['pick_delta']>=0}),
    flips=[{k:r[k] for k in ('state_id','prompt_id','seed','action','pick_delta','hps_delta','siglip_delta')} for r in flips],
    bootstrap_clusters=report['n_prompts'],optimizer_steps=0,teacher_type='offline horizontal scene reflection/VAE, not learned shared recurrence'),indent=2))
from PIL import Image,ImageDraw
actions=('baseline','checked_zero','checked_inferred_uncorrelated_target','checked_inferred_correlated_target','checked_seed_correlated_target','checked_seed_correlated_wrong_axis')
lookup={(r['state_id'],r['action']):r for r in records};states=sorted(triggered)
board=Image.new('RGB',(216*len(actions),246*len(states)),'white');draw=ImageDraw.Draw(board)
for i,sid in enumerate(states):
    for j,action in enumerate(actions):
        im=Image.open(lookup[sid,action]['image_path']).convert('RGB');im.thumbnail((216,216));board.paste(im,(j*216,i*246))
        draw.text((j*216,i*246+216),sid+'\n'+action,fill='black')
board.save(out/'v23_train_terminal_contact.jpg',quality=92)
for name in ('export_train_encoded_correlation.py','export_results.py','evaluate_repair_pairwise.py','score_repair_candidates.py','prepare_geneval_layout.py'):
    shutil.copy2(root/name,out/'source'/name)
shutil.copy2(args.work/'stage2b/geneval/evaluation/evaluate_images.py',out/'source/official_evaluate_images.py')
(out/'source_snapshot.json').write_text(json.dumps({f.name:file_hash(f) for f in (out/'source').iterdir()},indent=2))
from training_pairs import validate_training_pairs,selected_pairs,preservation_states,TrainingGateError
try:
    data_gate=validate_training_pairs(report,records)
    teachers=selected_pairs(report,records);preservation=preservation_states(report,records)
    for sid,row in teachers.items():
        if file_hash(row['final_latent'])!=row['final_latent_sha256']:raise RuntimeError('Changed train teacher target')
    data_gate.update(selected_teacher_ids=list(teachers),real_baseline_correct_preservation_states=preservation,optimizer_steps=0)
except TrainingGateError as error:
    data_gate=dict(passed=False,error=str(error),optimizer_steps=0);teachers={}
(out/'training_readiness.json').write_text(json.dumps(data_gate,indent=2))
diagnostic=root/'scene_mirror_geometry_train10_v18'
geometry_rows=list(map(json.loads,(diagnostic/'records.jsonl').read_text().splitlines()))
board=Image.new('RGB',(216*6,246*10),'white');draw=ImageDraw.Draw(board)
order=('original_clean','original_roundtrip','target_pixels','target_roundtrip','opposite_pixels','opposite_roundtrip')
for i,sid in enumerate(sorted({r['state_id'] for r in geometry_rows})):
    for j,a in enumerate(order):
        row=next(r for r in geometry_rows if r['state_id']==sid and r['action']==a)
        im=Image.open(row['image_path']).convert('RGB');im.thumbnail((216,216));board.paste(im,(216*j,246*i));draw.text((216*j,246*i+216),sid+'\n'+a,fill='black')
board.save(diagnostic/'v18_scene_mirror_geometry_contact.jpg',quality=92)
archive=root/'transfers/stage3_train_v23_results.tgz'
with tarfile.open(archive,'w:gz') as tar:
    for f in list(out.rglob('*'))+list(diagnostic.rglob('*')):
        if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.jsonl','.log','.py','.jpg','.pt') and (f.suffix!='.pt' or 'teacher_latents' in f.parts):
            tar.add(f,arcname=str(f.relative_to(root)))
print(json.dumps(dict(training_readiness=data_gate,archive=str(archive),sha256=file_hash(archive),summary=summary,qualified_teachers=[r['state_id'] for r in flips if r['hps_delta']>=0 and r['pick_delta']>=0])))
