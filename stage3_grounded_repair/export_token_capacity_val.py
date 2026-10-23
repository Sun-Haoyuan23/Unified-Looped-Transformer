"""Archive completed fixed-checkpoint pilot and independent val, without PT states."""
import argparse
import json
import pathlib
import shutil
import tarfile
from candidate_provenance import file_hash
from export_results import export

p=argparse.ArgumentParser();p.add_argument('--work',type=pathlib.Path,required=True)
args=p.parse_args();root=args.work/'stage3_grounded_repair'
train=root/'token_capacity_and_val_v25';val=root/'token_repair_val_v26'
if json.loads((train/'pipeline_status.json').read_text())['phase']!='complete':
    raise RuntimeError('Training/val pipeline is not complete')
summary={'optimizer_updates':0,'val_terminal_images':0,'controller_started':False,
         'core_comparison':'exploratory; adapter initialization seed not matched in v25',
         'inference':'no external mask or pixel teacher','FLOPs':None,'cores':{}}
hashes={}
for core in (8,2):
    directory=train/f'core{core}';experiment=val/f'core{core}'
    rows=list(map(json.loads,(directory/'train.jsonl').read_text().splitlines()))
    complete=json.loads((directory/'training_complete.json').read_text())
    if len(rows)!=50 or [r['step'] for r in rows]!=list(range(1,51)) or complete['steps']!=50:
        raise RuntimeError('Expected exactly fifty recorded optimizer updates')
    if json.loads((experiment/'evaluation_status.json').read_text())['phase']!='complete':
        raise RuntimeError('Incomplete independent val scoring')
    export(experiment)
    records=list(map(json.loads,(experiment/'all_actions.jsonl').read_text().splitlines()))
    if len(records)!=120 or any(r['split']!='val' for r in records):
        raise ValueError('Validation split/count differs')
    report=json.loads((experiment/'paired_report.json').read_text())
    if report['n_states']!=24 or report['n_prompts']!=12 or not report['matched_zero_intervention_passed']:
        raise ValueError('Validation frozen state/zero gate differs')
    fit=[r for r in rows if not r['preservation_step']]
    summary['cores'][str(core)]={'optimizer_updates':len(rows),'teacher_updates':len(fit),
        'preservation_updates':sum(r['preservation_step'] for r in rows),
        'first_teacher_terminal_loss':fit[0]['terminal_latent_loss'],
        'last_teacher_terminal_loss':fit[-1]['terminal_latent_loss'],
        'checkpoint_sha256':file_hash(directory/'operator_step50.pt'),
        'max_memory_gib':max(r['peak_gb'] for r in rows),
        'val_states':report['n_states'],'val_prompts':report['n_prompts'],
        'val_by_action':report['by_action'],
        'quality_guarded_semantic_oracle':report['offline_oracle']['quality_guarded_semantic']}
    summary['optimizer_updates']+=len(rows);summary['val_terminal_images']+=len(records)
    for name in ('operator_step25.pt','operator_step50.pt','training_config.json','train.jsonl','training_complete.json'):
        f=directory/name;hashes[str(f.relative_to(root))]=file_hash(f)
    for name in ('paired_report.json','official_results_keyed.jsonl','all_actions.jsonl','scores.jsonl'):
        f=experiment/name;hashes[str(f.relative_to(root))]=file_hash(f)
    # Display all five actions for the same frozen state; no success-based selection.
    from PIL import Image,ImageDraw
    actions=('baseline','conditional_zero','conditional_revisit','shuffled_evidence','opposite_evidence')
    lookup={(r['state_id'],r['action']):r for r in records}
    states=sorted({r['state_id'] for r in records})
    board=Image.new('RGB',(216*5,246*len(states)),'white');draw=ImageDraw.Draw(board)
    for i,sid in enumerate(states):
        for j,a in enumerate(actions):
            im=Image.open(lookup[sid,a]['image_path']).convert('RGB');im.thumbnail((216,216))
            board.paste(im,(j*216,i*246));draw.text((j*216,i*246+216),sid+'\n'+a,fill='black')
    board.save(experiment/'val_all_states_contact.jpg',quality=90)
(train/'completed_summary.json').write_text(json.dumps(summary,indent=2))
(train/'completed_artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
audit=train/'export_source';audit.mkdir(exist_ok=True)
for name in ('export_token_capacity_val.py','export_results.py','evaluate_repair_pairwise.py','score_repair_candidates.py','prepare_geneval_layout.py'):
    shutil.copy2(root/name,audit/name)
(train/'export_source_hashes.json').write_text(json.dumps({f.name:file_hash(f) for f in audit.iterdir()},indent=2))
archive=root/'transfers/stage3_token_v25_v26_results.tgz'
with tarfile.open(archive,'w:gz') as tar:
    for directory in (train,val):
        for f in sorted(directory.rglob('*')):
            if not f.is_file() or f.is_symlink():continue
            if f.suffix not in ('.json','.jsonl','.log','.py','.jpg','.pt'):continue
            if f.suffix=='.pt' and not f.name.startswith('operator_step'):continue
            tar.add(f,arcname=str(f.relative_to(root)))
print(json.dumps({'archive':str(archive),'sha256':file_hash(archive),
                  'optimizer_updates':summary['optimizer_updates'],'val_images':summary['val_terminal_images']}))
