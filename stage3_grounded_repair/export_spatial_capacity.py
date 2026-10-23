"""Export fixed-checkpoint TRAIN capacity comparison; not a val success claim."""
import argparse,hashlib,json,pathlib,statistics,tarfile
from export_results import export

p=argparse.ArgumentParser();p.add_argument('--work',type=pathlib.Path,required=True)
args=p.parse_args();root=args.work/'stage3_grounded_repair';out=root/'spatial_capacity_v28'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
if json.loads((out/'pipeline_status.json').read_text())['phase']!='complete':raise ValueError('Incomplete run')
f=json.loads((out/'film/training_config.json').read_text())['adapter_initialization_hashes']
s=json.loads((out/'spatial/training_config.json').read_text())['adapter_initialization_hashes']
if any(s[k]!=v for k,v in f.items()):raise ValueError('Unmatched common initialization')
summary={'scope':'TRAIN-only capacity diagnostic; no independent val',
    'common_adapter_initialization_matched':True,'controller_started':False,'FLOPs':None,'arms':{}}
for arm in ('film','spatial'):
    t=out/arm;d=out/(arm+'_trainfit')
    rows=list(map(json.loads,(t/'train.jsonl').read_text().splitlines()))
    if [q['step'] for q in rows]!=list(range(1,51)):raise ValueError('Update count differs')
    export(d);records=list(map(json.loads,(d/'all_actions.jsonl').read_text().splitlines()))
    if len(records)!=8 or any(q['split']!='train' for q in records):raise ValueError('Trainfit split/count differs')
    report=json.loads((d/'paired_report.json').read_text())
    summary['arms'][arm]={'optimizer_updates':50,'parameters':json.loads((t/'training_config.json').read_text())['trainable_parameters'],
        'checkpoint_sha256':sha(t/'operator_step50.pt'),'by_action':report['by_action'],
        'fit_distances':[{k:q[k] for k in ('state_id','action','trainfit_teacher_distance','trainfit_baseline_teacher_distance')} for q in records],
        'first_teacher_loss':next(q['terminal_latent_loss'] for q in rows if not q['preservation_step']),
        'last_teacher_loss':next(q['terminal_latent_loss'] for q in reversed(rows) if not q['preservation_step']),
        'max_memory_gib':max(q['peak_gb'] for q in rows)}
    from PIL import Image,ImageDraw
    actions=('baseline','conditional_zero','conditional_revisit','opposite_evidence')
    lookup={(q['state_id'],q['action']):q for q in records};ids=sorted({q['state_id'] for q in records})
    board=Image.new('RGB',(432*4,470*2),'white');draw=ImageDraw.Draw(board)
    for i,sid in enumerate(ids):
        for j,a in enumerate(actions):
            im=Image.open(lookup[sid,a]['image_path']).convert('RGB');im.thumbnail((432,432))
            board.paste(im,(j*432,i*470));draw.text((j*432,i*470+432),sid+'\n'+a,fill='black')
    board.save(d/'trainfit_contact.jpg',quality=95)
(out/'completed_summary.json').write_text(json.dumps(summary,indent=2))
hashes={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and not f.is_symlink()
        and f.suffix in ('.json','.jsonl','.py','.pt') and 'official_layout' not in f.parts and f.name!='artifact_hashes.json'}
(out/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
archive=root/'transfers/stage3_spatial_capacity_v28_results.tgz'
with tarfile.open(archive,'w:gz') as tar:
    for f in out.rglob('*'):
        if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.jsonl','.py','.pt','.log','.jpg','.png') and 'official_layout' not in f.parts:
            tar.add(f,arcname=str(f.relative_to(root)))
print('archive_sha256='+sha(archive))
print('archive_size='+str(archive.stat().st_size))
