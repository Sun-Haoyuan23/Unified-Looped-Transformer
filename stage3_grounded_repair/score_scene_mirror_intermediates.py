"""Locate quality losses before/after VAE and subsequent flow; TRAIN only."""
import concurrent.futures,json,os,subprocess,time
from pathlib import Path
from candidate_provenance import file_hash

work=Path('/home/shy_23/iterative_visual_stage1_20261003');root=work/'stage3_grounded_repair'
out=root/'scene_mirror_geometry_train10_v18/quality_diagnostic_v1';out.mkdir()
rows=list(map(json.loads,(out.parent/'records.jsonl').read_text().splitlines()))
baselines={r['state_id']:r for r in map(json.loads,(root/'oracle_scene_mirror_train24x2_v19/all_actions.jsonl').read_text().splitlines()) if r['action']=='baseline'}
records=[]
for row in rows:
    base=baselines[row['state_id']]
    if base['split']!='train' or base['state_fingerprint']!=row['state_fingerprint'] or base['prompt']!=row['prompt'] or file_hash(row['image_path'])!=row['image_sha256']:
        raise ValueError('Changed TRAIN intermediate source')
    r={k:base[k] for k in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t','state_fingerprint')}
    r.update(action='baseline' if row['action']=='original_clean' else row['action'],image_path=row['image_path'],image_path_sha256=row['image_sha256'])
    records.append(r)
path=out/'actions_shard0.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
(out/'source_hashes.json').write_text(json.dumps({n:file_hash(root/n) for n in ('score_scene_mirror_intermediates.py','score_repair_candidates.py')},indent=2))
status=out/'status.json';status.write_text(json.dumps(dict(phase='scoring',images=len(records),optimizer_steps=0)))
GPUS={'pickscore':'GPU-67e8fd78-bc55-3da3-30da-a6bc83ab7a23','hps':'GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da','siglip':'GPU-4a177c46-c73f-d963-8d61-b7fd90196038'}
def run(metric):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[metric],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(work/'deps'))
    with (out/(metric+'.log')).open('w') as log:
        subprocess.run(['/home/shy_23/venvs/show_o2/bin/python',str(root/'score_repair_candidates.py'),'--work',str(work),
                        '--records',str(path),'--metrics',metric,'--batch-size','4','--output',str(out/(metric+'.jsonl'))],
                        cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,GPUS))
    from score_repair_candidates import merge_score_rows,read_jsonl
    merged=merge_score_rows(records,[read_jsonl(out/(m+'.jsonl')) for m in GPUS],require_metrics=tuple(GPUS))
    (out/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in merged))
    lookup={(r['state_id'],r['action']):r for r in merged};deltas=[]
    for r in merged:
        base=lookup[r['state_id'],'baseline']
        deltas.append(dict(state_id=r['state_id'],action=r['action'],**{m:r[m]-base[m] for m in GPUS}))
    (out/'paired_quality_deltas.json').write_text(json.dumps(deltas,indent=2))
    status.write_text(json.dumps(dict(phase='complete',images=len(records),optimizer_steps=0,time=time.time())))
except Exception as error:
    status.write_text(json.dumps(dict(phase='failed',error=str(error),time=time.time())));raise
