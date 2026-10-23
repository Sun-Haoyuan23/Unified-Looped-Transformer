"""Freeze unused balanced TRAIN prompts, without reading image outcomes."""
import json
from collections import defaultdict
from pathlib import Path
from candidate_provenance import file_hash,freeze_config


def prepare(source,exclude,output,count=32,seeds=(101,202)):
    source,exclude,output=map(Path,(source,exclude,output))
    if count<4 or count%4:raise ValueError('Balanced count required')
    rows=list(map(json.loads,source.read_text().splitlines()))
    excluded=json.loads(exclude.read_text());lookup={r['prompt_id']:r for r in rows}
    if len(lookup)!=len(rows) or len({r['prompt'] for r in rows})!=len(rows):raise ValueError('Duplicate TRAIN source')
    for r in rows:
        if r['split']!='train' or r['metadata']['prompt']!=r['prompt']:raise ValueError('Split/prompt mismatch')
    excluded_ids={r['prompt_id'] for r in excluded}
    if len(excluded_ids)!=len(excluded):raise ValueError('Duplicate excluded prompt')
    for r in excluded:
        if r['split']!='train' or r['prompt_id'] not in lookup or r['prompt']!=lookup[r['prompt_id']]['prompt']:
            raise ValueError('Excluded tasks differ from TRAIN source')
    buckets=defaultdict(list)
    for r in rows:
        relation=r['metadata']['include'][1]['position'][0]
        if relation not in ('left of','right of','above','below'):raise ValueError('Invalid relation')
        if r['prompt_id'] not in excluded_ids:buckets[relation].append(r)
    chosen=[]
    for relation in ('left of','right of','above','below'):
        group=sorted(buckets[relation],key=lambda r:r['prompt_id'])
        if len(group)<count//4:raise ValueError('Insufficient unused TRAIN prompts')
        chosen.extend(dict(r,seeds=list(seeds)) for r in group[:count//4])
    chosen.sort(key=lambda r:r['prompt_id']);output.parent.mkdir(parents=True,exist_ok=True)
    freeze_config(output.with_suffix('.config.json'),dict(source_sha256=file_hash(source),excluded_tasks_sha256=file_hash(exclude),
        count=count,seeds=list(seeds),excluded_ids=sorted(excluded_ids),terminal_outcomes_used=False,
        selection='first unused sorted prompt_id per relation; no image/score selection'))
    text=json.dumps(chosen,ensure_ascii=False,indent=2)
    if output.exists() and output.read_text()!=text:raise ValueError('Changed frozen expansion')
    output.write_text(text);return chosen
