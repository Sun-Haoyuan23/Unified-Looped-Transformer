"""Freeze a balanced train-only pilot without examining any image outcomes."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from candidate_provenance import file_hash,freeze_config


def prepare(source,output,count=24,seeds=(101,202)):
    source,output=Path(source),Path(output)
    if count<4 or count%4:raise ValueError('Pilot must be balanced across four relations')
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    buckets=defaultdict(list)
    for row in rows:
        if row['split']!='train':raise ValueError('Training pilot source contains another split')
        if row['metadata']['prompt']!=row['prompt']:
            raise ValueError('Training prompt and official metadata disagree')
        relation=row['metadata']['include'][1]['position'][0]
        if relation not in ('left of','right of','above','below'):raise ValueError('Unknown spatial relation')
        buckets[relation].append(row)
    chosen=[]
    for relation in ('left of','right of','above','below'):
        group=sorted(buckets[relation],key=lambda r:r['prompt_id'])
        if len(group)<count//4:raise ValueError('Not enough prompts for balanced pilot')
        chosen.extend(dict(row,seeds=list(seeds)) for row in group[:count//4])
    chosen=sorted(chosen,key=lambda r:r['prompt_id'])
    if len({row['prompt'] for row in chosen})!=count:raise ValueError('Duplicate training pilot prompts')
    output.parent.mkdir(parents=True,exist_ok=True)
    freeze_config(output.with_suffix('.config.json'),
        {'source_sha256':file_hash(source),'count':count,'seeds':list(seeds),
         'selection':'first sorted prompt_id within each relation; no image outcome selection'})
    text=json.dumps(chosen,ensure_ascii=False,indent=2)
    if output.exists() and output.read_text(encoding='utf-8')!=text:
        raise RuntimeError('Frozen training pilot differs')
    output.write_text(text,encoding='utf-8')
    return chosen


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=24)
    args=parser.parse_args()
    print(json.dumps({'train_prompts':len(prepare(args.source,args.output,args.count))}))
