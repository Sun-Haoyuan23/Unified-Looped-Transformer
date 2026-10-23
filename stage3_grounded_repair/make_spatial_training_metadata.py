"""Generate balanced NEW spatial train prompts, excluding all official prompts.

Reference GenEval schema: include[0] is the reference; include[1] is the
subject with position=[relation,0]. These are additional TRAIN prompts only.
"""
import argparse
import json
import random
from pathlib import Path
from make_prompt_splits import read_metadata,normalize_prompt


def generate(official,names,count=120,seed=20261004):
    if count%4: raise ValueError('Use a multiple of four for relation balance')
    forbidden={normalize_prompt(r['prompt']) for r in official}
    relations=['left of','right of','above','below']
    pairs=[(a,b) for a in names for b in names if a!=b]
    rng=random.Random(seed);rng.shuffle(pairs)
    rows=[]
    for rel in relations:
        chosen=0
        for subject,reference in pairs:
            article=lambda obj:'an' if obj[0].lower() in 'aeiou' else 'a'
            prompt=f'a photo of {article(subject)} {subject} {rel} {article(reference)} {reference}'
            key=normalize_prompt(prompt)
            if key in forbidden: continue
            rows.append({'tag':'position','include':[{'class':reference,'count':1},
                {'class':subject,'count':1,'position':[rel,0]}],'prompt':prompt,
                'source':'stage3_new_training_combinations'})
            forbidden.add(key);chosen+=1
            if chosen==count//4: break
        if chosen<count//4: raise RuntimeError('Insufficient unique object combinations')
    rng.shuffle(rows)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--object-names',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=120)
    args=p.parse_args()
    names=[s.strip() for s in args.object_names.read_text().splitlines() if s.strip()]
    rows=generate(read_metadata(args.metadata),names,args.count)
    if args.output.exists(): raise FileExistsError('Do not overwrite frozen prompt source')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps({'new_train_prompts':len(rows),'official_overlap':0,'file':str(args.output)}))


if __name__=='__main__': main()
