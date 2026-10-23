"""Map arbitrary state/action IDs to official numeric GenEval folders."""
import argparse
import json
from pathlib import Path
from score_repair_candidates import load_action_records


def prepare(records,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    mapping=[]
    for index,row in enumerate(sorted(records,key=lambda r:(r['state_id'],r['action']))):
        image=Path(row['image_path']).resolve()
        metadata_path=image.parent.parent/'metadata.jsonl'
        if not metadata_path.is_file():raise FileNotFoundError(metadata_path)
        metadata=json.loads(metadata_path.read_text())
        if metadata['prompt']!=row['prompt']:raise ValueError('GenEval must use original prompt metadata')
        folder=output/f'{index:06d}'
        samples=folder/'samples';samples.mkdir(parents=True,exist_ok=True)
        dest=samples/'0000.png'
        if dest.is_symlink() or dest.exists():
            if dest.resolve()!=image:raise RuntimeError('Existing official sample maps to a different action')
        else:dest.symlink_to(image)
        (folder/'metadata.jsonl').write_text(json.dumps(metadata)+'\n')
        mapping.append({'state_id':row['state_id'],'action':row['action'],
            'filename':str(dest),'image_path':str(image)})
    (output/'layout_mapping.json').write_text(json.dumps(mapping,indent=2))
    return mapping


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--records',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=load_action_records(args.records)
    mapping=prepare(rows,args.output)
    print(json.dumps({'official_samples':len(mapping),'layout':str(args.output)}))


if __name__=='__main__':main()
