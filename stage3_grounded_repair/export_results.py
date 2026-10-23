"""Export original official results with explicit keys for portable reanalysis."""
import argparse
import json
from pathlib import Path
from evaluate_repair_pairwise import read_jsonl,evaluate


def export(experiment):
    experiment=Path(experiment)
    records=read_jsonl(experiment/'all_actions.jsonl')
    lookup={str(Path(row['image_path']).resolve()):row for row in records}
    if len(lookup)!=len(records):raise RuntimeError('Action images have duplicate resolved paths')
    keyed=[]
    for row in read_jsonl(experiment/'official_results.jsonl'):
        source=lookup[str(Path(row['filename']).resolve())]
        keyed.append(dict(row,state_id=source['state_id'],action=source['action']))
    output=experiment/'official_results_keyed.jsonl'
    output.write_text(''.join(json.dumps(row)+'\n' for row in keyed))
    recomputed=evaluate(records,keyed,read_jsonl(experiment/'scores.jsonl'))
    original=json.loads((experiment/'paired_report.json').read_text())
    if recomputed!=original:raise RuntimeError('Exported keyed analysis differs from original report')
    print(json.dumps({'export':str(output),'rows':len(keyed),'analysis_exact':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',type=Path)
    export(parser.parse_args().experiment)
