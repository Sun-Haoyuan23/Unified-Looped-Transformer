"""Metadata-only pairing recovery; original images/scores/config remain intact."""
import copy
import json
from pathlib import Path
import shutil
import time
from candidate_provenance import file_hash,freeze_config
from evaluate_repair_pairwise import evaluate,read_jsonl


def normalize_cost_metadata(row):
    result=copy.deepcopy(row)
    flag=result['cost'].pop('cached_inner_prefix_used')
    if flag is not False:raise ValueError('Outer continuation cannot reuse inner prefix')
    result['cost_notes']['cached_inner_prefix_used']=flag
    return result


def main():
    r=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair')
    parent=r/'outer_capacity_v33';original=parent/'trainfit';out=parent/'evaluation_recovery_v1';out.mkdir()
    if json.loads((original/'evaluation_status.json').read_text())['error']!='cost.cached_inner_prefix_used must be a finite number or explicit null':
        raise ValueError('Unexpected pairing failure')
    rows=read_jsonl(original/'all_actions.jsonl')
    if len(rows)!=10 or any(q['split']!='train' for q in rows):raise ValueError('Unexpected TRAIN diagnosis')
    inputs=('all_actions.jsonl','official_results.jsonl','scores.jsonl','pickscore_scores.jsonl','hps_scores.jsonl','siglip_scores.jsonl')
    freeze_config(out/'config.json',dict(source_sha256=file_hash(__file__),
        raw_source_sha256={n:file_hash(original/n) for n in inputs},
        normalization='Move cached_inner_prefix_used=False from numerical cost dictionary into metadata; all images/scores/actions unchanged',
        generation_repeated=False,scoring_repeated=False,optimizer_steps=0))
    normalized=[normalize_cost_metadata(q) for q in rows]
    (out/'all_actions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in normalized))
    for n in inputs[1:]:shutil.copy2(original/n,out/n)
    report=evaluate(normalized,read_jsonl(out/'official_results.jsonl'),read_jsonl(out/'scores.jsonl'))
    (out/'paired_report.json').write_text(json.dumps(report,indent=2))
    (out/'paired_results.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in report['pairs']))
    shutil.copy2(__file__,out/'recover_outer_pairwise.py')
    from export_results import export
    export(out)
    status=dict(phase='complete',time=time.time(),optimizer_updates=50,trainfit_terminal_images=10,
        source_images_and_scores_unchanged=True,metadata_normalized=True,controller_started=False,
        paired_report=str(out/'paired_report.json'))
    (out/'evaluation_status.json').write_text(json.dumps(status,indent=2))
    (parent/'recovery_status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
