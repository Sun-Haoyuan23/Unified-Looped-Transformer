"""Merge independently verified TRAIN records; no new images or optimizer steps."""
import json
from pathlib import Path
import shutil
import time
from candidate_provenance import file_hash,freeze_config
from evaluate_repair_pairwise import evaluate,read_jsonl
from training_pairs import validate_training_pairs,selected_pairs,preservation_states


def main():
    r=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair')
    out=r/'repair_train_combined_v36';out.mkdir()
    inputs=(r/'oracle_encoded_correlation_train24x2_v23',r/'oracle_minimal_expansion_train32x2_v35')
    rows=[];official=[];scores=[];sets=[]
    for directory in inputs:
        records=read_jsonl(directory/'all_actions.jsonl')
        if any(q['split']!='train' for q in records):raise ValueError('Non-TRAIN data in merged corpus')
        sets.append({q['state_id'] for q in records});rows.extend(records)
        official.extend(read_jsonl(directory/'official_results_keyed.jsonl'));scores.extend(read_jsonl(directory/'scores.jsonl'))
    if sets[0]&sets[1] or len(sets[0])!=48 or len(sets[1])!=64:raise ValueError('Overlapping/changed TRAIN cohorts')
    report=evaluate(rows,official,scores);gate=validate_training_pairs(report,rows)
    if gate['selected_teacher_states']!=3 or gate['preservation_states']!=35:raise ValueError('Merged strict gate changed')
    source_hash={str(d/n):file_hash(d/n) for d in inputs for n in ('all_actions.jsonl','scores.jsonl','official_results_keyed.jsonl','paired_report.json')}
    freeze_config(out/'config.json',dict(source_inputs_sha256=source_hash,source_sha256=file_hash(__file__),
        training_gate=gate,split='train',states=112,prompts=56,optimizer_steps=0,
        interpretation='Training readiness only; no shared-loop success, independent validation or human preference claim'))
    for name,values in [('all_actions.jsonl',rows),('official_results.jsonl',official),('official_results_keyed.jsonl',official),
                         ('scores.jsonl',scores),('paired_results.jsonl',report['pairs'])]:
        (out/name).write_text(''.join(json.dumps(q)+'\n' for q in values))
    (out/'paired_report.json').write_text(json.dumps(report,indent=2));(out/'training_gate.json').write_text(json.dumps(gate,indent=2))
    state_files={q.stem:q for directory in (r/'states/train_pilot_minimal_v14',r/'states/train_pilot_expansion_v24')
                 for q in directory.glob('shard*/*_i35.pt')}
    if set(state_files)!=sets[0]|sets[1]:raise ValueError('Missing/extra merged TRAIN states')
    target=r/'states/repair_train_combined_v36';target.mkdir()
    for sid,path in state_files.items():(target/(sid+'.pt')).symlink_to(path.resolve())
    (out/'state_manifest.json').write_text(json.dumps({sid:dict(path=str(path.resolve()),sha256=file_hash(path))
        for sid,path in state_files.items()},indent=2))
    shutil.copy2(__file__,out/Path(__file__).name)
    (out/'pipeline_status.json').write_text(json.dumps(dict(phase='complete',time=time.time(),training_gate=gate,
        states=112,prompts=56,optimizer_steps=0,controller_started=False),indent=2))
    print(json.dumps(gate))


if __name__=='__main__':main()
