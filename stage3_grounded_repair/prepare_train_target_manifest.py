"""Canonical TRAIN targets from complete, separately verified action cohorts.

This selects TRAIN supervision after terminal scoring. It is not an action
comparison or a controller evaluation; missing cross-cohort actions are never
invented. Each state has its original baseline and one sourced training target.
"""
import copy
import json
from pathlib import Path
import shutil
import time
import torch
from candidate_provenance import file_hash,freeze_config
from runtime import tensor_hash
from evaluate_repair_pairwise import evaluate,read_jsonl
from training_pairs import validate_training_pairs,selected_pairs


def main():
    r=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair')
    out=r/'repair_train_targets_v36';out.mkdir()
    sources=(r/'oracle_encoded_correlation_train24x2_v23',r/'oracle_minimal_expansion_train32x2_v35')
    old=json.loads((r/'v23_portable_verification.json').read_text())
    new=json.loads((r/'v33_v35_portable_verification.json').read_text())['v35']
    if (not old['portable_exact'] or old['paired_report_sha256']!=file_hash(sources[0]/'paired_report.json') or
            not new['portable_exact'] or new['paired_report_sha256']!=file_hash(sources[1]/'paired_report.json')):
        raise ValueError('Require matching local portable verification for both TRAIN sources')
    records=[];official=[];scores=[];choices={};seen=set();selected_all={}
    input_hash={str(d/n):file_hash(d/n) for d in sources for n in ('all_actions.jsonl','official_results_keyed.jsonl','scores.jsonl','paired_report.json')}
    for directory in sources:
        rows=read_jsonl(directory/'all_actions.jsonl');report=json.loads((directory/'paired_report.json').read_text())
        selected=selected_pairs(report,rows);selected_all.update(selected)
        keys={(q['state_id'],q['action']):q for q in rows}
        semantics={(q['state_id'],q['action']):q for q in read_jsonl(directory/'official_results_keyed.jsonl')}
        quality={(q['state_id'],q['action']):q for q in read_jsonl(directory/'scores.jsonl')}
        states=sorted({q['state_id'] for q in rows})
        if seen & set(states):raise ValueError('Overlapping TRAIN states')
        seen.update(states)
        for sid in states:
            chosen=selected[sid]['action'] if sid in selected else 'checked_zero'
            choices[sid]=dict(source=str(directory),source_action=chosen,qualified_teacher=sid in selected)
            for source_action,canonical in (('baseline','baseline'),(chosen,'training_target')):
                row=copy.deepcopy(keys[sid,source_action])
                if row['split']!='train':raise ValueError('TRAIN-only target selection')
                row.update(action=canonical,original_source_action=source_action,
                    original_configuration_sha256=row['configuration_sha256'],
                    target_selection_role='TRAIN supervision; not deployed action or independent evaluation')
                if sid in selected and canonical=='training_target':
                    path=Path(row['final_latent'])
                    if file_hash(path)!=row['final_latent_sha256']:raise ValueError('Teacher bytes changed')
                    payload=torch.load(path,map_location='cpu',weights_only=False)
                    if payload['state_id']!=sid or payload['action']!=source_action:raise ValueError('Teacher payload identity mismatch')
                    original_hash=tensor_hash(payload['latent'])
                    payload=dict(payload,action=canonical,original_source_action=source_action,original_latent_sha256=original_hash)
                    target=out/'teacher_latents'/(sid+'.pt');target.parent.mkdir(exist_ok=True);torch.save(payload,target)
                    if tensor_hash(torch.load(target,map_location='cpu',weights_only=False)['latent'])!=original_hash:
                        raise ValueError('Metadata relabeling changed latent values')
                    row.update(original_final_latent=row['final_latent'],original_final_latent_sha256=row['final_latent_sha256'],
                        final_latent=str(target.resolve()),final_latent_sha256=file_hash(target),teacher_latent_tensor_sha256=original_hash)
                records.append(row)
                result=copy.deepcopy(semantics[sid,source_action]);result.update(action=canonical,original_source_action=source_action);official.append(result)
                result=copy.deepcopy(quality[sid,source_action]);result.update(action=canonical,original_source_action=source_action);scores.append(result)
    if len(seen)!=112 or len(selected_all)!=3:raise ValueError('Unexpected strict TRAIN target count')
    fingerprint=freeze_config(out/'config.json',dict(source_sha256=file_hash(__file__),source_inputs_sha256=input_hash,
        choices=choices,split='train',role='Selected TRAIN supervision only; do not report as controller/independent quality gain',
        canonical_actions=['baseline','training_target'],no_missing_actions_fabricated=True,no_image_or_score_values_changed=True,
        teacher_latent_values_unchanged=True,optimizer_steps=0))
    for row in records:row['configuration_sha256']=fingerprint
    report=evaluate(records,official,scores);gate=validate_training_pairs(report,records)
    if gate['selected_teacher_states']!=3 or gate['preservation_states']!=35:raise ValueError('Canonical training gate changed')
    for name,values in [('all_actions.jsonl',records),('official_results_keyed.jsonl',official),('scores.jsonl',scores)]:
        (out/name).write_text(''.join(json.dumps(q)+'\n' for q in values))
    (out/'paired_report.json').write_text(json.dumps(report,indent=2));(out/'training_gate.json').write_text(json.dumps(gate,indent=2))
    files={q.stem:q for d in (r/'states/train_pilot_minimal_v14',r/'states/train_pilot_expansion_v24') for q in d.glob('shard*/*_i35.pt')}
    if set(files)!=seen:raise ValueError('Missing canonical TRAIN states')
    target=r/'states/repair_train_targets_v36';target.mkdir()
    for sid,path in files.items():(target/(sid+'.pt')).symlink_to(path.resolve())
    (out/'state_manifest.json').write_text(json.dumps({sid:dict(path=str(q.resolve()),sha256=file_hash(q)) for sid,q in files.items()},indent=2))
    shutil.copy2(__file__,out/Path(__file__).name)
    (out/'pipeline_status.json').write_text(json.dumps(dict(phase='complete',time=time.time(),states=112,prompts=56,
        training_gate=gate,optimizer_steps=0,controller_started=False,role='Training readiness only'),indent=2))
    print(json.dumps(gate))


if __name__=='__main__':main()
