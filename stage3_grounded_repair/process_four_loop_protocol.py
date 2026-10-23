"""Frozen initialization, additive hidden targets and v77 prerequisites."""
import json
import torch
from analytic_readout_operator import load_adapter
from anchored_recurrent_repair import branch_inputs
from candidate_provenance import file_hash
from check_four_loop_response_interface import NAMES as PRIOR_NAMES
from expanded_recurrent_protocol import ROOT,WORK,proof_inputs,make_repair
from process_four_loop_repair import ProcessFourLoopRepair,hidden_target
from runtime import tensor_hash

PROTOCOL='STAGE3_PROCESS_FOUR_LOOP_PROTOCOL_20261009.md'
CHECK=ROOT/'process_four_loop_gradient_v77';TRAIN=ROOT/'process_four_loop_train_v78'
NAMES=tuple(dict.fromkeys(PRIOR_NAMES+('process_four_loop_repair.py','process_four_loop_protocol.py',
    'test_process_four_loop_repair.py','check_process_four_loop_gradient.py','run_process_four_loop_gradient.py',
    'train_process_four_loop.py','run_process_four_loop_training.py')))


def inputs():
    corpus,old_path,rms,prior=proof_inputs()
    warm=ROOT/'expanded_fullbatch_capacity_v74/trainable_loop/repair_lbfgs_final.pt'
    done=json.loads((warm.parent/'training_complete.json').read_text())
    if file_hash(warm)!=done['checkpoint_sha256']:raise ValueError('v74 warm checkpoint changed')
    for version in (74,75,76):
        suffix='v74_complete_portable_verification.json' if version==74 else f'v{version}_portable_verification.json'
        if not json.loads((ROOT/suffix).read_text())['passed']:raise ValueError('Completed portable prerequisite missing')
    return corpus,old_path,rms,warm


def build(runtime,arm,old_path,rms,warm):
    old='no_loop' if arm=='no_loop' else ('frozen_loop' if arm=='anchor_frozen4' else 'trainable_loop')
    template=make_repair(runtime,old,old_path,rms)
    load_adapter(template.operator,torch.load(warm,map_location='cpu',weights_only=False)['operator'])
    with torch.no_grad():
        for branch in (template.operator.outer,template.operator.inner):
            branch.feedback.weight.zero_()
            if branch.feedback.bias is not None:branch.feedback.bias.zero_()
    return ProcessFourLoopRepair(template.operator,template.head,arm)


def aligned_hidden(corpus,state,target,features,positions,operator):
    sid=state['state_id'];preserve=target['preservation'];payload=None;artifact=None
    if not preserve:
        manifest=json.loads((ROOT/'expanded_recurrent_gradient_v71/target_feature_manifest.json').read_text())
        q=next(r for r in manifest if r['state_id']==sid);artifact=q['sha256']
        if file_hash(q['path'])!=artifact:raise ValueError('Scored additive feature artifact changed')
        payload=torch.load(q['path'],map_location=target['input'].device,weights_only=False)
        if payload['state_id']!=sid or payload['split']!='train':raise ValueError('Hidden target identity/split')
        if payload['target_input_sha256']!=tensor_hash(target['input']) or payload['target_final_sha256']!=tensor_hash(target['final']):
            raise ValueError('Hidden target is not the scored additive input/final')
        if not torch.equal(payload['modality_positions'],positions):raise ValueError('Conditional visual token span changed')
        for n,v in payload['features'].items():
            if tensor_hash(v)!=q['tensor_sha256'][n]:raise ValueError('Raw target hidden changed')
    targets={};scales={};fingerprints={}
    for name in ('outer','inner'):
        branch=getattr(operator,name);h,_=features[name]
        offset,length=positions[0,0].tolist();source=h[:1,offset+1:offset+length]
        truth=source if preserve else payload['features'][name]
        if source.shape!=truth.shape or not torch.isfinite(truth).all():raise ValueError('Invalid conditional hidden target')
        targets[name],scales[name]=hidden_target(source,truth,preserve)
        fingerprints[name]=dict(source_raw_sha256=tensor_hash(source),target_raw_sha256=tensor_hash(truth),
            normalized_target_sha256=tensor_hash(targets[name]),scale=float(scales[name]))
    row=dict(state_id=sid,split='train',preservation=preserve,input_sha256=tensor_hash(target['input']),
        delta_sha256=tensor_hash(target['delta']),final_sha256=tensor_hash(target['final']),target_artifact_sha256=artifact,branches=fingerprints)
    return targets,scales,row


def verified_training_inputs():
    corpus,old_path,unused,warm=inputs();proof=json.loads((CHECK/'pipeline_status.json').read_text())
    portable=json.loads((ROOT/'v77_portable_verification.json').read_text())
    if proof['phase']!='passed' or not portable['passed'] or not proof['all_four_initial_tensors_equal']:raise ValueError('Complete v77 proof required')
    if file_hash(ROOT/'transfers/stage3_process_four_loop_v77_results.tgz')!=portable['archive_sha256']:raise ValueError('v77 archive changed')
    config=json.loads((CHECK/'config.json').read_text())['configuration']
    for name,digest in config['source_sha256'].items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Proven source changed: '+name)
    if file_hash(ROOT/PROTOCOL)!=config['protocol_sha256']:raise ValueError('Frozen protocol changed')
    rms_path=CHECK/'common_input_rms.pt'
    if file_hash(rms_path)!=proof['common_input_rms_file_sha256']:raise ValueError('Four-route RMS changed')
    rms=torch.load(rms_path,map_location='cpu',weights_only=False)['input_rms']
    if tensor_hash(rms)!=proof['common_input_rms_tensor_sha256']:raise ValueError('Four-route RMS tensor changed')
    if file_hash(CHECK/'target_manifest.json')!=proof['target_manifest_sha256']:raise ValueError('Fixed hidden scales/targets changed')
    return corpus,old_path,rms,warm,proof
