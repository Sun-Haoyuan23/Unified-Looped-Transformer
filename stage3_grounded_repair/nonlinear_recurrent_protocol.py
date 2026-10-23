"""Frozen inputs and common initialization for the nonlinear recurrent pilot."""
import json
from pathlib import Path
import torch
from analytic_readout_operator import load_adapter
from candidate_provenance import file_hash
from nonlinear_recurrent_repair import NonlinearRecurrentRepair
from nonlinear_token_readout import NonlinearTokenReadout,patch_delta
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator
from runtime import tensor_hash

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003');ROOT=WORK/'stage3_grounded_repair'


def frozen_inputs():
    prior=ROOT/'optimizer_separation_v62'
    if json.loads((prior/'pipeline_status.json').read_text())['phase']!='complete':
        raise ValueError('Require completed fixed v62')
    cfg=json.loads((prior/'config.json').read_text())['configuration']
    for name,digest in cfg['source_sha256'].items():
        if name not in ('train_optimizer_separation.py','run_optimizer_separation.py','bounded_lbfgs.py','test_bounded_lbfgs.py'):
            if file_hash(ROOT/name)!=digest:raise ValueError('Frozen interface changed: '+name)
    corpus=ProcessCorpus(ROOT)
    if corpus.input_hashes!=cfg['process_corpus_hashes']:raise ValueError('Frozen TRAIN corpus changed')
    checkpoint=ROOT/'process_recurrent_train_v52/joint_process/operator_step100.pt'
    if file_hash(checkpoint)!=cfg['checkpoint_sha256']:raise ValueError('Fixed v52 operator changed')
    heads={mode:prior/mode/'head_lbfgs_final.pt' for mode in ('trained_loop2','trained_no_loop')}
    results=json.loads((prior/'results.json').read_text())
    for mode,path in heads.items():
        if file_hash(path)!=results[mode]['head_checkpoint_sha256']:raise ValueError('Fixed v62 head changed')
    target_path=ROOT/'nonlinear_readout_capacity_v60/training_targets.pt'
    if file_hash(target_path)!=cfg['training_targets_sha256']:raise ValueError('Physical TRAIN targets changed')
    targets=torch.load(target_path,map_location='cpu',weights_only=False)
    keys=targets['keys'];labels=targets['labels'];ids=sorted(set(corpus.teachers)|set(corpus.correct))
    if [q['state_id'] for q in keys]!=ids or ids!=cfg['state_ids']:raise ValueError('TRAIN ordering changed')
    refs={q['state_id']:q for q in json.loads((ROOT/'preservation_rank_audit_v59/design_fingerprints.json').read_text())}
    for sid,label in zip(ids,labels):
        if tensor_hash(label)!=refs[sid]['target_patch_sha256']:raise ValueError('Physical target fingerprint changed')
        if file_hash(ROOT/'states/repair_train_targets_v36'/(sid+'.pt'))!=cfg['state_sha256'][sid]:
            raise ValueError('Physical TRAIN state bytes changed')
    sources=dict(operator=file_hash(checkpoint),heads={m:file_hash(p) for m,p in heads.items()},
        v62_config=file_hash(prior/'config.json'),training_targets=file_hash(target_path),
        corpus=corpus.input_hashes,state_sha256=cfg['state_sha256'])
    return corpus,checkpoint,heads,dict(zip(ids,labels)),sources


def make_repair(runtime,arm,checkpoint,heads,fresh_zero=False):
    torch.manual_seed(20261008);torch.cuda.manual_seed_all(20261008)
    op=build_operator(runtime,'joint_ilsd')
    load_adapter(op,torch.load(checkpoint,map_location='cpu',weights_only=False)['operator'])
    mode='trained_no_loop' if arm=='no_loop' else 'trained_loop2'
    payload=torch.load(heads[mode],map_location=runtime.device,weights_only=False)
    if payload['mode']!=mode or not payload['final_accepted_iterate']:raise ValueError('Wrong fixed warm-start head')
    head=NonlinearTokenReadout(payload['head']['input_rms'],64,128).to(runtime.device)
    if not fresh_zero:head.load_state_dict(payload['head'])
    return NonlinearRecurrentRepair(op,head,arm)


def training_target(runtime,corpus,state,features,repair,labels):
    sid=state['state_id'];preserve=sid in corpus.correct
    target=corpus.target(runtime,state,features,repair.operator,preserve)
    original=state['latent'].to(runtime.device).float()
    gold=patch_delta(runtime,labels[sid][None].to(runtime.device),original)
    if not torch.equal(gold,(target['input']-original)):
        raise ValueError('Patch ordering or physical target changed')
    return dict(delta=gold.detach(),final=target['final'].detach(),preservation=preserve)


def tensor_fingerprints(repair):
    return {name:tensor_hash(value) for name,value in repair.portable_state()['operator'].items()}, \
           {name:tensor_hash(value) for name,value in repair.head.state_dict().items()}


def gradient_groups(repair):
    groups=dict(head=0.,evidence=0.,outer_core=0.,inner_core=0.)
    for name,param in repair.named_parameters():
        if param.grad is None:continue
        if not torch.isfinite(param.grad).all():raise ValueError('Nonfinite gradient')
        value=float(param.grad.abs().sum())
        if name.startswith('head.'):groups['head']+=value
        elif '.encoder.' in name:groups['evidence']+=value
        elif '.outer.low_rank.' in name:groups['outer_core']+=value
        elif '.inner.low_rank.' in name:groups['inner_core']+=value
    return groups
