"""Read-only modulation audit; exact replay checks that observers do not alter inference."""
import json
import os
from pathlib import Path
import time
import torch
from runtime import Runtime, write_json, rng_restore, tensor_hash
from two_level_operator import make_operator
from two_level_replay import install_levels, replay_levels
from evidence_encoder import condition_tensor
from candidate_provenance import file_hash, freeze_config

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'modulation_capacity_audit_v40';out.mkdir()
try:
    diagnostics=r/'two_level_trainfit_v39';training=r/'two_level_train_v38'
    if json.loads((diagnostics/'pipeline_status.json').read_text())['phase']!='complete':
        raise ValueError('Require completed fixed TRAIN diagnosis')
    cohort=json.loads((diagnostics/'cohort.json').read_text())
    source=('audit_two_level_modulation.py','two_level_operator.py','two_level_replay.py',
            'outer_shared_revisit.py','token_conditioned_revisit.py','runtime.py','evidence_encoder.py')
    freeze_config(out/'config.json',dict(source_sha256={n:file_hash(r/n) for n in source},
        cohort_sha256=file_hash(diagnostics/'cohort.json'),state_ids=cohort['teacher_state_ids'],
        checkpoint_sha256={a:file_hash(training/a/'operator_step100.pt') for a in ('outer','inner','joint')},
        optimizer_updates=0,observer_only=True,split='train',saturation_definition='abs gamma/beta >= 0.95 * fixed0.1 bound'))
    (out/'source').mkdir()
    import shutil
    for n in source:shutil.copy2(r/n,out/'source'/n)
    write_json(out/'pipeline_status.json',dict(phase='running',optimizer_updates=0))
    runtime=Runtime(w,vae=False);rows=[];verified=[]
    for arm in ('outer','inner','joint'):
        op=make_operator(runtime,arm)
        payload=torch.load(training/arm/'operator_step100.pt',map_location='cpu',weights_only=False)
        expected={k for k in op.state_dict() if '.shared_core.' not in k}
        if set(payload['operator'])!=expected:raise ValueError('Adapter checkpoint architecture mismatch')
        op.load_state_dict(payload['operator'],strict=False)
        for sid in cohort['teacher_state_ids']:
            path=r/'states/repair_train_targets_v36'/(sid+'.pt')
            if file_hash(path)!=cohort['state_sha256'][sid]:raise ValueError('Diagnostic state changed')
            state=torch.load(path,map_location='cpu',weights_only=False)
            op.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
            handles=[];contexts={}
            def attach(level,module):
                def before(mod,args,kwargs):
                    if level=='outer' and not mod.enabled:return
                    h=args[0] if args else kwargs['hidden_states']
                    positions=mod.positions if level=='outer' else kwargs['modality_positions']
                    start,length=positions[0,0].tolist();a,b=start+1,start+length
                    contexts[level]=dict(h=h,a=a,b=b,record=dict(arm=arm,state_id=sid,level=level,
                        true_t=float(mod.condition[:,3]),tokens=b-a,bound=mod.max_update))
                def field(mod,args,result):
                    context=contexts[level];h=context['h'];a,b=context['a'],context['b']
                    raw_gamma,raw_beta=result.chunk(2,-1)
                    gamma=raw_gamma.tanh()*module.max_update;beta=raw_beta.tanh()*module.max_update
                    if gamma.shape[1]!=b-a:gamma,beta=gamma[:,a:b],beta[:,a:b]
                    normalized=module.norm(h[:1,a:b]).float()
                    intended=gamma*normalized+beta
                    cast_delta=(intended.to(h.dtype) if level=='outer' else
                        gamma.to(h.dtype)*module.norm(h[:1,a:b])+beta.to(h.dtype))
                    base=h[:1,a:b].float();changed=(h[:1,a:b]+cast_delta).float()-base
                    rms=lambda z:float(z.square().mean().sqrt())
                    context['record'].update(gamma_rms=rms(gamma),beta_rms=rms(beta),
                        gamma_saturated_fraction=float((gamma.abs()>=.95*module.max_update).float().mean()),
                        beta_saturated_fraction=float((beta.abs()>=.95*module.max_update).float().mean()),
                        hidden_rms=rms(base),intended_delta_rms=rms(intended),actual_input_delta_rms=rms(changed),
                        intended_delta_over_hidden_rms=rms(intended)/max(rms(base),1e-12),
                        bf16_input_unchanged_fraction=float((changed==0).float().mean()))
                def after(mod,args,kwargs,result):
                    if level=='outer' and not mod.enabled:return
                    context=contexts.pop(level);h=context['h'];a,b=context['a'],context['b']
                    result=result if level=='outer' else result[0]
                    update=(result[:1,a:b].float()-h[:1,a:b].float())
                    context['record'].update(shared_residual_rms=float(update.square().mean().sqrt()),
                        shared_output_unchanged_fraction=float((update==0).float().mean()))
                    rows.append(context['record'])
                handles.extend((module.register_forward_pre_hook(before,with_kwargs=True),
                    module.film.register_forward_hook(field),module.register_forward_hook(after,with_kwargs=True)))
            if op.outer is not None:attach('outer',op.outer)
            if op.inner is not None:attach('inner',op.inner)
            rng_restore(state['rng'])
            try:
                with torch.no_grad(),install_levels(runtime,op.outer):
                    final,cost=replay_levels(runtime,state,outer=op.outer,inner=op.inner,window=4)
            finally:
                for handle in handles:handle.remove()
            saved=torch.load(diagnostics/arm/'conditional_revisit'/sid/'final_latent.pt',map_location=runtime.device,weights_only=False)
            if saved['state_id']!=sid or not torch.equal(final,saved['latent']):
                raise ValueError('Observer changed the previously scored inference')
            verified.append(dict(arm=arm,state_id=sid,latent_exact=True,final_tensor_sha256=tensor_hash(final)))
    if len(rows)!=48 or len(verified)!=9:raise ValueError('Missing actual modulation windows')
    (out/'modulation_records.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows))
    write_json(out/'replay_verification.json',verified)
    groups=[]
    for arm in ('outer','inner','joint'):
        for level in ('outer','inner'):
            subset=[q for q in rows if q['arm']==arm and q['level']==level]
            if not subset:continue
            fields=('gamma_saturated_fraction','beta_saturated_fraction','bf16_input_unchanged_fraction',
                    'intended_delta_over_hidden_rms','shared_output_unchanged_fraction','shared_residual_rms')
            groups.append(dict(arm=arm,level=level,windows=len(subset),
                means={k:sum(q[k] for q in subset)/len(subset) for k in fields}))
    write_json(out/'summary.json',dict(groups=groups,interpretation='Observational bounds/quantization audit, not causal evidence of a bottleneck'))
    write_json(out/'pipeline_status.json',dict(phase='complete',optimizer_updates=0,windows=48,
        exact_previously_scored_replays=9,val_images=0,controller_started=False))
except Exception as error:
    write_json(out/'pipeline_status.json',dict(phase='failed',error=str(error),optimizer_updates=0));raise
