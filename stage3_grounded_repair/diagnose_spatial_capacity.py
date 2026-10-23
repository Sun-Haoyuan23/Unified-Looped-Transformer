"""Read fixed TRAIN checkpoint; inspect grid mixing/folding, no training/images."""
import json,os,pathlib,time
import torch
from torch.nn import functional as F
from runtime import Runtime,replay,rng_restore
from spatial_token_revisit import make_operator
from evidence_encoder import condition_tensor
from candidate_provenance import file_hash

w=pathlib.Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'spatial_transport_audit_v29';out.mkdir()
status=out/'pipeline_status.json'
def update(phase,**kw):status.write_text(json.dumps(dict(phase=phase,optimizer_steps=0,time=time.time(),**kw),indent=2))
try:
    update('loading')
    checkpoint=r/'spatial_capacity_v28/spatial/operator_step50.pt'
    (out/'config.json').write_text(json.dumps(dict(checkpoint_sha256=file_hash(checkpoint),
        source_sha256={n:file_hash(r/n) for n in ('diagnose_spatial_capacity.py','spatial_token_revisit.py','runtime.py')},
        split='train',actions=['conditional_revisit','opposite_evidence'],window=4,
        description='Coordinates diagnostic; reflected grid distance is teacher-action reference, not ground-truth hidden correspondence'),indent=2))
    rt=Runtime(w,vae=False);op=make_operator(rt,checkpoint=checkpoint,core_index=8)
    records=[];context={}
    def hook(module,inputs,kwargs):
        h=kwargs.get('hidden_states',inputs[0] if inputs else None);pos=kwargs['modality_positions']
        field,coords,side,a,b=module.token_field(h,pos)
        offset=module.displacement(field).tanh()*module.max_displacement
        grid=(coords+offset).reshape(1,side,side,2)
        gx=grid[:,:-1,1:]-grid[:,:-1,:-1];gy=grid[:,1:,:-1]-grid[:,:-1,:-1]
        det=(gx[...,0]*gy[...,1]-gx[...,1]*gy[...,0])/((2/(side-1))**2)
        index=((grid+1)*(side-1)/2).clamp(0,side-1);frac=index-index.floor()
        dominant=torch.maximum(frac,1-frac).prod(-1)
        source=h[:1,a:b].float().transpose(1,2).reshape(1,h.shape[-1],side,side)
        moved=F.grid_sample(source,grid,mode='bilinear',padding_mode='border',align_corners=True)
        anchor=F.grid_sample(source,coords.reshape(1,side,side,2),mode='bilinear',padding_mode='border',align_corners=True)
        scale=source.square().mean().clamp_min(1e-8)
        reflection=coords.clone();reflection[...,0]*=-1
        records.append(dict(**context,true_t=float(module.condition[0,3]),
            displacement_rms_tokens=float(offset.square().mean().sqrt()*(side-1)/2),
            displacement_max_tokens=float(offset.abs().max()*(side-1)/2),
            out_of_bounds_token_fraction=float((grid.abs()>1).any(-1).float().mean()),
            nonpositive_jacobian_fraction=float((det<=0).float().mean()),jacobian_mean=float(det.mean()),
            dominant_bilinear_weight_mean=float(dominant.mean()),
            mixed_token_fraction=float((dominant<.9).float().mean()),
            transported_feature_relative_mse=float((moved-anchor).square().mean()/scale),
            reflection_coordinate_mse=float((grid.reshape_as(coords)-reflection).square().mean()),
            identity_coordinate_mse=float(offset.square().mean())))
    op.register_forward_pre_hook(hook,with_kwargs=True)
    states=sorted((r/'states/trainfit_v27').glob('*_i35.pt'))
    for path in states:
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['split']!='train':raise ValueError('TRAIN-only audit')
        for action in ('conditional_revisit','opposite_evidence'):
            ev=dict(state['evidence'])
            if action=='opposite_evidence':
                for a,b in (('target_margin','opposite_margin'),('target_hidden','opposite_hidden')):ev[a],ev[b]=ev[b],ev[a]
                ev['evidence']=-ev['evidence']
            context.update(state_id=state['state_id'],action=action)
            op.condition=condition_tensor(ev,state['true_t'],rt.device)
            rng_restore(state['rng'])
            with torch.no_grad():replay(rt,state,repair=op,window=4)
            (out/'grid_audit.json').write_text(json.dumps(records,indent=2))
    update('complete',rows=len(records),terminal_images=0)
except Exception as error:update('failed',error=str(error));raise
