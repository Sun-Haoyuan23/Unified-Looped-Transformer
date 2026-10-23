"""Per-iteration connected-reference graph with unchanged recurrent forward."""
import torch
from anchored_recurrent_repair import branch_inputs
from multibudget_response_repair import shared_response_prefixes
from nonlinear_recurrent_repair import NonlinearRecurrentRepair
from nonlinear_token_readout import patch_delta
from process_alignment_diagnostics import recomputed_reference

ARMS=('cached_trainable4','recomputed_trainable4','recomputed_frozen4','no_loop')
OLD_ARMS=dict(cached_trainable4='anchor_trainable4',recomputed_trainable4='anchor_trainable4',recomputed_frozen4='anchor_frozen4',no_loop='no_loop')


class ReferenceRecomputedRepair(NonlinearRecurrentRepair):
    def __init__(self,operator,head,arm):
        if arm not in ARMS:raise ValueError('Unknown reference arm')
        mode='no_loop' if arm=='no_loop' else ('frozen_loop' if arm=='recomputed_frozen4' else 'trainable_loop')
        super().__init__(operator,head,mode);self.reference_arm=arm

    def forward_state(self,runtime,state,features,positions):
        if features.get('_provenance')!=(state['state_fingerprint'],state['prompt'],state['flow_index']):raise ValueError('Stale physical features')
        values=[];hidden={};calls={}
        for name in ('outer','inner'):
            branch=getattr(self.operator,name);h,kw=features[name]
            field,coords,feedback,begin,end=branch_inputs(branch,h,self.operator.condition,positions)
            if self.reference_arm=='no_loop':v=h+feedback;calls[name]=0
            elif self.reference_arm=='cached_trainable4':v,c=shared_response_prefixes(branch,h,feedback,kw,(4,),True);v=v[4];calls[name]=c
            else:v=recomputed_reference(branch,h,feedback,kw,4);calls[name]=16
            hidden[name]=v[:1,begin:end]
            values.append(torch.cat((branch.norm(hidden[name]).float(),field,coords),-1))
        self.last_shared_calls=calls
        return patch_delta(runtime,self.head(torch.cat(values,-1)),state['latent']),hidden

    def latent_delta(self,runtime,state,features,positions):return self.forward_state(runtime,state,features,positions)[0]

    def portable_state(self):
        return dict(**super().portable_state(),reference_arm=self.reference_arm,outer_budget=0 if self.reference_arm=='no_loop' else 4,
                    inner_budget=0 if self.reference_arm=='no_loop' else 4,reference_connected=True,reference_recomputed_per_iteration=self.reference_arm.startswith('recomputed'))


def build_reference(runtime,arm,op_path,rms,warm):
    from process_four_loop_protocol import build
    template=build(runtime,OLD_ARMS[arm],op_path,rms,warm)
    return ReferenceRecomputedRepair(template.operator,template.head,arm)
