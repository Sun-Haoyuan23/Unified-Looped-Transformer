"""Train the shared recurrent transformation with low-rank attention weights.

Frozen original weights are referenced, never copied or changed. Every recurrent
depth uses the same low-rank factors. One forward exposes exact nested prefixes
to a shared latent readout, enabling terminal intra-loop self-distillation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from state_space_revisit import StateSpaceRevisit, make_operator as make_frozen, full_model_context, capture_features


class LowRankWeight(nn.Module):
    def __init__(self, incoming, outgoing, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, incoming))
        self.B = nn.Parameter(torch.zeros(outgoing, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    def delta(self):
        return self.B @ self.A


class RecurrentBranch(nn.Module):
    def __init__(self, branch, rank=4):
        super().__init__()
        for name in ('shared_core','encoder','norm','visual_project','position_project','feedback','readout'):
            setattr(self, name, getattr(branch, name))
        self.low_rank = nn.ModuleDict()
        self.targets = []
        for index, core in enumerate(self.shared_core):
            targets = {}
            for name, module in core.named_modules():
                if isinstance(module, nn.Linear) and name.rsplit('.',1)[-1] in ('q_proj','k_proj','v_proj','o_proj'):
                    key = f'{index}__' + name.replace('.', '__')
                    self.low_rank[key] = LowRankWeight(module.in_features, module.out_features, rank)
                    targets[name+'.weight'] = key
            if len(targets) != 4:
                raise ValueError('Require exactly q/k/v/o attention projections in each shared block')
            self.targets.append(targets)
        self.learn_core = True

    def set_core_learning(self, enabled):
        self.learn_core = enabled
        self.low_rank.requires_grad_(enabled)

    def run_core(self, index, hidden, kwargs):
        core = self.shared_core[index]
        if not self.learn_core:
            return core(hidden_states=hidden, **kwargs)[0]
        weights = dict(core.named_parameters())
        replacement = {name: weights[name] + self.low_rank[key].delta().to(weights[name].dtype)
                       for name,key in self.targets[index].items()}
        # functional_call restores the module's frozen parameter references before
        # returning, while the forward graph retains gradients into A/B.
        return torch.func.functional_call(core, replacement, (), dict(hidden_states=hidden, **kwargs), strict=False)[0]

    def read_budgets(self, hidden, kwargs, condition, positions, budgets):
        if not budgets or min(budgets)<1 or max(budgets)>2 or len(set(budgets))!=len(budgets):
            raise ValueError('Fixed protocol supports recurrent budgets1 and2')
        if kwargs.get('use_cache',False) or kwargs.get('past_key_value') is not None:
            raise ValueError('No KV reuse in recurrent shared blocks')
        offset,length = positions[0,0].tolist();begin,end=offset+1,offset+length
        count=length-1;side=math.isqrt(count)
        if hidden.shape[0]!=2 or side*side!=count or count<=0 or end>hidden.shape[1]:
            raise ValueError('Invalid conditional visual span')
        line=torch.linspace(-1,1,side,device=hidden.device);yy,xx=torch.meshgrid(line,line,indexing='ij')
        coords=torch.stack((xx,yy),-1).reshape(1,count,2)
        field=F.silu(self.visual_project(self.norm(hidden[:1,begin:end]).float())+
                     self.encoder(condition)[:,None]+self.position_project(coords))
        delta=torch.zeros_like(hidden);delta[0,begin:end]=self.feedback(field)[0].to(hidden.dtype)
        h=hidden+delta;outputs={};kw={k:v for k,v in kwargs.items() if k!='hidden_states'}
        for loop in range(1,max(budgets)+1):
            for index in range(len(self.shared_core)):h=self.run_core(index,h,kw)
            if loop in budgets:
                outputs[loop]=self.readout(torch.cat((self.norm(h[:1,begin:end]).float(),field,coords),-1))
        return outputs


class RecurrentStateSpace(StateSpaceRevisit):
    def __init__(self,outer,inner,arm='joint'):
        super().__init__(outer,inner,arm);self.loop_count=2

    def trainable_parameters(self):
        return [p for name,p in self.named_parameters() if '.shared_core.' not in name and p.requires_grad]

    def set_core_learning(self,enabled):
        for branch in (self.outer,self.inner):
            if branch is not None:branch.set_core_learning(enabled)

    def latent_deltas(self,runtime,features,positions,latent,budgets=(1,2)):
        values={k:[] for k in budgets}
        for name,branch in (('outer',self.outer),('inner',self.inner)):
            if branch is None:continue
            hidden,kwargs=features[name]
            patches=branch.read_budgets(hidden,kwargs,self.condition,positions,budgets)
            for k,p in patches.items():
                decoded=runtime.model.unpatchify(p,latent.shape[-2]//runtime.patch,latent.shape[-1]//runtime.patch)
                values[k].append(decoded.transpose(1,2).reshape(1,*latent.shape[1:]).float())
        results={}
        for k,v in values.items():
            delta=torch.stack(v).sum(0);results[k]=torch.cat((delta,delta),0)
        return results

    def latent_delta(self,runtime,features,positions,latent):
        return self.latent_deltas(runtime,features,positions,latent,(self.loop_count,))[self.loop_count]


def make_operator(runtime,arm='joint',level=2,width=64,rank=4):
    base=make_frozen(runtime,arm,level,width)
    outer=RecurrentBranch(base.outer,rank) if base.outer is not None else None
    inner=RecurrentBranch(base.inner,rank) if base.inner is not None else None
    return RecurrentStateSpace(outer,inner,arm).to(runtime.device)


def replay_delta(runtime,state,delta,checkpoint_rollout=False):
    """Readout budget is bound before all14 fresh original velocity evaluations."""
    from runtime import rng_restore,tensor_hash
    from torch.utils.checkpoint import checkpoint
    if tensor_hash(state['latent'])!=state['state_fingerprint']:
        raise ValueError('Corrupted physical state')
    if not torch.equal(state['time_grid'].to(runtime.device),runtime.grid) or state['CFG']!=runtime.cfg:
        raise ValueError('Solver mismatch')
    if delta.shape!=state['latent'].shape or not torch.equal(delta[0],delta[1]):
        raise ValueError('Physical CFG delta shape/identity mismatch')
    x=state['latent'].to(runtime.device).clone()+delta;options=runtime.inputs(state['prompt'])
    grid=state['time_grid'].to(runtime.device);rng_restore(state['rng'])
    for i in range(state['flow_index'],len(grid)-1):
        def velocity(z,now=grid[i]):return runtime.velocity(z,now,options)
        v=checkpoint(velocity,x,use_reentrant=False) if checkpoint_rollout else velocity(x)
        x=x+(grid[i+1]-grid[i])*v
    return x
