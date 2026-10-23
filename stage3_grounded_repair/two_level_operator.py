"""Outer/inner/joint capacity arms with matched common initialization."""
import torch
from torch import nn
from outer_shared_revisit import make_operator as make_outer
from token_conditioned_revisit import make_operator as make_inner


class TwoLevelOperator(nn.Module):
    def __init__(self,outer=None,inner=None):
        super().__init__();self.outer=outer;self.inner=inner
    def trainable_parameters(self):return [p for p in self.parameters() if p.requires_grad]
    @property
    def encoder(self):return (self.outer if self.outer is not None else self.inner).encoder
    @property
    def shared_core(self):
        return nn.ModuleDict({name:op.shared_core for name,op in (('outer',self.outer),('inner',self.inner)) if op is not None})
    @property
    def condition(self):return (self.outer if self.outer is not None else self.inner).condition
    @condition.setter
    def condition(self,value):
        for op in (self.outer,self.inner):
            if op is not None:op.condition=value.clone()


def make_operator(runtime,arm,level=2,width=128):
    if arm not in ('outer','inner','joint'):raise ValueError('Unknown recurrence arm')
    outer=make_outer(runtime,level,width)
    inner=make_inner(runtime,level,width,core_index=8)
    inner.encoder.load_state_dict(outer.encoder.state_dict())
    inner.position_project.load_state_dict(outer.position_project.state_dict())
    if arm=='joint':inner.encoder=outer.encoder
    return TwoLevelOperator(outer if arm!='inner' else None,inner if arm!='outer' else None)
