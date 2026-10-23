"""Matched branch initialization and nested terminal objective controls."""
from torch.nn import functional as F
from trainable_recurrent_state import make_operator

STUDIES=('outer_ilsd','inner_ilsd','joint_ilsd','joint_terminal')

def build_operator(runtime,study):
    if study not in STUDIES:raise ValueError('Unknown fixed recurrent study')
    # Construct both branches first. Thus inner-only uses precisely the same
    # low-rank factors and token readout initialization as the joint model.
    op=make_operator(runtime,'joint',level=2,width=64,rank=4)
    arm=study.split('_',1)[0]
    if arm=='outer':op.inner=None
    elif arm=='inner':op.outer=None
    op.arm=arm;return op

def nested_loss(high,low,target,baseline,preserve,step,steps,rank_weight=.1):
    scale=(baseline.square().mean() if preserve else (target-baseline).square().mean()).clamp_min(1e-5)
    maximum=F.mse_loss(high,target)/scale
    distbase=F.mse_loss(high,baseline)/scale
    margin=.5*F.mse_loss(target,baseline).detach()/scale.detach()
    rank=maximum.new_zeros(()) if preserve else F.relu(margin+maximum-distbase)
    if low is None:
        return maximum+rank_weight*rank,dict(terminal=maximum,rank=rank,preserve=distbase if preserve else maximum.new_zeros(()),student=maximum.new_zeros(()),distill=maximum.new_zeros(()),ground_truth_fraction=0.)
    student=F.mse_loss(low,target)/scale
    studentbase=F.mse_loss(low,baseline)/scale
    studentrank=student.new_zeros(()) if preserve else F.relu(margin+student-studentbase)
    distill=F.mse_loss(low,high.detach())/scale
    fraction=(steps-step)/(steps-1)
    loss=.5*(maximum+rank_weight*rank+fraction*(student+rank_weight*studentrank)+(1-fraction)*distill)
    return loss,dict(terminal=maximum,rank=rank,preserve=distbase if preserve else maximum.new_zeros(()),student=student,distill=distill,ground_truth_fraction=fraction)
