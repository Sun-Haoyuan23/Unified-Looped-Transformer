"""Joint shared recurrence with a nonlinear physical-state readout.

Freezing low-rank parameters preserves their forward contribution. The route
changes only whether shared blocks execute, never whether loaded LoRA applies.
Teachers, masks, pixels and endpoint labels are absent from the forward.
"""
import torch
from torch import nn
from nonlinear_token_readout import joint_token_features,patch_delta

ARMS=('no_loop','frozen_loop','trainable_loop')


class NonlinearRecurrentRepair(nn.Module):
    def __init__(self,operator,head,arm):
        super().__init__()
        if arm not in ARMS or operator.outer is None or operator.inner is None:
            raise ValueError('Require a declared joint recurrent arm')
        self.operator,self.head,self.arm=operator,head,arm
        operator.requires_grad_(False)
        for branch in (operator.outer,operator.inner):
            # set_core_learning(False) would REMOVE the loaded low-rank update.
            # All routes retain the same loaded effective weights when invoked.
            branch.learn_core=True
        for name,param in operator.named_parameters():
            if '.shared_core.' in name or '.readout.' in name:
                continue
            param.requires_grad_('low_rank' not in name or arm=='trainable_loop')
        head.requires_grad_(True)
        self.last_shared_calls=None

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def latent_delta(self,runtime,state,features,positions):
        identity=(state['state_fingerprint'],state['prompt'],state['flow_index'])
        if features.get('_provenance')!=identity:
            raise ValueError('Pre-feedback features belong to a different physical state')
        tokens,calls=joint_token_features(self.operator,features,positions,
                                        'no_loop' if self.arm=='no_loop' else 'loop2')
        self.last_shared_calls=calls
        return patch_delta(runtime,self.head(tokens),state['latent'])

    def portable_state(self):
        return dict(operator=self.operator.adapter_state(),
                    head={k:v.detach().cpu() for k,v in self.head.state_dict().items()},arm=self.arm)


def normalized_losses(delta,gold_delta,original,final,target_final,baseline,preservation):
    """Equal physical/terminal normalized objectives; targets are TRAIN-only."""
    state_scale=(original.square().mean() if preservation else gold_delta.square().mean()).clamp_min(1e-5).detach()
    terminal_scale=(baseline.square().mean() if preservation else (target_final-baseline).square().mean()).clamp_min(1e-5).detach()
    physical=(delta-gold_delta.detach()).square().mean()/state_scale
    terminal=(final-target_final.detach()).square().mean()/terminal_scale
    return .5*(physical+terminal),physical,terminal


def balanced_schedule(teachers,correct,seed=20261008):
    """Exactly 105 teacher and 105 correct updates, all 35 correct seen 3 times."""
    import random
    teachers,correct=sorted(teachers),sorted(correct)
    if len(teachers)!=3 or len(correct)!=35 or set(teachers)&set(correct):
        raise ValueError('Require the fixed three teachers and 35 true correct TRAIN states')
    rng=random.Random(seed);rows=[]
    for _ in range(3):
        preservation=correct.copy();rng.shuffle(preservation)
        # Balance repairs globally below for exact teacher coverage.
        for sid in preservation:
            rows.append((None,False));rows.append((sid,True))
    repairs=teachers*35;rng.shuffle(repairs);cursor=0
    for index,(sid,preserve) in enumerate(rows):
        if not preserve: rows[index]=(repairs[cursor],False);cursor+=1
    return rows
