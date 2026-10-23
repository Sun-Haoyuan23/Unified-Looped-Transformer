"""Differentiate a multi-step rollout: checkpoint must preserve per-step context."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from runtime import replay, tensor_hash
from evidence_conditioned_revisit import ConditionedRevisit


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)
        self.requires_grad_(False)

    def forward(self, hidden_states, adaln_input, **kwargs):
        return (hidden_states + .08 * torch.tanh(self.projection(hidden_states)
            + adaln_input),)


class TinyRuntime:
    def __init__(self):
        self.device = torch.device('cpu')
        self.cfg = 5.
        self.grid = torch.tensor([0., .1, .3, .55, .8, 1.])
        self.original = [Core() for _ in range(10)]
        self.model = nn.Module()
        self.model.diffusion_head_a = nn.ModuleList(self.original)
        self.positions = torch.tensor([[[1, 4]], [[1, 4]]])

    def inputs(self, prompt):
        return {'modality_positions': self.positions}

    def velocity(self, x, t, kwargs):
        h = x
        for core in self.model.diffusion_head_a:
            h = core(hidden_states=h, adaln_input=t, **kwargs)[0]
        return h

    def from_prefix(self, prefix, x, t, kwargs, repair=None):
        h = prefix
        if repair is not None:
            h = repair(hidden_states=h, adaln_input=t, **kwargs)[0]
        for core in self.original[8:]:
            h = core(hidden_states=h, adaln_input=t, **kwargs)[0]
        return h


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_matches_full_rollout_gradients_with_changing_time(self):
        torch.manual_seed(71)
        r = TinyRuntime()
        x = torch.randn(2, 5, 8)
        prefix = x
        for core in r.original[:8]:
            prefix = core(hidden_states=prefix, adaln_input=r.grid[0])[0]
        state = {'latent': x, 'time_grid': r.grid, 'flow_index': 0, 'CFG': r.cfg,
            'prefix_cache': prefix, 'prefix_cache_hash': tensor_hash(prefix),
            'prefix_cache_version': 'after-original-block-7-v1', 'prompt': 'test',
            'runtime': {'base_model_hash': 'a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c'}}
        op = ConditionedRevisit(r.original[2], hidden_size=8, level=1, width=16)
        op.condition = torch.tensor([[.2, -.3, .5, 0.]])
        with torch.no_grad():
            op.film.weight.normal_(std=.02)
            op.film.bias.normal_(std=.02)
        initial_condition = op.condition.clone()
        outcomes, gradients = [], []
        target = torch.randn_like(x)
        for use_checkpoint in (False, True):
            op.zero_grad(set_to_none=True)
            op.condition = initial_condition.clone()
            final, cost = replay(r, state, repair=op, window=4,
                                 checkpoint_rollout=use_checkpoint)
            (final - target).square().mean().backward()
            outcomes.append(final.detach())
            gradients.append([p.grad.detach().clone() for p in op.trainable_parameters()])
            self.assertEqual(cost['shared_revisit_calls'], 8)
            self.assertTrue(all(p.grad is None for core in r.original for p in core.parameters()))
        torch.testing.assert_close(outcomes[0], outcomes[1], rtol=0, atol=0)
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients[0]), 0.)
        for plain, checkpointed in zip(*gradients):
            torch.testing.assert_close(plain, checkpointed, rtol=1e-5, atol=1e-7)


if __name__ == '__main__': unittest.main()
