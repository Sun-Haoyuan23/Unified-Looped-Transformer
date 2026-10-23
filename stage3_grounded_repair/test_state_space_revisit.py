import unittest
from types import SimpleNamespace
import torch
from torch import nn
from state_space_revisit import StateBranch, StateSpaceRevisit, make_operator, full_model_context
from evidence_encoder import EvidenceEncoder


class Core(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(1.25)); self.calls = 0
    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        return (hidden_states * self.weight,)


class Interfaces(unittest.TestCase):
    def fixture(self):
        enc = EvidenceEncoder(level=1, width=4)
        cores = [Core(), Core()]
        branch = StateBranch(cores, 8, 4, enc, 4)
        hidden = torch.randn(2, 8, 8)
        return branch, cores, hidden, torch.randn(1, 4), torch.tensor([[[1, 5]], [[1, 5]]])

    def test_zero_readout_and_true_revisit_calls(self):
        b, cores, h, c, pos = self.fixture()
        y = b(h, {}, c, pos)
        self.assertTrue(torch.equal(y, torch.zeros_like(y)))
        self.assertEqual([q.calls for q in cores], [1, 1])
        b(h, {}, c, pos, revisit=False)
        self.assertEqual([q.calls for q in cores], [1, 1])

    def test_feedback_modality_boundary_and_frozen_core_gradient(self):
        b, cores, h, c, pos = self.fixture()
        with torch.no_grad():
            b.feedback.weight.fill_(.02); b.readout.weight.normal_(std=.01)
        seen = []
        handle = cores[0].register_forward_pre_hook(lambda m, a, kw: seen.append(kw['hidden_states'].detach()), with_kwargs=True)
        b(h, {}, c, pos).square().mean().backward(); handle.remove()
        self.assertTrue(torch.equal(seen[0][1], h[1]))
        self.assertTrue(torch.equal(seen[0][0, :2], h[0, :2]))
        self.assertTrue(torch.equal(seen[0][0, 6:], h[0, 6:]))
        self.assertFalse(torch.equal(seen[0][0, 2:6], h[0, 2:6]))
        self.assertTrue(all(q.weight.grad is None for q in cores))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in b.encoder.parameters()), 0)

    def test_cfg_physical_state_is_identical(self):
        b, cores, h, c, pos = self.fixture()
        with torch.no_grad(): b.readout.bias.fill_(.5)
        op = StateSpaceRevisit(b, None, 'outer'); op.condition = c
        model = SimpleNamespace(unpatchify=lambda v, a, z: v)
        r = SimpleNamespace(model=model, patch=1)
        latent = torch.zeros(2, 4, 2, 2)
        delta = op.latent_delta(r, {'outer': (h, {})}, pos, latent)
        self.assertEqual(delta.shape, latent.shape)
        self.assertTrue(torch.equal(delta[0], delta[1]))
        self.assertTrue(torch.equal(delta, torch.full_like(delta, .5)))

    def test_matched_initialization_tying_and_context_restore(self):
        layers = nn.ModuleList([Core() for _ in range(4)])
        model = SimpleNamespace(showo=SimpleNamespace(model=SimpleNamespace(layers=layers), config=SimpleNamespace(hidden_size=8, use_cache=True)),
            diffusion_head_config=SimpleNamespace(hidden_size=8), diffusion_head_a=nn.ModuleList([Core(), Core()]))
        r = SimpleNamespace(model=model, original=list(model.diffusion_head_a), channels=4, patch=1, device='cpu')
        torch.manual_seed(5); outer = make_operator(r, 'outer', level=1, width=4)
        torch.manual_seed(5); head = make_operator(r, 'head_only', level=1, width=4)
        torch.manual_seed(5); joint = make_operator(r, 'joint', level=1, width=4)
        self.assertIs(joint.outer.encoder, joint.inner.encoder)
        for k, v in outer.adapter_state().items():
            self.assertTrue(torch.equal(v, head.adapter_state()[k]))
            self.assertTrue(torch.equal(v, joint.adapter_state()[k]))
        original_head = model.diffusion_head_a
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with full_model_context(r):
                self.assertFalse(model.showo.config.use_cache)
                raise RuntimeError('probe')
        self.assertIs(model.diffusion_head_a, original_head)
        self.assertTrue(model.showo.config.use_cache)


if __name__ == '__main__': unittest.main()
