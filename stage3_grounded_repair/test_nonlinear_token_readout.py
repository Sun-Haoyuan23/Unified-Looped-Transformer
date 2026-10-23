import types
import unittest
import torch
from torch import nn
from nonlinear_token_readout import NonlinearTokenReadout, joint_token_features, patch_delta


class FakeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_core = [None, None]
        self.norm = nn.LayerNorm(4, elementwise_affine=False)
        self.visual_project = nn.Linear(4, 3, bias=False)
        self.position_project = nn.Linear(2, 3, bias=False)
        self.encoder = nn.Linear(2, 3, bias=False)
        self.feedback = nn.Linear(3, 4, bias=False)
        self.core_calls = 0
    def run_core(self, index, hidden, kwargs):
        self.core_calls += 1
        # Nonuniform change avoids LayerNorm cancelling a pure common offset.
        return hidden + torch.tensor([.1, .2, .4, .8])


class ReadoutTests(unittest.TestCase):
    def test_initial_zero_then_hidden_gradient(self):
        torch.manual_seed(8)
        head = NonlinearTokenReadout(torch.ones(6), outputs=3, width=4)
        x = torch.randn(2, 5, 6, requires_grad=True)
        self.assertTrue(torch.equal(head(x), torch.zeros(2, 5, 3)))
        (head(x) - 1).square().mean().backward()
        self.assertGreater(float(head.output.weight.grad.norm()), 0)
        with torch.no_grad():head.output.weight.fill_(.1)
        x.grad = None
        head(x).square().mean().backward()
        self.assertGreater(float(x.grad.norm()), 0)

    def test_matched_trainable_initialization_and_input_validation(self):
        torch.manual_seed(9);a = NonlinearTokenReadout(torch.ones(6), 3, 4)
        torch.manual_seed(9);b = NonlinearTokenReadout(torch.ones(6) * 2, 3, 4)
        for pa, pb in zip(a.parameters(), b.parameters()):self.assertTrue(torch.equal(pa, pb))
        with self.assertRaises(ValueError):a(torch.ones(2, 5))
        with self.assertRaises(ValueError):a(torch.ones(2, 6).double())

    def test_actual_shared_route_and_no_loop(self):
        op = types.SimpleNamespace(outer=FakeBranch(), inner=FakeBranch(), condition=torch.ones(1, 2))
        hidden = torch.arange(48).reshape(2, 6, 4).float()
        features = {name: (hidden, {'use_cache': False}) for name in ('outer', 'inner')}
        positions = torch.tensor([[[0, 5]]])
        plain, pc = joint_token_features(op, features, positions, 'no_loop')
        loop, lc = joint_token_features(op, features, positions, 'loop2')
        self.assertEqual(pc, {'outer': 0, 'inner': 0})
        self.assertEqual(lc, {'outer': 4, 'inner': 4})
        self.assertEqual(plain.shape, (1, 4, 18))
        self.assertFalse(torch.equal(plain, loop))
        features['outer'][1]['use_cache'] = True
        with self.assertRaises(ValueError):joint_token_features(op, features, positions)

    def test_physical_cfg_delta(self):
        def unpatchify(p, h, w):return p
        runtime = types.SimpleNamespace(patch=1, model=types.SimpleNamespace(unpatchify=unpatchify))
        patches = torch.arange(8).reshape(1, 4, 2).float().requires_grad_()
        latent = torch.zeros(2, 2, 2, 2)
        delta = patch_delta(runtime, patches, latent)
        self.assertEqual(delta.shape, latent.shape)
        self.assertTrue(torch.equal(delta[0], delta[1]))
        delta.sum().backward()
        self.assertTrue(torch.equal(patches.grad, torch.ones_like(patches) * 2))


if __name__ == '__main__':unittest.main()
