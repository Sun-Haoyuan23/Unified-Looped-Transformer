import unittest
import torch
from global_token_readout import GlobalTokenReadout

class TestGlobalReadout(unittest.TestCase):
    def test_zero_and_budget(self):
        h=GlobalTokenReadout(torch.ones(3716));self.assertEqual(sum(p.numel() for p in h.parameters()),483968)
        self.assertTrue(torch.equal(h(torch.randn(2,7,3716),torch.randn(2,7,64)),torch.zeros(2,7,64)))
    def test_states_do_not_mix(self):
        torch.manual_seed(17);h=GlobalTokenReadout(torch.ones(5),3,4,2)
        with torch.no_grad():h.transport_output.weight.copy_(torch.eye(3));h.local_output.weight.fill_(.1)
        x=torch.randn(2,7,5);v=torch.randn(2,7,3)
        self.assertTrue(torch.allclose(h(x,v),torch.cat([h(x[i:i+1],v[i:i+1]) for i in range(2)]),rtol=1e-6,atol=1e-7))
    def test_nonlocal_transport(self):
        h=GlobalTokenReadout(torch.ones(5),3,4,2)
        with torch.no_grad():h.query.weight.zero_();h.key.weight.zero_();h.transport_output.weight.copy_(torch.eye(3))
        x=torch.zeros(1,4,5);v=torch.zeros(1,4,3);v[0,0,0]=4
        out=h(x,v);self.assertTrue(torch.equal(out[0,:,0],torch.tensor([-3.,1.,1.,1.])))
    def test_query_key_gradients(self):
        torch.manual_seed(9);h=GlobalTokenReadout(torch.ones(5),3,4,2)
        with torch.no_grad():h.transport_output.weight.copy_(torch.eye(3))
        h(torch.randn(1,7,5),torch.randn(1,7,3)).square().mean().backward()
        for p in (h.query.weight,h.key.weight):
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
