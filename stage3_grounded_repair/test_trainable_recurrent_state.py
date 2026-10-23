import unittest
from types import SimpleNamespace
import torch
from torch import nn
from evidence_encoder import EvidenceEncoder
from state_space_revisit import StateBranch
from trainable_recurrent_state import RecurrentBranch, RecurrentStateSpace


class Core(nn.Module):
    def __init__(self):
        super().__init__(); self.self_attn=nn.ModuleDict({n:nn.Linear(8,8,bias=False) for n in ('q_proj','k_proj','v_proj','o_proj')});self.calls=0
    def forward(self,hidden_states,**kw):
        self.calls+=1
        a=self.self_attn
        return (hidden_states+a['o_proj'](torch.tanh(a['q_proj'](hidden_states)+a['k_proj'](hidden_states)+a['v_proj'](hidden_states))),)


class Interfaces(unittest.TestCase):
    def fixture(self):
        cores=[Core(),Core()]
        b=RecurrentBranch(StateBranch(cores,8,4,EvidenceEncoder(level=1,width=4),4),rank=2)
        return b,cores,torch.randn(2,8,8),torch.randn(1,4),torch.tensor([[[1,5]],[[1,5]]])

    def test_zero_low_rank_core_equivalence(self):
        b,cores,h,c,pos=self.fixture()
        original=cores[0](hidden_states=h)[0]
        self.assertTrue(torch.equal(b.run_core(0,h,{}),original))
        self.assertIs(b.shared_core[0],cores[0])

    def test_low_rank_gradient_and_original_parameter_restore(self):
        b,cores,h,c,pos=self.fixture();before={n:t.detach().clone() for n,t in cores[0].state_dict().items()}
        refs={n:id(p) for n,p in cores[0].named_parameters()}
        b.run_core(0,h,{}).square().mean().backward()
        self.assertGreater(sum(float(q.B.grad.abs().sum()) for k,q in b.low_rank.items() if k.startswith('0__')),0)
        self.assertTrue(all(p.grad is None for p in cores[0].parameters()))
        self.assertEqual(refs,{n:id(p) for n,p in cores[0].named_parameters()})
        self.assertTrue(all(torch.equal(v,cores[0].state_dict()[n]) for n,v in before.items()))

    def test_exact_prefix_and_shared_call_counts(self):
        b,cores,h,c,pos=self.fixture()
        with torch.no_grad():b.readout.weight.normal_(std=.01)
        both=b.read_budgets(h,{},c,pos,(1,2))
        self.assertEqual([q.calls for q in cores],[2,2])
        one=b.read_budgets(h,{},c,pos,(1,))
        self.assertTrue(torch.equal(both[1],one[1]))
        self.assertNotEqual(both[1].data_ptr(),both[2].data_ptr())

    def test_zero_all_budgets_and_cfg_identity(self):
        b,cores,h,c,pos=self.fixture();op=RecurrentStateSpace(b,None,'outer');op.condition=c
        r=SimpleNamespace(patch=1,model=SimpleNamespace(unpatchify=lambda p,a,z:p));x=torch.zeros(2,4,2,2)
        results=op.latent_deltas(r,{'outer':(h,{})},pos,x,(1,2))
        for d in results.values():
            self.assertTrue(torch.equal(d,torch.zeros_like(x)));self.assertTrue(torch.equal(d[0],d[1]))
        with torch.no_grad():b.readout.bias.fill_(.1)
        results=op.latent_deltas(r,{'outer':(h,{})},pos,x,(1,2))
        self.assertTrue(all(torch.equal(d[0],d[1]) for d in results.values()))

    def test_freeze_control_and_ilsd_teacher_stop_gradient(self):
        b,cores,h,c,pos=self.fixture();b.set_core_learning(False)
        self.assertTrue(all(not p.requires_grad for p in b.low_rank.parameters()))
        b.set_core_learning(True)
        with torch.no_grad():b.readout.weight.normal_(std=.01)
        outputs=b.read_budgets(h,{},c,pos,(1,2));outputs[1].retain_grad();outputs[2].retain_grad()
        (outputs[1]-outputs[2].detach()).square().mean().backward()
        self.assertIsNotNone(outputs[1].grad);self.assertIsNone(outputs[2].grad)
        self.assertTrue(all(p.grad is None for core in cores for p in core.parameters()))

    def test_protocol_branch_initialization_matches_joint(self):
        from recurrent_training_protocol import build_operator
        layers=nn.ModuleList([Core() for _ in range(4)])
        r=SimpleNamespace(device='cpu',channels=4,patch=1,original=[Core(),Core()],
            model=SimpleNamespace(showo=SimpleNamespace(model=SimpleNamespace(layers=layers),config=SimpleNamespace(hidden_size=8)),
                                  diffusion_head_config=SimpleNamespace(hidden_size=8)))
        initial={}
        for study in ('outer_ilsd','inner_ilsd','joint_ilsd','joint_terminal'):
            torch.manual_seed(20261005)
            initial[study]=build_operator(r,study).adapter_state()
        for study in ('outer_ilsd','inner_ilsd','joint_terminal'):
            for name,value in initial[study].items():
                self.assertTrue(torch.equal(value,initial['joint_ilsd'][name]))

if __name__=='__main__':unittest.main()
