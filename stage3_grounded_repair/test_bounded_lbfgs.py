import unittest
import torch
from bounded_lbfgs import BoundedLBFGS


def optimizer(p, cap=125):
    return BoundedLBFGS([p], lr=1, max_iter=100, max_eval=cap, history_size=20,
                        tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn='strong_wolfe')


class BoundedTests(unittest.TestCase):
    def test_quadratic_convergence_and_counter_separation(self):
        p=torch.nn.Parameter(torch.tensor([3.,-2.],dtype=torch.float64));op=optimizer(p);calls=[]
        def closure():
            op.zero_grad(set_to_none=True);loss=(p-torch.tensor([1.,4.])).square().sum();loss.backward();calls.append(1);return loss
        op.step(closure)
        self.assertLess(float((p-torch.tensor([1.,4.])).abs().max()),1e-7)
        self.assertEqual(op.audit['optimizer_step_calls'],1)
        self.assertEqual(len(calls),op.audit['closure_evaluations'])
        self.assertGreater(op.audit['accepted_parameter_updates'],0)
        self.assertLessEqual(op.audit['accepted_parameter_updates'],op.audit['attempted_internal_iterations'])
        with self.assertRaises(ValueError):op.step(closure)

    def test_budget_inside_trial_restores_exact_accepted_state(self):
        p=torch.nn.Parameter(torch.tensor([3.,-2.],dtype=torch.float64));original=p.detach().clone();op=optimizer(p,1)
        def closure():
            op.zero_grad(set_to_none=True);loss=p.square().sum();loss.backward();return loss
        op.step(closure)
        self.assertTrue(torch.equal(p,original))
        self.assertEqual(op.audit['closure_evaluations'],1)
        self.assertEqual(op.audit['accepted_parameter_updates'],0)
        self.assertTrue(op.audit['hard_evaluation_limit_hit'])

    def test_unexpected_trial_failure_also_restores_parameters(self):
        p=torch.nn.Parameter(torch.tensor([3.,-2.],dtype=torch.float64));original=p.detach().clone();op=optimizer(p);calls=[]
        def closure():
            calls.append(1)
            if len(calls)==2:raise RuntimeError('intentional trial failure')
            op.zero_grad(set_to_none=True);loss=p.square().sum();loss.backward();return loss
        with self.assertRaises(RuntimeError):op.step(closure)
        self.assertTrue(torch.equal(p,original))
        self.assertEqual(op.audit['accepted_parameter_updates'],0)

    def test_nonfinite_objective_is_a_failure_not_budget_completion(self):
        p=torch.nn.Parameter(torch.tensor([3.],dtype=torch.float64));original=p.detach().clone();op=optimizer(p)
        with self.assertRaises(ValueError):op.step(lambda:p.sum()*float('nan'))
        self.assertTrue(torch.equal(p,original))
        self.assertFalse(op.audit['hard_evaluation_limit_hit'])


if __name__=='__main__':unittest.main()
