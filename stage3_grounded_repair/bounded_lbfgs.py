"""Strict closure budget with accepted-iterate rollback and explicit counters.

Torch checks max_eval after completing a line search. This wrapper enforces the
same cap inside the line search and restores its accepted starting parameters
if a trial cannot be evaluated. It never selects an iterate by image scores.
The runtime Torch source is frozen with each experiment using this wrapper.
"""
import torch


class EvaluationBudgetReached(Exception):
    pass


class BoundedLBFGS(torch.optim.LBFGS):
    def __init__(self, params, *, on_accept=None, **kwargs):
        if kwargs.get('line_search_fn') != 'strong_wolfe' or not isinstance(kwargs.get('max_eval'), int) or kwargs['max_eval'] < 1:
            raise ValueError('A positive explicit closure cap and strong_wolfe are required')
        super().__init__(params, **kwargs)
        self.hard_eval_limit = kwargs['max_eval']
        self.on_accept = on_accept
        self._in_trial = False
        self._called = False
        self.audit = dict(optimizer_step_calls=0, closure_evaluations=0, accepted_parameter_updates=0,
                          hard_evaluation_limit_hit=False)

    def _directional_evaluate(self, closure, x, t, d):
        self._in_trial = True
        try:
            return super()._directional_evaluate(closure, x, t, d)
        finally:
            # The upstream method does not restore x when closure raises.
            self._set_param(x)
            self._in_trial = False

    def _add_grad(self, step_size, update):
        before = None if self._in_trial else self._clone_param()
        super()._add_grad(step_size, update)
        if before is not None:
            changed = any(not torch.equal(p, b) for p, b in zip(self._params, before))
            if changed:
                self.audit['accepted_parameter_updates'] += 1
                square = sum(float((p.detach().double() - b.double()).square().sum())
                             for p, b in zip(self._params, before))
                row = dict(update=self.audit['accepted_parameter_updates'],
                           attempted_internal_iteration=int(self.state[self._params[0]].get('n_iter', 0)),
                           closure_evaluations=self.audit['closure_evaluations'],
                           parameter_delta_l2=square ** .5)
                if self.on_accept is not None:self.on_accept(row)

    def step(self, closure):
        if self._called:raise ValueError('This frozen diagnostic permits one optimizer.step call')
        self._called = True
        self.audit['optimizer_step_calls'] = 1
        def bounded_closure():
            if self.audit['closure_evaluations'] >= self.hard_eval_limit:
                self.audit['hard_evaluation_limit_hit'] = True
                raise EvaluationBudgetReached()
            self.audit['closure_evaluations'] += 1
            value = closure()
            if not torch.isfinite(value).all():raise ValueError('Nonfinite L-BFGS objective')
            return value
        try:
            result = super().step(bounded_closure)
            self.audit['termination'] = 'torch_return'
        except EvaluationBudgetReached:
            # _directional_evaluate already restored the last accepted iterate.
            result = None
            self.audit['termination'] = 'strict_closure_budget_last_accepted_iterate'
        finally:
            state = self.state[self._params[0]]
            self.audit['attempted_internal_iterations'] = int(state.get('n_iter', 0))
            self.audit['torch_recorded_function_evals'] = int(state.get('func_evals', 0))
        return result
