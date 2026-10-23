"""Separate analytical readout deployment and actual shared-core call accounting."""
from contextlib import contextmanager
import torch


READOUT_KEYS = ('outer.readout.weight', 'inner.readout.weight',
                'outer.readout.bias', 'inner.readout.bias')


def converted_readouts(adapter, coefficient):
    outer = adapter['outer.readout.weight']
    inner = adapter['inner.readout.weight']
    outputs, outer_dim = outer.shape
    inner_outputs, inner_dim = inner.shape
    if (any(adapter[key].dtype != torch.float32 for key in READOUT_KEYS) or
            adapter['outer.readout.bias'].shape != (outputs,) or
            adapter['inner.readout.bias'].shape != (outputs,) or
            inner_outputs != outputs or coefficient.shape != (outer_dim + inner_dim + 1, outputs)
            or coefficient.dtype != torch.float64 or not torch.isfinite(coefficient).all()):
        raise ValueError('Invalid FP64 coefficient or actual readout dimensions')
    result = dict(adapter)
    result.update({
        'outer.readout.weight': coefficient[:outer_dim].T.contiguous().to(outer.dtype),
        'inner.readout.weight': coefficient[outer_dim:outer_dim+inner_dim].T.contiguous().to(inner.dtype),
        'outer.readout.bias': coefficient[-1].to(adapter['outer.readout.bias'].dtype),
        'inner.readout.bias': torch.zeros_like(adapter['inner.readout.bias'])})
    if any(not torch.isfinite(result[key]).all() for key in READOUT_KEYS):
        raise ValueError('Nonfinite FP32 conversion')
    for key in set(adapter) - set(READOUT_KEYS):
        if not torch.equal(adapter[key], result[key]):
            raise ValueError('Non-readout adapter changed')
    return result, dict(outer_input_dimensions=outer_dim, inner_input_dimensions=inner_dim,
                        output_patch_dimensions=outputs, common_bias_branch='outer')


@contextmanager
def recurrence_mode(operator, mode):
    """Keep feedback/readout unchanged, count only cores actually executed."""
    if mode not in ('loop2', 'no_loop'):
        raise ValueError('Unknown fixed recurrence mode')
    counts = {'outer': 0, 'inner': 0}
    previous = []
    try:
        for name in ('outer', 'inner'):
            branch = getattr(operator, name)
            original = branch.run_core
            had_override = 'run_core' in branch.__dict__
            previous.append((branch, original, had_override))
            def call(index, hidden, kwargs, name=name, original=original):
                if mode == 'no_loop':
                    return hidden
                counts[name] += 1
                return original(index, hidden, kwargs)
            branch.run_core = call
        yield counts
    finally:
        for branch, original, had_override in previous:
            if had_override:
                branch.run_core = original
            else:
                del branch.run_core


def load_adapter(operator, adapter):
    expected = {key for key in operator.state_dict() if '.shared_core.' not in key}
    if set(adapter) != expected:
        raise ValueError('Adapter keys do not match the frozen architecture')
    loaded = operator.load_state_dict(adapter, strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) != set(operator.state_dict()) - expected:
        raise ValueError('Unexpected frozen-core checkpoint loading mismatch')


def compare_analytical_prediction(operator, runtime, features, positions, latent, coefficient, mode):
    """Inspect the actual FP32 head inputs, without reading any teacher targets."""
    inputs, outputs, hooks = {}, {}, []
    try:
        for name in ('outer', 'inner'):
            def hook(module, args, output, name=name):
                if name in inputs:
                    raise ValueError('Expected exactly one requested budget readout')
                inputs[name] = args[0].detach().cpu().double().squeeze(0)
                outputs[name] = output.detach().cpu().double().squeeze(0)
            hooks.append(getattr(operator, name).readout.register_forward_hook(hook))
        with recurrence_mode(operator, mode) as counts:
            delta = operator.latent_delta(runtime, features, positions, latent)
        x = torch.cat((inputs['outer'], inputs['inner'],
                       torch.ones(inputs['outer'].shape[0], 1, dtype=torch.float64)), dim=1)
        reference = x @ coefficient.cpu()
        actual = outputs['outer'] + outputs['inner']
        # Standard FP32 matmul tolerance fixed before endpoint/quality scores.
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
        patches = actual.float().unsqueeze(0).to(latent.device)
        reconstructed = runtime.model.unpatchify(patches, latent.shape[-2]//runtime.patch,
                                                 latent.shape[-1]//runtime.patch)
        reconstructed = reconstructed.transpose(1, 2).reshape_as(delta[:1])
        torch.testing.assert_close(delta[:1], reconstructed, rtol=1e-5, atol=1e-6)
        if not torch.equal(delta[0], delta[1]):
            raise ValueError('Physical delta differs across CFG branches')
        difference = actual - reference
        return dict(max_absolute_patch_error=float(difference.abs().max()),
            rms_patch_error=float(difference.square().mean().sqrt()),
            relative_rms_patch_error=float(difference.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-12)),
            feature_dimensions=x.shape[1], patch_dimensions=reference.shape[1],
            actual_shared_calls=dict(counts), strict_cfg_delta_equal=True,
            fp32_conversion_tolerance=dict(rtol=1e-4, atol=1e-5))
    finally:
        for hook in hooks:
            hook.remove()
