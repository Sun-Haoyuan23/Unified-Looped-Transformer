"""Whole-head N-layer loop for Show-o2, with a shared single-prefix exit.

ELT paper: g_theta is an N-layer composite block repeated L times, not a
two-layer tied tail. Here N=5, Lmax=2, matching the teacher's ten head calls.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F


class DtypeBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, adaln_input, **kwargs):
        dtype = next(self.block.parameters()).dtype
        return self.block(hidden_states=hidden_states.to(dtype),
                          adaln_input=adaln_input.to(dtype), **kwargs)


class DtypeFinal(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, adaln_input, modality_positions):
        dtype = next(self.layer.parameters()).dtype
        return self.layer(x.to(dtype), adaln_input.to(dtype), modality_positions)


class ResidualTailBlock(nn.Module):
    """Reuse a frozen pretrained block with a pass-specific residual update."""

    def __init__(self, base, hidden_size, adapter_width):
        super().__init__()
        self.base = base
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_width),
            nn.GELU(),
            nn.Linear(adapter_width, hidden_size),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, hidden_states, adaln_input, **kwargs):
        base_dtype = next(self.base.parameters()).dtype
        output = self.base(hidden_states=hidden_states.to(base_dtype),
                           adaln_input=adaln_input.to(base_dtype), **kwargs)
        corrected = output[0].float() + self.adapter(hidden_states.float())
        return (corrected,) + output[1:]


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features,
                                          device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank,
                                          device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        self.scale = rank ** -0.5

    def forward(self, x):
        base_out = self.base(x.to(self.base.weight.dtype))
        correction = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return (base_out.float() + correction).to(base_out.dtype)


class DeltaLoRALinear(LoRALinear):
    """A shared projection plus a loop-position-specific low-rank delta."""

    def __init__(self, base, target, rank, svd_init=False):
        rank = min(rank, base.in_features, base.out_features)
        super().__init__(base, rank)
        if target.bias is not None:
            baseline = (base.bias.float() if base.bias is not None else
                        torch.zeros_like(target.bias, dtype=torch.float32))
            self.delta_bias = nn.Parameter(target.bias.float().detach() - baseline)
        else:
            self.register_parameter("delta_bias", None)
        if svd_init:
            self.initialize_svd(target)

    @torch.no_grad()
    def initialize_svd(self, target):
        delta = target.weight.float() - self.base.weight.float()
        if not torch.any(delta):
            self.a.zero_()
            self.b.zero_()
            return
        rank = self.a.shape[0]
        # Near-full-rank randomized SVD is numerically unreliable on these
        # pretrained projection differences. Use the deterministic full
        # decomposition and truncate its factors for high-rank adapters.
        if rank >= min(delta.shape) // 2:
            u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
            v = vh.transpose(0, 1)
        else:
            q = min(rank + 32, *delta.shape)
            u, singular, v = torch.svd_lowrank(delta, q=q, niter=4)
        root = singular[:rank].clamp_min(0).sqrt() * rank ** 0.25
        self.a.copy_(root[:, None] * v[:, :rank].T)
        self.b.copy_(u[:, :rank] * root[None, :])

    def forward(self, x):
        if hasattr(self, "_merged_weight"):
            return F.linear(x.to(self._merged_weight.dtype),
                            self._merged_weight, self._merged_bias)
        output = super().forward(x)
        if self.delta_bias is not None:
            output = output + self.delta_bias.to(output.dtype)
        return output

    @torch.no_grad()
    def merge_for_inference(self):
        """Materialize the low-rank correction to isolate arithmetic error.

        This optional cache increases inference memory; it is an explicit
        diagnostic and must be reported separately from tied-weight runtime.
        """
        delta = (self.b @ self.a) * self.scale
        self._merged_weight = (self.base.weight.float() + delta).to(
            self.base.weight.dtype)
        bias = self.base.bias.float() if self.base.bias is not None else None
        if self.delta_bias is not None:
            bias = ((bias if bias is not None else
                     torch.zeros_like(self.delta_bias)) + self.delta_bias)
        self._merged_bias = (bias.to(self.base.weight.dtype)
                             if bias is not None else None)


def merge_svd_adapters_for_inference(model):
    """Optionally cache composed weights for a numerical-ablation run."""
    for module in model.diffusion_head_a.modules():
        if isinstance(module, DeltaLoRALinear):
            module.merge_for_inference()


def _shallow_module_clone(module):
    clone = copy.copy(module)
    clone._modules = module._modules.copy()
    return clone


def _delta_loop_block(base, target, attention_rank, mlp_rank,
                      adaln_rank, svd_init):
    """Reuse all five block cores; adapt each second-loop projection."""
    block = _shallow_module_clone(base)
    attention = _shallow_module_clone(base.self_attn)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(attention, name, DeltaLoRALinear(
            getattr(base.self_attn, name), getattr(target.self_attn, name),
            attention_rank, svd_init))
    # RMSNorm weights are tiny and layer-specific; initialize exactly.
    for name in ("q_norm", "k_norm"):
        if hasattr(target.self_attn, name):
            setattr(attention, name, copy.deepcopy(getattr(target.self_attn, name)))
    block.self_attn = attention
    mlp = _shallow_module_clone(base.mlp)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, name, DeltaLoRALinear(
            getattr(base.mlp, name), getattr(target.mlp, name), mlp_rank, svd_init))
    block.mlp = mlp
    block.adaLN_modulation = nn.Sequential(
        base.adaLN_modulation[0],
        DeltaLoRALinear(base.adaLN_modulation[1],
                        target.adaLN_modulation[1], adaln_rank, svd_init))
    block.input_layernorm = copy.deepcopy(target.input_layernorm)
    block.post_attention_layernorm = copy.deepcopy(target.post_attention_layernorm)
    return block


def install_svd_loop_head(model, rank=128, svd_init=False, trainable=True,
                          attention_rank=None, mlp_rank=None, adaln_rank=None):
    """Five complete visual blocks run twice; second pass has low-rank deltas.

    First pass is the pretrained first five blocks. The second pass reuses
    their attention, MLP and AdaLN matrices, adding only rank-r corrections.
    The original last five blocks provide a one-time SVD initialization.
    """
    original = list(model.diffusion_head_a)
    if len(original) != 10:
        raise ValueError("Expected ten pretrained visual-head blocks")
    ar = attention_rank or rank
    mr = mlp_rank or rank
    tr = adaln_rank or rank
    first = original[:5]
    second = [_delta_loop_block(first[i], original[i + 5], ar, mr, tr, svd_init)
              for i in range(5)]
    model.diffusion_head_a = nn.ModuleList(
        [DtypeBlock(block) for block in first + second])
    model._elt_architecture = (("svd_loop", 5, 2, rank)
                               if (ar, mr, tr) == (rank, rank, rank) else
                               ("svd_loop", 5, 2, ar, mr, tr))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for block in second:
            for module in block.modules():
                if isinstance(module, DeltaLoRALinear):
                    module.a.requires_grad_(True)
                    module.b.requires_grad_(True)
                    if module.delta_bias is not None:
                        module.delta_bias.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def install_svd_sequence_head(model, rank=128, svd_init=False, trainable=True,
                              attention_rank=None, mlp_rank=None,
                              adaln_rank=None):
    """Repeat each of five visual cores immediately, for ten executed calls.

    The first visit uses pretrained even layer 2*i; the second visit shares
    its core and approximates pretrained odd layer 2*i+1 with SVD deltas.
    This changes only recurrence order relative to the 5+5 cycle layout.
    """
    original = list(model.diffusion_head_a)
    if len(original) != 10:
        raise ValueError("Expected ten pretrained visual-head blocks")
    ar = attention_rank or rank
    mr = mlp_rank or rank
    tr = adaln_rank or rank
    paired = [_delta_loop_block(original[2*i], original[2*i+1],
                                ar, mr, tr, svd_init) for i in range(5)]
    ordered = [block for i in range(5)
               for block in (original[2*i], paired[i])]
    model.diffusion_head_a = nn.ModuleList(
        [DtypeBlock(block) for block in ordered])
    model._elt_architecture = (("svd_sequence", 5, 2, rank)
                               if (ar, mr, tr) == (rank, rank, rank) else
                               ("svd_sequence", 5, 2, ar, mr, tr))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for block in paired:
            for module in block.modules():
                if isinstance(module, DeltaLoRALinear):
                    module.a.requires_grad_(True)
                    module.b.requires_grad_(True)
                    if module.delta_bias is not None:
                        module.delta_bias.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def install_svd_middle_head(model, rank=128, svd_init=False, trainable=True,
                            attention_rank=None, mlp_rank=None,
                            adaln_rank=None):
    """Keep two prefix/suffix blocks; revisit three complete middle blocks.

    Execution: 0,1,2,3,4,2',3',4',8,9. The primed calls share the cores
    of 2,3,4 with SVD deltas targeting pretrained blocks 5,6,7.
    """
    original = list(model.diffusion_head_a)
    if len(original) != 10:
        raise ValueError("Expected ten pretrained visual-head blocks")
    ar = attention_rank or rank
    mr = mlp_rank or rank
    tr = adaln_rank or rank
    revisited = [_delta_loop_block(original[2+i], original[5+i],
                                   ar, mr, tr, svd_init) for i in range(3)]
    ordered = original[:5] + revisited + original[8:]
    model.diffusion_head_a = nn.ModuleList(
        [DtypeBlock(block) for block in ordered])
    model._elt_architecture = (("svd_middle", 2, 3, 2, 2, rank)
                               if (ar, mr, tr) == (rank, rank, rank) else
                               ("svd_middle", 2, 3, 2, 2, ar, mr, tr))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for block in revisited:
            for module in block.modules():
                if isinstance(module, DeltaLoRALinear):
                    module.a.requires_grad_(True)
                    module.b.requires_grad_(True)
                    if module.delta_bias is not None:
                        module.delta_bias.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def install_shared_middle_head(model, trainable=True):
    """Compress the visual head to 2 prefix + (3 tied middle x 2) + 2 suffix.

    The one-visit exit executes seven blocks; the two-visit exit executes ten.
    Both exits use the same pretrained suffix and prediction head. No layer
    copies, low-rank deltas, or hidden full teacher survive in the student.
    """
    original = list(model.diffusion_head_a)
    if len(original) != 10:
        raise ValueError("Expected ten original visual blocks")
    prefix = [DtypeBlock(block) for block in original[:2]]
    shared = [DtypeBlock(block.to(torch.float32)) for block in original[2:5]]
    suffix = [DtypeBlock(block) for block in original[8:]]
    model.diffusion_head_a = nn.ModuleList(prefix + shared + shared + suffix)
    model._elt_architecture = ("shared_middle", 2, 3, 2, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for block in shared:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def install_shared_single_head(model, source_index, target_index, trainable=False):
    """Replace one pretrained visual block with a second visit to another.

    This preserves nine of ten distinct blocks and permits a nine-call exit
    by skipping the second visit. It is a diagnostic for minimally disruptive
    parameter reduction, not a quality claim.
    """
    original = list(model.diffusion_head_a)
    if len(original) != 10 or not (0 <= source_index < 10 and
                                0 <= target_index < 10 and
                                source_index < target_index):
        raise ValueError("Expected ten blocks and distinct valid indices")
    shared = original[source_index].to(torch.float32)
    # Separate wrappers distinguish the visits while keeping weights tied.
    ordered = [DtypeBlock(shared if i in (source_index, target_index) else block)
               for i, block in enumerate(original)]
    model.diffusion_head_a = nn.ModuleList(ordered)
    model._elt_architecture = ("shared_single", source_index, target_index)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for parameter in shared.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def install_svd_single_head(model, source_index, target_index, rank=256,
                            svd_init=True, trainable=False):
    """One true shared-block revisit with a low-rank visit-specific delta.

    Keep nine complete original blocks. Visit `source_index` once with its
    pretrained core, then revisit that same core at `target_index` with a
    compressed delta initialized against the removed pretrained block.
    The target must be immediately after the source so the shorter path is a
    literal prefix and the full path is a local correction.
    """
    original = list(model.diffusion_head_a)
    if len(original) != 10 or not (0 <= source_index < 9 and
                                target_index == source_index + 1):
        raise ValueError("Expected adjacent shared source/target in ten blocks")
    source = original[source_index].to(torch.float32)
    revisit = _delta_loop_block(source, original[target_index],
                                rank, rank, rank, svd_init)
    ordered = [DtypeBlock(revisit if i == target_index else
                          source if i == source_index else block)
               for i, block in enumerate(original)]
    model.diffusion_head_a = nn.ModuleList(ordered)
    model._elt_architecture = ("svd_single", source_index, target_index, rank)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for parameter in source.parameters():
            parameter.requires_grad_(True)
        for module in revisit.modules():
            if isinstance(module, DeltaLoRALinear):
                module.a.requires_grad_(True)
                module.b.requires_grad_(True)
                if module.delta_bias is not None:
                    module.delta_bias.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


class LoRAMLP(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.gate_proj = LoRALinear(base.gate_proj, rank)
        self.up_proj = LoRALinear(base.up_proj, rank)
        self.down_proj = LoRALinear(base.down_proj, rank)
        self.act_fn = base.act_fn

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def install_lora_mlp_tail_head(model, prefix_layers=6, unique_layers=2,
                               rank=128, trainable=True):
    """Reuse pretrained MLPs and learn only a second-pass low-rank correction."""
    original = list(model.diffusion_head_a)
    if len(original) != 10 or prefix_layers + 2 * unique_layers != 10:
        raise ValueError("Expected ten original visual blocks and two tail passes")
    for i in range(unique_layers):
        original[prefix_layers + unique_layers + i].mlp = LoRAMLP(
            original[prefix_layers + i].mlp, rank)
    model.diffusion_head_a = nn.ModuleList([DtypeBlock(block) for block in original])
    model._elt_architecture = ("lora_mlp", prefix_layers, unique_layers, rank)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for block in original[prefix_layers + unique_layers:]:
            for layer in (block.mlp.gate_proj, block.mlp.up_proj,
                          block.mlp.down_proj):
                layer.a.requires_grad_(True)
                layer.b.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def install_residual_tail_head(model, prefix_layers=6, unique_layers=2,
                               adapter_width=1536, trainable=True):
    original = list(model.diffusion_head_a)
    if len(original) != 10 or prefix_layers + 2 * unique_layers != 10:
        raise ValueError("Expected ten original visual blocks and two tail passes")
    hidden_size = model.diffusion_head_config.hidden_size
    first_pass = [DtypeBlock(block) for block in original[:prefix_layers + unique_layers]]
    second_pass = [ResidualTailBlock(original[prefix_layers + i],
                                     hidden_size, adapter_width).to(
                                         device=next(original[0].parameters()).device,
                                         dtype=next(original[0].parameters()).dtype)
                   for i in range(unique_layers)]
    # Keep the frozen pretrained blocks in BF16; adapters use FP32 for stable
    # optimizer updates, as in the full-weight loop experiments.
    for block in second_pass:
        block.adapter.float()
    model.diffusion_head_a = nn.ModuleList(first_pass + second_pass)
    model.diffusion_head_b = DtypeFinal(model.diffusion_head_b)
    model._elt_architecture = ("residual_tail", prefix_layers,
                               unique_layers, adapter_width)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for block in second_pass:
            for parameter in block.adapter.parameters():
                parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def install_mlp_shared_tail_head(model, prefix_layers=6, unique_layers=2,
                                 trainable=True):
    """Share the costly MLP across passes while retaining layer-specific attention."""
    original = list(model.diffusion_head_a)
    if len(original) != 10 or prefix_layers + 2 * unique_layers != 10:
        raise ValueError("Expected ten original visual blocks and two tail passes")
    for block in original[prefix_layers:]:
        block.to(torch.float32)
    for i in range(unique_layers):
        original[prefix_layers + unique_layers + i].mlp = \
            original[prefix_layers + i].mlp
    model.diffusion_head_a = nn.ModuleList([DtypeBlock(block) for block in original])
    model.diffusion_head_b = DtypeFinal(model.diffusion_head_b.to(torch.float32))
    model._elt_architecture = ("mlp_shared", prefix_layers, unique_layers, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for i in range(unique_layers):
            for parameter in original[prefix_layers + i].mlp.parameters():
                parameter.requires_grad_(True)
        for block in original[prefix_layers + unique_layers:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in model.diffusion_head_b.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def install_elt_head(model, unique_layers=5, loops=2, precision=torch.float32,
                     trainable=True, init="first", prefix_layers=0):
    original = list(model.diffusion_head_a)
    if len(original) != 10 or prefix_layers + unique_layers * loops != 10:
        raise ValueError("Show-o2 teacher requires 10 calls and prefix+N*L=10")
    if prefix_layers and init != "first":
        raise ValueError("Tail loops must preserve their pretrained first pass")
    if init == "first":
        selected = original[prefix_layers:prefix_layers + unique_layers]
    elif init == "last":
        selected = original[unique_layers:]
    elif init == "pair_average":
        selected = original[:unique_layers]
        with torch.no_grad():
            for early, late in zip(selected, original[unique_layers:]):
                for left, right in zip(early.parameters(), late.parameters()):
                    if left.shape != right.shape:
                        raise RuntimeError("Cannot average unmatched head blocks")
                    left.copy_((left.float() + right.float()).mul_(0.5).to(left.dtype))
    else:
        raise ValueError(init)
    shared = [DtypeBlock(block.to(precision)) for block in selected]
    prefix = [DtypeBlock(block) for block in original[:prefix_layers]]
    model.diffusion_head_a = nn.ModuleList(prefix + shared * loops)
    model.diffusion_head_b = DtypeFinal(model.diffusion_head_b.to(precision))
    model._elt_architecture = (prefix_layers, unique_layers, loops)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for block in shared:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in model.diffusion_head_b.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def load_head_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = getattr(model, "_elt_architecture", None)
    found = payload.get("architecture_tuple")
    if expected is not None and found is not None and tuple(found) != expected:
        raise RuntimeError(f"Checkpoint architecture {found} != model {expected}")
    parameters = dict(model.named_parameters())
    for name, value in payload["head"].items():
        if name not in parameters or parameters[name].shape != value.shape:
            raise RuntimeError(f"Checkpoint shape/name mismatch: {name}")
        parameters[name].data.copy_(value.to(parameters[name].device,
                                             dtype=parameters[name].dtype))
    return payload
