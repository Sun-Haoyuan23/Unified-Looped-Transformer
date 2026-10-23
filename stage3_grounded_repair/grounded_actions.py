"""Oracle-v2 spatial prompt and regional-velocity TEACHER interfaces.

Public APIs:
  concise_layout_prompt(evidence, opposite=False) -> str
  spatial_field(relation, height, width, device) -> (subject, reference)
  regional_velocity(runtime, x, t, base_velocity, evidence,
                    opposite=False, strength=.5) -> (velocity, 2, 20)

Evidence contains subject/reference object names and one of left of, right of,
above, below. Counterfactual actions keep subject first and swap its intended
side with the reference side. Fields are smooth central sigmoid partitions of
shape [1,1,H,W], each in [0,1], with reference=1-subject. Both fields broadcast
over channels and the duplicated CFG latent batch. Every velocity evaluation
uses Runtime.velocity, including its existing guided conditional/unconditional
path; no separate CFG rule or wrong-axis intervention is introduced.

Regional velocity uses two concise single-object location prompts and two full
network evaluations. It is an OFFLINE teacher repairability probe, not a learned
shared-loop repair action. The parent action record should include
REGIONAL_TEACHER_NOTE and must retain the original prompt for all scoring.
Torch is imported only when constructing a field, so prompt helpers and their
tests run in a stdlib-only environment. This module never downloads weights.
"""

from __future__ import annotations

import math
import re


REGIONAL_TEACHER_NOTE = (
    "Offline regional velocity teacher: two full network evaluations per intervention; "
    "not the core shared-loop method. Score every resulting image against the original prompt."
)

SIDES = {
    "left of": ("left", "right"),
    "right of": ("right", "left"),
    "above": ("top", "bottom"),
    "below": ("bottom", "top"),
}


def _relation(value):
    if not isinstance(value, str):
        raise ValueError("Spatial relation must be a string")
    relation = re.sub(r"\s+", " ", value).strip().casefold()
    if relation not in SIDES:
        raise ValueError(f"Unsupported spatial relation: {value!r}")
    return relation


def _object(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Evidence {label} must be a nonempty object name")
    return re.sub(r"\s+", " ", value).strip()


def _article(name):
    initial = name.casefold()
    # Official object names include airplane, apple, elephant, orange, umbrella;
    # the small exception set also keeps the helper sensible for extra metadata.
    if initial.startswith(("hour", "honest", "honor", "heir")):
        return "an"
    if initial.startswith(("unicorn", "university", "uniform", "unique", "unicycle", "user", "european", "one ")):
        return "a"
    return "an" if initial[0] in "aeiou" else "a"


def _evidence_sides(evidence, opposite=False):
    if not isinstance(evidence, dict):
        raise ValueError("Evidence must be a dictionary")
    if not isinstance(opposite, bool):
        raise ValueError("opposite must be a bool")
    subject = _object(evidence.get("subject"), "subject")
    reference = _object(evidence.get("reference"), "reference")
    relation = _relation(evidence.get("relation"))
    subject_side, reference_side = SIDES[relation]
    if opposite:
        subject_side, reference_side = reference_side, subject_side
    return subject, reference, subject_side, reference_side, relation


def concise_layout_prompt(evidence, opposite=False):
    """Describe two distinct image sides; original scoring prompt stays external."""
    subject, reference, subject_side, reference_side, _ = _evidence_sides(evidence, opposite)
    return (f"a photo of {_article(subject)} {subject} on the {subject_side}, "
            f"and {_article(reference)} {reference} on the {reference_side}.")


def _single_location_prompt(name, side):
    return f"a photo of {_article(name)} {name} on the {side} side of the image."


def spatial_field(relation, height, width, device):
    """Smooth subject support at its target side; reference is its complement."""
    relation = _relation(relation)
    for label, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    import torch
    # Pixel-center coordinates keep a one-pixel axis centered at zero as well.
    xs = (torch.arange(width, device=device, dtype=torch.float32) + .5) / width - .5
    ys = (torch.arange(height, device=device, dtype=torch.float32) + .5) / height - .5
    if relation in ("left of", "right of"):
        direction = -1. if relation == "left of" else 1.
        field = torch.sigmoid(12. * direction * xs)[None, :].expand(height, width)
    else:
        direction = -1. if relation == "above" else 1.
        field = torch.sigmoid(12. * direction * ys)[:, None].expand(height, width)
    subject = field[None, None, :, :]
    return subject, 1. - subject


def regional_velocity(runtime, x, t, base_velocity, evidence, opposite=False, strength=.5):
    """Blend two object-specific guided velocities using complementary fields.

    The third and second return values are ADDED full-network visual-block and
    velocity calls. The baseline evaluation is supplied by the caller and is not
    counted here. The caller controls no_grad / autograd and rollout duration.
    """
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError("strength must be finite and between 0 and 1")
    subject, reference, subject_side, reference_side, relation = _evidence_sides(evidence, opposite)
    effective_relation = ({"left of": "right of", "right of": "left of",
                           "above": "below", "below": "above"}[relation] if opposite else relation)
    support, reference_support = spatial_field(effective_relation, x.shape[-2], x.shape[-1], x.device)
    support = support.to(device=base_velocity.device, dtype=base_velocity.dtype)
    reference_support = reference_support.to(device=base_velocity.device, dtype=base_velocity.dtype)
    subject_velocity = runtime.velocity(x, t, runtime.inputs(_single_location_prompt(subject, subject_side)))
    reference_velocity = runtime.velocity(x, t, runtime.inputs(_single_location_prompt(reference, reference_side)))
    if subject_velocity.shape != base_velocity.shape or reference_velocity.shape != base_velocity.shape:
        raise RuntimeError("Regional teacher velocities must match the baseline latent shape")
    combined = support * subject_velocity + reference_support * reference_velocity
    velocity = (1. - strength) * base_velocity + strength * combined
    return velocity, 2, 20
