# Reciprocal recurrence in a unified AR–flow model

## Empirical starting point

Show-o2 already fuses understanding-oriented and generation-oriented image
embeddings in its autoregressive backbone, then uses a separate flow head for
image synthesis. Our current recurrent intervention changes only that flow
head. Exact agreement of frozen understanding logits is a useful regression
check, but it cannot establish generation-to-understanding transfer. A genuine
reciprocal result requires an intervention in the shared autoregressive path
and controlled evaluation of both capabilities.

The current 50-step, 72-prompt GenEval pilot gives 0.6389 for the original
head, 0.5972 for rank-1280 `CYCLE`, and 0.5833 for rank-1280 `MIDDLE`.
`MIDDLE` has lower held-out velocity MSE (0.01076 versus `CYCLE` 0.01533),
yet a worse GenEval score; its PickScore trails the original by 0.147 on the
same 72 prompt/seed pairs. In particular color attribution falls from 0.50
to 0.167 in this small subset. Therefore velocity imitation cannot be the
main optimization or model-selection criterion.

## Proposed mechanism: semantic feedback controls recurrent visual depth

At a selected flow time, let the hybrid model produce a provisional clean
latent estimate. Decode and re-encode it with Show-o2's own understanding
path, then measure claim-level visual evidence for the prompt and for
perturbed claims (wrong object, color, count, or position). Condition a small
adapter on that evidence and flow time to adjust the next visit of a shared
visual core. Keep resolution transitions and final readout independent.
The original flow prediction remains an anchor to the released model prior.
The model may spend an extra middle-core visit when semantic evidence is weak,
and skip that visit when evidence is strong. Compare the resulting quality
versus wall time, FLOPs, and stored parameters with static recurrence, more
ODE steps, stronger CFG, and multiple-sample reranking.

To permit transfer in the reverse direction, train a small *shared AR adapter*
on grounded image-to-text evidence derived from generated/reconstructed
images. Evaluate its effect on understanding tasks using real held-out images,
not just on the model's own generated images. The flow-core adapters and AR
adapter are separately switchable. This creates four controlled variants:
base, visual recurrence only, AR grounding only, and their joint training.
The claim of reciprocity requires a joint gain that exceeds each one-way
variant under the same data and compute budget.

## Boundary against close prior work

[ELT](https://arxiv.org/html/2604.09168) motivates real block reuse and
compute–quality elasticity. [Looped flow-matching TTS](https://arxiv.org/html/2609.29768)
shows that reuse order and position matter at fixed executed depth.
[LoopVAE](https://arxiv.org/html/2609.11516) demonstrates scale/loop
conditioning within the tokenizer and warns that shared recurrence does not
make early exits reliable. The [fractal diffusion thesis](https://ekmair.ukma.edu.ua/items/929a99dd-a89e-47d1-a7d8-1c7a068ae2f4)
uses diffusion-time/fractal-depth conditioning and independent readout blocks.

[RecA](https://reconstruction-alignment.github.io/) transfers understanding
features to generation by reconstruction. [Transferability Between
Understanding and Generation](https://arxiv.org/html/2607.04423) finds that
the amount of architectural sharing governs cross-task transfer.
[Dreaming in Flow](https://arxiv.org/html/2609.08282) already reports
bidirectional post-training on Show-o2 and improves GenEval; generic
understanding–generation feedback is thus *not* novel. Our proposed research
must show that recurrent depth provides a measurable, quality-directed
feedback substrate beyond the same feedback trained on an unlooped backbone.
[RealUnify](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_RealUnify_Do_Unified_Models_Truly_Benefit_from_Unification_A_Comprehensive_CVPR_2026_paper.html)
tests understanding-to-generation and generation-to-understanding jointly,
and is suitable for the final synergy claim.

## Required evidence

1. Show positive same-step GenEval, independent preference, and image FID
   deltas with comparable diversity; report human blind ratings and latency.
2. Show positive understanding changes on held-out MME, POPE, HallusionBench,
   and both RealUnify directions. Self-consistency on generated images is only
   a diagnostic and is prone to ceiling effects.
3. Demonstrate mediation: semantic feedback changes which loop visits occur,
   the corrections reduce the specifically detected errors, and removing the
   shared AR adapter or visual feedback breaks the joint gain.
4. Run video FVD only with the same UCF-101 class-conditional protocol as the
   comparator. Keep the text-to-image Show-o2 experiment distinct from ELT's
   ImageNet/UCF results.
