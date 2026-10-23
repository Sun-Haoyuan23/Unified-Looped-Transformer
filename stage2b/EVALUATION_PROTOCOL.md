# Quality-first evaluation protocol

## Gate 1: same-model paired pilot

Use identical prompt lists, seed lists, guidance, output resolution, VAE, and
flow step count for the original Show-o2 head and each recurrent layout.
Record stored and trainable parameters, executed Transformer calls, wall time,
peak GPU memory, and estimated FLOPs. Compare `CYCLE`, `SEQUENCE`, and `MIDDLE`
with 50 steps. A 60-step original-head control measures whether extra flow
integration alone explains a quality gain.

The first 72 GenEval prompts contain 12 prompts from each official category and
one image per prompt. Run the official object detector and report paired prompt
bootstrap intervals with category results. Use PickScore for prompt-conditioned
preference, SigLIP for alignment, and blinded human pairwise ratings for actual
visual preference. For diversity, generate at least four independent seeds on a
held-out prompt subset and report within-prompt feature dispersion together
with a quality threshold, since diversity alone can reward defective images.
Do not claim significance from a favorable 14-image visual pilot.

## Gate 2: larger image quality study

If the same-step pilot improves GenEval and independent visual preference, run
the full official 553-prompt GenEval protocol with four seeds, plus held-out
prompts for quality and diversity. For a distributional image metric, sample
enough images from a fixed, documented dataset (for example COCO val2017 with
one sample per caption) and compute FID using the same real-image preprocessing,
feature extractor, sample count, seeds, and resolution for both Show-o2 variants.
Report confidence from repeated sampling sets when feasible. The 72-image
GenEval subset is too small for a credible FID estimate.

## ELT comparison and video track

ELT reports class-conditional ImageNet-256 FID 2.0 and class-conditional
UCF-101 FVD 72.8. Those are reference points, not scores directly comparable
to a 432x432 text-to-image Show-o2 experiment. A direct FID comparison needs
the same ImageNet split, class prompts/conditioning, 256x256 resolution,
generation count, FID implementation, and guidance protocol. A direct FVD
comparison needs a video-capable hybrid checkpoint, the same UCF-101 split,
class conditioning, generated clip length/frame rate/resolution, I3D feature
extractor, and number of generated clips. Report FVD with frame-level FID and
temporal consistency diagnostics; a good image FID can coexist with poor
motion. Both comparisons also report model size, FLOPs and sampling latency.

Keep tokenizer reconstruction FID (rFID, as in LoopVAE) separate from generated
image FID. Keep image FID separate from video FVD. Never compare the numeric
values across these different metrics.
