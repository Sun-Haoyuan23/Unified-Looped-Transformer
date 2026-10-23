# v73: matched expanded recurrent mapping audit

Frozen before execution. This is a read-only diagnostic of fixed v72 step420, not training or quality evaluation.

Use all three completed v72 arms, all six strictly qualified TRAIN teachers and all35 real baseline-correct TRAIN states from v70. No state selection from this audit, no optimizer, analytical fitting, image decoding, evaluator, val or controller. Preserve v72 checkpoints and all old source/data.

Reproduce all123 normalized physical MSE values at rtol1e-4/atol1e-6. For every state measure update/target/residual RMS and spatial residual concentration. For every teacher also measure update direction cosine and the unconstrained optimal scalar residual lower bound; this bound is a diagnostic and will not be deployed or scanned.

For all41 states compute physical-loss gradients. Report six individual teacher gradients and the mean35 preservation physical gradient, split into common trainable parameters and shared-core low-rank parameters. For all six teachers in each arm run the complete14 remaining original velocities and compute terminal-loss gradients and normalized terminal error against the already scored v50/v69 additive targets. Thus18 terminal diagnostics; no new semantic/quality claim. Physical/terminal gradient angles and cross-state Gram matrices are local diagnostics, not causal attribution. Preservation terminal gradients are not measured and must not be represented by physical gradients.

On the same teacher latent and unchanged original prompt, exchange target/opposite MMU evidence and measure physical predictions only. No flow continuation or quality claim for reversed evidence. Compare effective BF16 q/k/v/o weights at v72 step420 with the exact common v52 operator warm start. Check base and adapter hashes before/after, checkpoint/source hashes and no parameter .grad buffers.

Expected nominal diagnostic velocities:123 original feature captures +18x14 terminal continuations =375. Shared recurrent calls, reverse-evidence calls, checkpoint backward recomputation, model loading/hash checks and walltime recorded separately; FLOPs null. All physical state changes disable old KV/inner prefix and recompute full14 velocities. Route forward reads only original mid-state information and frozen parameters; teacher tensors enter loss/diagnostic statistics only.

No capacity gate is changed: each6 teacher physical MSE<.01 and max35 preservation<.001. v72 all arms failed, so no automatic terminal image evaluation, val or controller. The next training interface must be justified by the actual audit results, not by amplitude/time/threshold scanning.
