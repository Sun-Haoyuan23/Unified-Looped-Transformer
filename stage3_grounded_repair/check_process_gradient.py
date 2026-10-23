"""Actual nested terminal/process gradient proof; no optimizer or trained checkpoint."""
from pathlib import Path
import json
import time
import torch

from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from process_supervision import recurrent_trace,frozen_feature_normalizers,process_terms
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator,nested_loss
from runtime import Runtime,write_json
from state_space_revisit import full_model_context,capture_features
from trainable_recurrent_state import replay_delta

work=Path('/home/shy_23/iterative_visual_stage1_20261003')
root=work/'stage3_grounded_repair'
out=root/'process_shared_gradient_v51'


def main():
    out.mkdir()
    def status(phase,**kw):
        write_json(out/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,**kw))
    try:
        corpus=ProcessCorpus(root)
        names=('check_process_gradient.py','process_supervision.py','test_process_supervision.py',
            'process_training_targets.py','recurrent_training_protocol.py','trainable_recurrent_state.py',
            'state_space_revisit.py','runtime.py','evidence_encoder.py','training_pairs.py','candidate_provenance.py')
        freeze_config(out/'config.json',dict(source_sha256={n:file_hash(root/n) for n in names},
            corpus_input_sha256=corpus.input_hashes,original_baseline_quality_deltas=corpus.original_baseline_deltas,
            training_gate=corpus.gate,probe_readout_std=.001,optimizer_updates=0,
            feature_normalizers='per state/branch/prefix fixed from zero-initialized operator',
            zero_proof_precedes_nonzero_readout_probe=True))
        (out/'source').mkdir()
        for n in names:(out/'source'/n).write_bytes((root/n).read_bytes())
        status('loading')
        runtime=Runtime(work,vae=False)
        # Deterministic second teacher: local minimum geometry, no scene mirror.
        sid=sorted(corpus.teachers)[1]
        state=corpus.state(sid)
        torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        op=build_operator(runtime,'joint_ilsd')
        op.condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
        initial={k:__import__('runtime').tensor_hash(v) for k,v in op.adapter_state().items()}
        base=list(runtime.model.parameters());core_hash=op.core_hashes()
        baseline=state['baseline_final_latent'].to(runtime.device).float()
        original=state['latent'].to(runtime.device).float()
        with full_model_context(runtime):
            features,options=capture_features(runtime,state,op)
            target=corpus.target(runtime,state,features,op,False)
            positions=options['modality_positions']
            with torch.no_grad(),recurrent_trace(op) as trace:
                zero=op.latent_deltas(runtime,features,positions,original,(1,2))
            scales=frozen_feature_normalizers(trace,target['features'],positions)
            with torch.no_grad():
                for budget in (1,2):
                    final=replay_delta(runtime,state,zero[budget])
                    if not torch.equal(final,baseline):raise ValueError('Untrained zero terminal not exact')
            # At untouched zero readout the auxiliary feature objective must
            # already train the shared core, rather than relying on its head.
            with recurrent_trace(op) as trace:
                deltas=op.latent_deltas(runtime,features,positions,original,(1,2))
            parts=process_terms(deltas,trace,original,target['input'],target['features'],positions,False,scales)
            parts['hidden'].backward()
            initial_core_grad={name:sum(float(p.grad.abs().sum()) for p in branch.low_rank.parameters() if p.grad is not None)
                for name,branch in (('outer',op.outer),('inner',op.inner))}
            if min(initial_core_grad.values())<=0:raise ValueError('Auxiliary loss does not train both initially zero-readout cores')
            op.zero_grad(set_to_none=True)
            # Explicit probe, not optimizer training; as with v46, a nonzero
            # readout exposes evidence gradients before the first optimizer step.
            with torch.no_grad():
                for branch in (op.outer,op.inner):branch.readout.weight.normal_(std=.001)
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            status('full_nested_process_gradient')
            with recurrent_trace(op) as trace:
                deltas=op.latent_deltas(runtime,features,positions,original,(1,2))
            terms=process_terms(deltas,trace,original,target['input'],target['features'],positions,False,scales)
            high=replay_delta(runtime,state,deltas[2],checkpoint_rollout=True)
            low=replay_delta(runtime,state,deltas[1],checkpoint_rollout=True)
            terminal,detail=nested_loss(high,low,target['final'],baseline,False,1,100)
            loss=terminal+terms['state']+terms['hidden']
            loss.backward()
            gradients={name:sum(float(p.grad.abs().sum()) for p in branch.low_rank.parameters() if p.grad is not None)
                for name,branch in (('outer',op.outer),('inner',op.inner))}
            evidence=sum(float(p.grad.abs().sum()) for p in op.encoder.parameters() if p.grad is not None)
            if min(gradients.values())<=0 or evidence<=0:raise ValueError('Missing core/evidence gradient')
            if any(p.grad is not None for p in base):raise ValueError('Frozen base receives gradients')
            if any(not bool(torch.isfinite(p.grad).all()) for p in op.trainable_parameters() if p.grad is not None):
                raise ValueError('Nonfinite process gradients')
        if op.core_hashes()!=core_hash:raise ValueError('Shared base core changed')
        status('passed',state_id=sid,zero_terminal_budgets_exact=[1,2],
            initially_zero_readout_aux_core_gradient_abs_sum=initial_core_grad,
            full_terminal_core_gradient_abs_sum=gradients,shared_evidence_gradient_abs_sum=evidence,
            base_parameter_gradients=False,shared_core_hash_unchanged=True,
            full_rollouts_per_backward=2,remaining_full_velocities_per_rollout=14,
            feature_loss=float(terms['hidden'].detach()),state_loss=float(terms['state'].detach()),
            terminal_loss=float(terminal.detach()),time_s=time.perf_counter()-start,
            peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
            added_parameters=0,initial_adapter_tensor_sha256=initial,
            limitation='Separate nonzero-head gradient probe; no optimizer or quality result')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
