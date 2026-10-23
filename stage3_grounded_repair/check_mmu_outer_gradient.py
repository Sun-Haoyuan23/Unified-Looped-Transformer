"""Official-MMU equivalence, single-branch zero identity and actual outer gradients.

No optimizer, quality result, validation or controller. This establishes the
future joint-task interface without changing frozen generation code or evidence.
"""
from pathlib import Path
import json,time,shutil
import torch
from runtime import Runtime,write_json,tensor_hash
from evidence_encoder import EvidenceReader,condition_tensor
from rms_scaled_revisit import make_operator
from mmu_outer_context import question_inputs,mmu_loop_context
from candidate_provenance import file_hash,freeze_config

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'mmu_outer_interface_v43';out.mkdir()
try:
    source=('check_mmu_outer_gradient.py','mmu_outer_context.py','test_mmu_outer_context.py',
            'rms_scaled_revisit.py','outer_shared_revisit.py','runtime.py','evidence_encoder.py')
    path=r/'states/repair_train_targets_v36/train_extra_00021_seed101_i35.pt'
    freeze_config(out/'config.json',dict(source_sha256={n:file_hash(r/n) for n in source},
        state_sha256=file_hash(path),split='train',optimizer_updates=0,
        official_mask_unchanged=True,synthetic_gradient_objective=True,nonzero_head_probe_std=.001))
    (out/'source').mkdir()
    for n in source:shutil.copy2(r/n,out/'source'/n)
    write_json(out/'pipeline_status.json',dict(phase='loading',optimizer_updates=0))
    runtime=Runtime(w,vae=False);base=list(runtime.model.parameters());reader=EvidenceReader(runtime)
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['split']!='train' or tensor_hash(state['latent'])!=state['state_fingerprint']:raise ValueError('Invalid TRAIN input')
    clean=state['latent'][:1].to(runtime.device).float()+(1-state['true_t'])*state['baseline_prediction'][:1].to(runtime.device)
    with torch.no_grad():visual=reader.base._visual_features(clean)
    torch.manual_seed(20261005);op=make_operator(runtime,'outer').outer
    op.condition=condition_tensor(state['evidence'],1.,runtime.device)
    core_hashes={k:tensor_hash(v) for k,v in op.shared_core.state_dict().items()};checks=[]
    for question in (state['evidence']['target_question'],state['evidence']['opposite_question']):
        inputs,span=question_inputs(reader,visual,question)
        with torch.no_grad():
            base_output=runtime.model.showo(**inputs)
            original_margin,original_hidden=reader._question(visual,question)
            logits=base_output['logits'][0,-1].float()
            margin=float(torch.logsumexp(logits[reader.base.yes_ids],0)-torch.logsumexp(logits[reader.base.no_ids],0))
            if margin!=original_margin or not torch.equal(base_output['hidden_states'][-1][0,-1].cpu(),original_hidden):
                raise ValueError('Question construction differs from the official-compatible reader')
            with mmu_loop_context(runtime,op,span):zero_output=runtime.model.showo(**inputs)
            if (not torch.equal(zero_output['logits'],base_output['logits']) or
                    not torch.equal(zero_output['hidden_states'][-1],base_output['hidden_states'][-1])):
                raise ValueError('MMU zero recurrence is not strictly identical')
        checks.append(dict(question=question,span=span,reader_equivalent=True,logits_and_hidden_exact=True))
    with torch.no_grad():op.film.weight.normal_(std=.001)
    started=time.perf_counter()
    with mmu_loop_context(runtime,op,span):
        output=runtime.model.showo(**inputs)
        loss=(output['logits'][0,-1,reader.base.yes_ids].float()-logits[reader.base.yes_ids]-.01).square().mean()
        loss.backward()
    grads=[p.grad for p in op.trainable_parameters() if p.grad is not None]
    memory=sum(float(p.grad.abs().sum()) for p in op.encoder.parameters() if p.grad is not None)
    if memory<=0 or not grads or any(not bool(torch.isfinite(g).all()) for g in grads):raise ValueError('Invalid MMU shared-evidence gradient')
    if any(p.grad is not None for p in base):raise ValueError('Base receives gradient')
    if core_hashes!={k:tensor_hash(v) for k,v in op.shared_core.state_dict().items()}:raise ValueError('Frozen core changed')
    write_json(out/'question_checks.json',checks)
    write_json(out/'pipeline_status.json',dict(phase='passed',optimizer_updates=0,questions=2,
        official_reader_equivalent=True,zero_logits_and_hidden_exact=True,base_parameter_gradients=False,
        shared_cores_exact=True,shared_evidence_gradient_abs_sum=memory,
        adapter_gradient_abs_sum=sum(float(g.abs().sum()) for g in grads),time_s=time.perf_counter()-started,
        peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
        interpretation='Task interface proof; no understanding accuracy/preservation gain or training'))
except Exception as error:
    write_json(out/'pipeline_status.json',dict(phase='failed',error=str(error),optimizer_updates=0));raise
