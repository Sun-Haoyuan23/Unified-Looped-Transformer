"""Read-only TRAIN opposite-relation and evidence-direction separation audit."""
import json,os,shutil,subprocess,time
from pathlib import Path
import torch
from torch.nn import functional as F
from analytic_readout_operator import load_adapter
from candidate_provenance import file_hash,freeze_config
from evidence_encoder import condition_tensor
from launch_grounding_diagnostic import GPUS
from nonlinear_token_readout import NonlinearTokenReadout,joint_token_features,patch_delta
from process_training_targets import ProcessCorpus
from recurrent_training_protocol import build_operator
from runtime import Runtime,tensor_hash,write_json
from state_space_revisit import capture_features,full_model_context

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003');ROOT=WORK/'stage3_grounded_repair'
OUT=ROOT/'condition_separation_audit_v61'
OPPOSITE={'left of':'right of','right of':'left of','above':'below','below':'above'}
MODES=('trained_loop2','trained_no_loop')


def flip_evidence(evidence):
    result=dict(evidence)
    for a,b in (('target_margin','opposite_margin'),('target_hidden','opposite_hidden'),('target_question','opposite_question')):
        result[a],result[b]=evidence[b],evidence[a]
    result['evidence']=-evidence['evidence']
    result['relation']=OPPOSITE[evidence['relation']]
    return result


def distance(a,b):
    a=a.detach().double().flatten();b=b.detach().double().flatten();difference=a-b
    return dict(rms_difference=float(difference.square().mean().sqrt()),
        relative_rms=float(difference.square().mean().sqrt()/((a.square().mean()+b.square().mean())*.5).sqrt().clamp_min(1e-12)),
        cosine=float(F.cosine_similarity(a,b,dim=0)),strict_equal=torch.equal(a,b))


def main():
    OUT.mkdir(exist_ok=False)
    def status(phase,**kw):write_json(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),
        optimizer_updates=0,new_scoring_images=0,analytical_fits=0,controller_started=False,**kw))
    try:
        q=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if q[0].strip()!=GPUS[0] or int(q[1])>=2000 or os.environ['CUDA_VISIBLE_DEVICES']!=GPUS[0]:raise ValueError('GPU0 occupied/UUID mismatch')
        prior=ROOT/'nonlinear_readout_capacity_v60'
        if json.loads((prior/'pipeline_status.json').read_text())['phase']!='complete':raise ValueError('v60 incomplete')
        corpus=ProcessCorpus(ROOT);all_ids=sorted(set(corpus.teachers)|set(corpus.correct))
        states={sid:corpus.state(sid) for sid in all_ids};pairs=[];controls=[]
        for teacher in sorted(corpus.teachers):
            a=states[teacher];ea=a['evidence']
            for correct in sorted(corpus.correct):
                b=states[correct];eb=b['evidence']
                if (ea['subject'],ea['reference'])==(eb['subject'],eb['reference']):
                    if a['seed']==b['seed'] and OPPOSITE[ea['relation']]==eb['relation']:
                        pairs.append(dict(teacher=teacher,preservation=correct,kind='same_objects_seed_opposite_relation'))
                    elif a['prompt']==b['prompt'] and a['seed']!=b['seed']:
                        controls.append(dict(teacher=teacher,preservation=correct,kind='same_prompt_other_seed'))
        ids=sorted(set(corpus.teachers)|{p['preservation'] for p in pairs+controls})
        if len(ids)!=6 or len(pairs)!=2 or len(controls)!=1:raise ValueError('Frozen metadata-selected diagnostic cohort changed')
        checkpoint=ROOT/'process_recurrent_train_v52/joint_process/operator_step100.pt'
        previous=json.loads((prior/'config.json').read_text())['configuration']
        if file_hash(checkpoint)!=previous['checkpoint_sha256']:raise ValueError('Prior operator changed')
        head_paths={mode:prior/mode/'head_step1000.pt' for mode in MODES}
        head_hashes={mode:file_hash(p) for mode,p in head_paths.items()}
        names=('audit_condition_separation.py','nonlinear_token_readout.py','process_training_targets.py',
            'recurrent_training_protocol.py','trainable_recurrent_state.py','state_space_revisit.py',
            'runtime.py','candidate_provenance.py','evidence_encoder.py','analytic_readout_operator.py','training_pairs.py')
        freeze_config(OUT/'config.json',dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            checkpoint_sha256=file_hash(checkpoint),head_checkpoint_sha256=head_hashes,v60_config_sha256=file_hash(prior/'config.json'),
            state_ids=ids,state_sha256={sid:file_hash(ROOT/'states/repair_train_targets_v36'/(sid+'.pt')) for sid in ids},
            pairs=pairs,controls=controls,selection='all3 strict TRAIN teachers plus every same-object opposite-relation same-seed correct partner and same-prompt other-seed correct control',
            perturbation='swap target/opposite MMU evidence on same original latent and original prompt; not a reconditioned prompt',
            physical_state_never_modified=True,no_flow_continuation=True,
            reference_mse_tolerance=dict(rtol=1e-4,atol=1e-6),FLOPs=None))
        (OUT/'source').mkdir()
        for n in names:shutil.copy2(ROOT/n,OUT/'source'/n)
        status('loading',states=6)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        runtime=Runtime(WORK,vae=False);torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
        op=build_operator(runtime,'joint_ilsd');load_adapter(op,torch.load(checkpoint,map_location='cpu',weights_only=False)['operator'])
        op.requires_grad_(False);base={k:tensor_hash(v) for k,v in runtime.model.named_parameters()}
        adapter={k:tensor_hash(v) for k,v in op.adapter_state().items()};heads={};head_before={}
        for mode,p in head_paths.items():
            payload=torch.load(p,map_location=runtime.device,weights_only=False)
            if payload['step']!=1000 or payload['mode']!=mode:raise ValueError('Fixed head metadata changed')
            head=NonlinearTokenReadout(payload['head']['input_rms'],64,128).to(runtime.device)
            head.load_state_dict(payload['head']);head.requires_grad_(False);heads[mode]=head
            head_before[mode]={k:tensor_hash(v) for k,v in head.state_dict().items()}
        original_results=json.loads((prior/'results.json').read_text())
        reference={mode:{r['state_id']:r for r in original_results[mode]['states']} for mode in MODES}
        artifacts={};records=[]
        with torch.no_grad(),full_model_context(runtime):
            for sid in ids:
                status('auditing_fixed_state',state_id=sid)
                state=states[sid];preserve=sid in corpus.correct
                original_condition=condition_tensor(state['evidence'],state['true_t'],runtime.device)
                reverse_condition=condition_tensor(flip_evidence(state['evidence']),state['true_t'],runtime.device)
                if not torch.equal(condition_tensor(flip_evidence(flip_evidence(state['evidence'])),state['true_t'],runtime.device),original_condition):
                    raise ValueError('Evidence flip involution failed')
                op.condition=original_condition;features,options=capture_features(runtime,state,op)
                target=corpus.target(runtime,state,features,op,preserve)
                original=state['latent'].to(runtime.device).float();gold=(target['input']-original)[:1]
                denominator=float((original.square().mean() if preserve else gold.square().mean()).clamp_min(1e-5))
                item=dict(latent=original.cpu(),condition=original_condition.cpu(),
                    encoded_condition=op.encoder(original_condition).cpu(),hidden={n:features[n][0].cpu() for n in ('outer','inner')},tokens={},updates={})
                for mode,head in heads.items():
                    op.condition=original_condition
                    tokens,calls=joint_token_features(op,features,options['modality_positions'],mode.split('_',1)[1])
                    patches=head(tokens);delta=patch_delta(runtime,patches,original)[:1]
                    mse=float((delta-gold).square().mean()/denominator)
                    expected=reference[mode][sid]['physical_relative_target_mse']
                    if abs(mse-expected)>1e-6+1e-4*abs(expected):raise ValueError('Fixed v60 readout reproduction failed')
                    op.condition=reverse_condition
                    reversed_tokens,reversed_calls=joint_token_features(op,features,options['modality_positions'],mode.split('_',1)[1])
                    reversed_delta=patch_delta(runtime,head(reversed_tokens),original)[:1]
                    if calls!=reversed_calls:raise ValueError('Evidence change altered recurrent budget')
                    records.append(dict(state_id=sid,prompt=state['prompt'],preservation=preserve,mode=mode,
                        target_margin=state['evidence']['target_margin'],opposite_margin=state['evidence']['opposite_margin'],
                        reference_mse=expected,reproduced_mse=mse,absolute_reference_error=abs(mse-expected),
                        raw_condition_flip_distance=distance(original_condition,reverse_condition),
                        token_flip_distance=distance(tokens,reversed_tokens),update_flip_distance=distance(delta,reversed_delta),
                        original_update_rms=float(delta.square().mean().sqrt()),actual_shared_calls=calls))
                    item['tokens'][mode]=(tokens/head.input_rms).cpu();item['updates'][mode]=delta.cpu()
                artifacts[sid]=item
        pair_results=[]
        for p in pairs+controls:
            a,b=(artifacts[p[key]] for key in ('teacher','preservation'))
            row=dict(p,latent_distance=distance(a['latent'],b['latent']),condition_distance=distance(a['condition'],b['condition']),
                encoded_condition_distance=distance(a['encoded_condition'],b['encoded_condition']),
                original_hidden_distance={n:distance(a['hidden'][n],b['hidden'][n]) for n in ('outer','inner')},
                normalized_token_distance={mode:distance(a['tokens'][mode],b['tokens'][mode]) for mode in MODES},
                update_distance={mode:distance(a['updates'][mode],b['updates'][mode]) for mode in MODES})
            pair_results.append(row)
        write_json(OUT/'state_results.json',records);write_json(OUT/'pair_results.json',pair_results)
        if base!={k:tensor_hash(v) for k,v in runtime.model.named_parameters()} or adapter!={k:tensor_hash(v) for k,v in op.adapter_state().items()}:
            raise ValueError('Frozen model or adapter changed')
        for mode,head in heads.items():
            if head_before[mode]!={k:tensor_hash(v) for k,v in head.state_dict().items()} or file_hash(head_paths[mode])!=head_hashes[mode]:
                raise ValueError('Frozen head changed')
        if any(p.grad is not None for p in runtime.model.parameters()) or any(p.grad is not None for p in op.parameters()):raise ValueError('Unexpected gradient')
        status('complete',states=len(ids),pairs=len(pairs),controls=len(controls),fixed_mse_reproductions=len(records),
            original_base_adapter_and_heads_hashes_unchanged=True,no_gradients=True,
            no_new_quality_or_semantic_measurement=True,scope='TRAIN feature separation and evidence sensitivity, not causal generalization or a changed prompt rollout')
    except Exception as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
