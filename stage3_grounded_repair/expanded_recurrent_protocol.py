"""Common fresh-head initialization and frozen expanded TRAIN cohort."""
import json
from pathlib import Path
import torch
from analytic_readout_operator import load_adapter
from candidate_provenance import file_hash
from expanded_process_corpus import ExpandedProcessCorpus
from nonlinear_recurrent_repair import NonlinearRecurrentRepair
from nonlinear_token_readout import NonlinearTokenReadout
from recurrent_training_protocol import build_operator
from runtime import tensor_hash

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003');ROOT=WORK/'stage3_grounded_repair'
NAMES=('expanded_process_corpus.py','expanded_recurrent_protocol.py','check_expanded_recurrent_gradient.py',
       'run_expanded_recurrent_gradient.py','train_expanded_recurrent_operator.py','run_expanded_recurrent_training.py',
       'test_expanded_process_corpus.py','nonlinear_recurrent_repair.py','nonlinear_token_readout.py',
       'nonlinear_recurrent_protocol.py','process_training_targets.py','recurrent_training_protocol.py',
       'trainable_recurrent_state.py','state_space_revisit.py','runtime.py','candidate_provenance.py',
       'evidence_encoder.py','training_pairs.py','analytic_readout_operator.py')


def frozen_corpus():
    corpus=ExpandedProcessCorpus(ROOT)
    directory=ROOT/'expanded_process_corpus_v70'
    if json.loads((directory/'pipeline_status.json').read_text())['phase']!='complete':
        raise ValueError('Expanded CPU corpus incomplete')
    config=json.loads((directory/'config.json').read_text())['configuration']
    if corpus.input_hashes!=config['input_hashes'] or corpus.ledger()!=config['states']:
        raise ValueError('Expanded frozen corpus changed')
    checkpoint=ROOT/'process_recurrent_train_v52/joint_process/operator_step100.pt'
    old=json.loads((ROOT/'optimizer_separation_v62/config.json').read_text())['configuration']
    if file_hash(checkpoint)!=old['checkpoint_sha256']:
        raise ValueError('Fixed shared operator checkpoint changed')
    return corpus,checkpoint


def make_repair(runtime,arm,checkpoint,input_rms):
    torch.manual_seed(20261008);torch.cuda.manual_seed_all(20261008)
    op=build_operator(runtime,'joint_ilsd')
    load_adapter(op,torch.load(checkpoint,map_location='cpu',weights_only=False)['operator'])
    head=NonlinearTokenReadout(input_rms,64,128).to(runtime.device)
    return NonlinearRecurrentRepair(op,head,arm)


def target_on_device(corpus,state,device):
    target=corpus.tensors(state)
    return {k:(v.to(device).float().detach() if isinstance(v,torch.Tensor) else v) for k,v in target.items()}


def proof_inputs():
    corpus,checkpoint=frozen_corpus()
    directory=ROOT/'expanded_recurrent_gradient_v71'
    proof=json.loads((directory/'pipeline_status.json').read_text())
    if (proof['phase']!='passed' or not proof['base_hash_unchanged'] or proof['base_gradients']
            or proof['optimizer_updates']!=0 or not proof['all_three_initializations_equal']):
        raise ValueError('Actual expanded model interface proof missing')
    config=json.loads((directory/'config.json').read_text())['configuration']
    for name,digest in config['source_sha256'].items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Proven source changed: '+name)
    if file_hash(ROOT/'expanded_process_corpus_v70/config.json')!=config['corpus_config_sha256']:
        raise ValueError('Proven corpus changed')
    rms_path=directory/'common_input_rms.pt'
    if file_hash(rms_path)!=proof['common_input_rms_file_sha256']:
        raise ValueError('Common normalizer changed')
    rms=torch.load(rms_path,map_location='cpu',weights_only=False)['input_rms']
    if tensor_hash(rms)!=proof['common_input_rms_tensor_sha256']:
        raise ValueError('Normalizer tensor changed')
    return corpus,checkpoint,rms,proof
