"""Causal check-then-rewind interface for existing saved trajectory states.

Retains the earlier latent/prefix and imports ONLY the later intermediate MMU
evidence. No later baseline_final_latent/image/outcome is read or imported.
This utility establishes provenance and extra-cost accounting; it does not
demonstrate a repair, and it is not currently a trained controller.
"""
import torch
from runtime import tensor_hash


def checked_then_rewound(earlier,checked):
    for key in ('prompt_id','prompt','seed','split','CFG','prefix_cache_version'):
        if earlier[key]!=checked[key]:raise ValueError(f'Rollback {key} mismatch')
    if earlier['split'] not in ('architecture_dev','train','val','test'):
        raise ValueError('Unknown prompt split')
    if not torch.equal(earlier['time_grid'],checked['time_grid']):raise ValueError('Rollback solver grid mismatch')
    if earlier['runtime']['base_model_hash']!=checked['runtime']['base_model_hash']:
        raise ValueError('Rollback model mismatch')
    if checked['flow_index']<=earlier['flow_index']:raise ValueError('Check must be strictly later than rewind')
    for state in (earlier,checked):
        if not state['zero_intervention_test']['passed']:raise ValueError('Unmatched captured trajectory')
        if tensor_hash(state['latent'])!=state['state_fingerprint']:raise ValueError('Corrupt rollback latent')
        if tensor_hash(state['prefix_cache'])!=state['prefix_cache_hash']:raise ValueError('Corrupt rollback prefix')
        expected=float(state['time_grid'][state['flow_index']])
        if abs(state['true_t']-expected)>1e-6:raise ValueError('Inconsistent flow time')
    evidence=checked['evidence']
    if any(evidence[key]!=earlier['evidence'][key] for key in ('subject','reference','relation')):
        raise ValueError('Rollback evidence objects/relation differ')
    result=dict(earlier)
    result['evidence']=dict(evidence)
    # The checking velocity itself was computed to obtain a predicted-clean
    # image. All these discarded evaluations must be counted at deployment.
    discarded=checked['flow_index']-earlier['flow_index']+1
    result['rollback']={'checked_state_id':checked['state_id'],'checked_state_fingerprint':checked['state_fingerprint'],
        'checked_flow_index':checked['flow_index'],'checked_true_t':checked['true_t'],
        'rewind_state_id':earlier['state_id'],'rewind_flow_index':earlier['flow_index'],
        'rewind_latent_unchanged':True,'later_terminal_outcome_used':False,
        'extra_discarded_velocity_calls_vs_no_rewind':discarded,
        'extra_discarded_visual_block_calls_vs_no_rewind':discarded*10,
        'checking_mmu_queries':2,'FLOPs':None,
        'cost_note':'Additional discarded forward calls; add actual repair, continuation, observer and VAE costs. Prefix reuse changes visual-block accounting separately.'}
    return result
