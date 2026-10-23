"""Strict TRAIN corpus: v50 residual-interface teachers plus v36 true correct states."""
import json
import math
from pathlib import Path

import torch
from candidate_provenance import file_hash
from runtime import tensor_hash
from state_space_revisit import capture_features
from training_pairs import selected_pairs, preservation_states, validate_training_pairs


class ProcessCorpus:
    def __init__(self, root):
        self.root=Path(root)
        canonical=self.root/'repair_train_targets_v36'
        report=json.loads((canonical/'paired_report.json').read_text())
        records=[json.loads(s) for s in (canonical/'all_actions.jsonl').read_text().splitlines()]
        self.gate=validate_training_pairs(report,records)
        self.original_teachers=selected_pairs(report,records)
        self.correct=preservation_states(report,records)
        self.baselines={q['state_id']:q for q in records if q['action']=='baseline'}
        if self.gate['selected_teacher_states']!=3 or self.gate['preservation_states']!=35 or len(self.baselines)!=112:
            raise ValueError('Frozen canonical TRAIN cohort changed')
        process=self.root/'teacher_process_targets_v49'
        reach=self.root/'teacher_additive_reachability_v50'
        for directory in (process,reach):
            if json.loads((directory/'pipeline_status.json').read_text())['phase']!='complete':
                raise ValueError('Process reconstruction/reachability incomplete')
        self.process={q['state_id']:q for q in map(json.loads,(process/'process_manifest.jsonl').read_text().splitlines())}
        reach_records=[json.loads(s) for s in (reach/'all_actions.jsonl').read_text().splitlines()]
        self.teachers={q['state_id']:q for q in reach_records if q['action']=='additive_teacher'}
        reach_report=json.loads((reach/'paired_report.json').read_text())
        reach_pairs={q['state_id']:q for q in reach_report['pairs'] if q['action']=='additive_teacher'}
        original_pairs={q['state_id']:q for q in report['pairs'] if q['action']=='training_target'}
        if set(self.teachers)!=set(self.original_teachers) or set(reach_pairs)!=set(self.teachers):
            raise ValueError('Aligned teacher set differs from three qualified original TRAIN states')
        self.original_baseline_deltas={}
        for sid,row in self.teachers.items():
            original=self.original_teachers[sid]
            for key in ('split','prompt','prompt_id','seed','flow_index','true_t','state_fingerprint'):
                if row[key]!=original[key]:raise ValueError('Aligned teacher identity mismatch')
            pair=reach_pairs[sid]
            if row['split']!='train' or pair['baseline_correct']!=0 or pair['candidate_correct']!=1:
                raise ValueError('Aligned TRAIN teacher is not a semantic repair')
            deltas={}
            for metric in ('hps','pickscore'):
                base=original_pairs[sid]['baseline_metrics'][metric]
                value=pair['candidate_metrics'][metric]
                if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (base,value)):
                    raise ValueError('Missing/nonfinite original-baseline quality')
                deltas[metric]=value-base
                if deltas[metric]<0:raise ValueError('Aligned teacher quality lost against original baseline')
            self.original_baseline_deltas[sid]=deltas
            if file_hash(row['final_latent'])!=row['final_latent_sha256']:
                raise ValueError('Aligned teacher bytes changed')
            if file_hash(self.process[sid]['artifact_path'])!=self.process[sid]['artifact_sha256']:
                raise ValueError('Frozen process artifact bytes changed')
        paths=(canonical/'paired_report.json',canonical/'all_actions.jsonl',process/'config.json',
            process/'process_manifest.jsonl',reach/'paired_report.json',reach/'all_actions.jsonl',reach/'scores.jsonl')
        self.input_hashes={str(p.relative_to(self.root)):file_hash(p) for p in paths}

    def state(self,sid):
        if sid not in self.teachers and sid not in self.correct:
            raise ValueError('State is not a qualified TRAIN teacher or true correct preservation state')
        path=self.root/'states/repair_train_targets_v36'/(sid+'.pt')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if (state['state_id']!=sid or state['split']!='train' or state['flow_index']!=35 or
                not state['zero_intervention_test']['passed'] or
                tensor_hash(state['latent'])!=state['state_fingerprint'] or
                state['state_fingerprint']!=self.baselines[sid]['state_fingerprint']):
            raise ValueError('TRAIN state corruption or split leakage')
        return state

    def target(self,runtime,state,features,operator,preservation):
        sid=state['state_id']
        original=state['latent'].to(runtime.device).float()
        if preservation:
            if sid not in self.correct:raise ValueError('Preservation requires actual baseline-correct TRAIN state')
            return dict(input=original,final=state['baseline_final_latent'].to(runtime.device).float(),
                features={n:features[n][0] for n in ('outer','inner') if n in features})
        if sid not in self.teachers:raise ValueError('Missing strict aligned teacher')
        row=self.process[sid]
        payload=torch.load(row['artifact_path'],map_location='cpu',weights_only=False)
        if (payload['state_id']!=sid or payload['split']!='train' or
                payload['source_state_fingerprint']!=state['state_fingerprint']):
            raise ValueError('Process payload identity mismatch')
        target_input=original+payload['physical_delta'].to(runtime.device).float()
        target_state=dict(state,latent=target_input.detach().cpu(),state_fingerprint=tensor_hash(target_input))
        with torch.no_grad():
            target_features,_=capture_features(runtime,target_state,operator)
        teacher_row=self.teachers[sid]
        final=torch.load(teacher_row['final_latent'],map_location=runtime.device,weights_only=False)
        if (final['state_id']!=sid or final['action']!='additive_teacher' or
                not torch.equal(final['latent'].cpu(),payload['additive_final'])):
            raise ValueError('Aligned endpoint does not correspond to process physical delta')
        return dict(input=target_input.detach(),final=final['latent'].float().detach(),
            features={n:target_features[n][0].detach() for n in ('outer','inner') if n in target_features})
