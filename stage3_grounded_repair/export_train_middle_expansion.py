"""Decode predicted-clean TRAIN states; no final outcomes enter this stage."""
import argparse
import json
from pathlib import Path
import torch
from runtime import Runtime,tensor_hash
from candidate_provenance import file_hash,freeze_config

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
p.add_argument('--pilot',type=Path,required=True);p.add_argument('--states',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-states',type=int,required=True);args=p.parse_args()
root=args.work/'stage3_grounded_repair';out=args.output;out.mkdir(exist_ok=True)
pilot=args.pilot
if not json.loads((pilot/'capture_status.json').read_text())['success']:raise ValueError('Train capture incomplete')
tasks={r['prompt_id']:r for r in json.loads((pilot/'train_prompts.json').read_text())}
paths=sorted(args.states.glob('shard*/*_i35.pt'))
if len(paths)!=args.expected_states or sum(len(q['seeds']) for q in tasks.values())!=args.expected_states:raise ValueError('Unexpected frozen TRAIN count')
if (out/'clean_records.jsonl').exists():raise ValueError('Already exported; do not overwrite')
freeze_config(out/'config.json',dict(source_sha256=file_hash(__file__),runtime_sha256=file_hash(root/'runtime.py'),
    pilot_prompts_sha256=file_hash(pilot/'train_prompts.json'),input_state_sha256={str(q):file_hash(q) for q in paths},
    split='train',states=args.expected_states,flow_index=35,terminal_outcomes_used=False,optimizer_steps=0))
r=Runtime(args.work)
seen=set()
with (out/'clean_records.jsonl').open('w') as log:
    for path in paths:
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['split']!='train' or state['flow_index']!=35 or state['prompt']!=tasks[state['prompt_id']]['prompt']:
            raise ValueError('Unexpected train prompt/split/time')
        if state['seed'] not in tasks[state['prompt_id']]['seeds'] or state['state_id'] in seen:
            raise ValueError('Duplicate or unexpected train seed')
        if tensor_hash(state['latent'])!=state['state_fingerprint'] or not state['zero_intervention_test']['passed']:
            raise ValueError('Corrupt/unverified input state')
        seen.add(state['state_id'])
        clean=state['latent'][:1].to(r.device).float()+(1-state['true_t'])*state['baseline_prediction'][:1].to(r.device)
        directory=out/'clean_images'/state['state_id'];(directory/'samples').mkdir(parents=True)
        image=directory/'samples/0000.png'
        with torch.no_grad():r.decode(clean).save(image)
        (directory/'metadata.jsonl').write_text(json.dumps(state['geneval_metadata'])+'\n')
        row={k:state[k] for k in ('state_id','prompt_id','prompt','seed','split','flow_index','true_t','state_fingerprint')}
        row.update(clean_image=str(image),clean_image_sha256=file_hash(image),original_evidence=state['evidence']['evidence'],
            objects={role:dict(object=state['evidence'][role]) for role in ('subject','reference')},
            terminal_outcomes_used=False)
        log.write(json.dumps(row)+'\n');log.flush()
        print(json.dumps(dict(state_id=state['state_id'],phase='train_clean_exported')),flush=True)
