"""Continue a geometrically validated offline pixel-to-VAE teacher.

Late check35 / same-trajectory rewind20, preserving its early noise residual.
All24 development states retained. Only previously flagged instances with BOTH
target-pixel and target-roundtrip intermediate geometry valid receive edits.
No terminal labels select actions. This is NOT a trained/shared-loop method.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from runtime import Runtime, replay, rng_restore
from rollback_evidence import checked_then_rewound
from candidate_provenance import file_hash, freeze_config, verify_completed

ACTIONS = ('baseline', 'checked_zero', 'checked_original_roundtrip',
           'checked_target_pixel_teacher', 'checked_opposite_pixel_teacher')


def geometry_gate(report):
    keyed = {}
    for row in report['rows']:
        key = (row['state_id'], row['action'])
        if key in keyed:
            raise ValueError('Duplicate intermediate geometry result')
        keyed[key] = row
    states = sorted({k[0] for k in keyed})
    expected = {'original_clean', 'original_roundtrip', 'target_pixels',
                'target_roundtrip', 'opposite_pixels', 'opposite_roundtrip'}
    if len(states) != 6 or set(k[1] for k in keyed) != expected or len(keyed) != 36:
        raise ValueError('Incomplete six-state intermediate geometry diagnosis')
    # This gate reads only manipulated INTERMEDIATE images, never final output.
    return {sid: bool(keyed[sid, 'target_pixels']['correct'] and
                      keyed[sid, 'target_roundtrip']['correct']) for sid in states}


def clean_import_modifier(state, desired):
    interval = state['time_grid'][21] - state['time_grid'][20]
    if float(interval) <= 0:
        raise ValueError('Unexpected rewind Euler interval')
    def modifier(rt, x, t, velocity):
        early_clean = x[:1].float() + (1 - t) * velocity[:1]
        # x=(1-t)*noise+t*clean: replace clean while retaining the SAME noise.
        correction = t * (desired - early_clean) / interval.to(x.device)
        return velocity + torch.cat((correction, correction)), 0, 0
    return modifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--checked-states', type=Path, required=True)
    p.add_argument('--mask-manifest', type=Path, required=True)
    p.add_argument('--teacher-diagnostic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=6)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    early = sorted(args.states.rglob('*_i20.pt'))
    late = {q.stem: q for q in args.checked_states.rglob('*_i35.pt')}
    if len(early) != 24 or len(late) != 24:
        raise ValueError('Expected24 matched early/late development states')
    mask_rows = {r['state_id']: r for r in json.loads(args.mask_manifest.read_text())['rows']}
    diagnostic = args.teacher_diagnostic
    gate = geometry_gate(json.loads((diagnostic / 'geometry_report.json').read_text()))
    teachers = {r['state_id']: r for r in json.loads((diagnostic / 'teacher_manifest.json').read_text())['rows']}
    if set(teachers) != set(gate):
        raise ValueError('Teacher geometry and latent states disagree')
    source = Path(__file__).parent
    log = args.output / f'actions_shard{args.shard}.jsonl'
    config = {'actions': ACTIONS, 'check_index': 35, 'rewind_index': 20, 'window': 1,
              'shard': args.shard, 'num_shards': args.num_shards,
              'mask_manifest_sha256': file_hash(args.mask_manifest),
              'geometry_report_sha256': file_hash(diagnostic / 'geometry_report.json'),
              'teacher_manifest_sha256': file_hash(diagnostic / 'teacher_manifest.json'),
              'teacher_sha256': {sid: r['teacher_sha256'] for sid, r in teachers.items()},
              'geometry_valid_states': [sid for sid in gate if gate[sid]],
              'source_sha256': {n: file_hash(source / n) for n in
                               ('runtime.py', 'rollback_evidence.py', 'build_pixel_teacher_candidates.py',
                                'candidate_provenance.py')},
              'states_sha256': {str(q): file_hash(q) for q in early[args.shard::args.num_shards]},
              'checked_states_sha256': {str(q): file_hash(q) for q in late.values()},
              'optimizer_steps': 0, 'terminal_outcome_used_by_action': False,
              'note': 'Offline external masks + pixel composition + deterministic VAE; not shared recurrence'}
    fingerprint = freeze_config(args.output / f'config_shard{args.shard}.json', config, log.exists())
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            verify_completed(row, fingerprint)
            done.add((row['state_id'], row['action']))
    runtime = Runtime(args.work)
    with log.open('a', encoding='utf-8') as stream:
        for path in early[args.shard::args.num_shards]:
            state = torch.load(path, map_location='cpu', weights_only=False)
            sid = state['state_id'].rsplit('_i', 1)[0] + '_i35'
            checked = torch.load(late[sid], map_location='cpu', weights_only=False)
            rewind = checked_then_rewound(state, checked)
            record = mask_rows[sid]
            if state['split'] != 'architecture_dev' or record['state_fingerprint'] != checked['state_fingerprint']:
                raise ValueError('Invalid development state identity')
            eligible = bool(record['eligible'] and gate.get(sid, False))
            payload = None
            if eligible:
                teacher = teachers[sid]
                if checked['evidence']['evidence'] >= 0 or file_hash(teacher['teacher_path']) != teacher['teacher_sha256']:
                    raise ValueError('Changed intermediate check/teacher')
                payload = torch.load(teacher['teacher_path'], map_location='cpu', weights_only=False)
                if payload['state_fingerprint'] != checked['state_fingerprint'] or payload['mask_sha256'] != record['mask_sha256']:
                    raise ValueError('Teacher from a different state/mask')
                for value in payload['latents'].values():
                    if value.shape != (1, 16, 54, 54) or not torch.isfinite(value).all():
                        raise ValueError('Invalid encoded teacher latent')
            for action in ACTIONS:
                if (state['state_id'], action) in done:
                    continue
                desired = None
                if eligible and action not in ('baseline', 'checked_zero'):
                    direction = {'checked_original_roundtrip': 'original', 'checked_target_pixel_teacher': 'target',
                                 'checked_opposite_pixel_teacher': 'opposite'}[action]
                    desired = payload['latents'][direction].to(runtime.device)
                modifier = None if desired is None else clean_import_modifier(state, desired)
                rng_restore(state['rng'])
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad():
                    final, cost = replay(runtime, state, window=1, velocity_modifier=modifier)
                image = runtime.decode(final)
                torch.cuda.synchronize()
                cost['wall_time_s'] = time.perf_counter() - started
                if desired is None and not torch.equal(final, state['baseline_final_latent'].to(runtime.device)):
                    raise RuntimeError('Zero/untriggered continuation was not exact')
                cost.update(full_network_velocity_calls=49 if action == 'baseline' else 65,
                            full_visual_block_calls=490 if action == 'baseline' else 642)
                directory = args.output / action / state['state_id']
                (directory / 'samples').mkdir(parents=True, exist_ok=True)
                image_path = directory / 'samples/0000.png'
                image.save(image_path)
                (directory / 'metadata.jsonl').write_text(json.dumps(state['geneval_metadata']) + '\n')
                final_path = directory / 'final_latent.pt'
                torch.save({'latent': final.detach().cpu(), 'state_id': state['state_id'], 'action': action}, final_path)
                row = {key: state[key] for key in ('state_id', 'prompt_id', 'prompt', 'seed', 'split', 'flow_index',
                                                 'true_t', 'state_fingerprint', 'zero_intervention_test', 'runtime')}
                row.update(action=action, configuration_sha256=fingerprint, image_path=str(image_path),
                           final_latent=str(final_path), image_path_sha256=file_hash(image_path),
                           final_latent_sha256=file_hash(final_path), cost=cost,
                           original_mask_eligible=record['eligible'], intermediate_geometry_valid=gate.get(sid, False),
                           intervention_eligible=eligible, intervention_applied=desired is not None,
                           checked_state_id=sid, checked_state_fingerprint=checked['state_fingerprint'],
                           rollback=rewind['rollback'] if action != 'baseline' else None,
                           mask_sha256=record['mask_sha256'], teacher_sha256=teachers[sid]['teacher_sha256'] if eligible else None,
                           added_trainable_parameters=0, unique_parameters=state['runtime']['model_unique_parameters'], FLOPs=None,
                           action_note='Offline pixel-to-VAE repairability upper bound, not final pixel editing or shared-loop success',
                           cost_notes={'prefix_cache_reused': True, 'late_evidence_precomputed': True,
                                       'teacher_geometry_precomputed': True,
                                       'cost_scope': 'wall_time is remaining continuation only; full flow calls include discarded late history; VAE/detector/composition diagnostic measured separately',
                                       'mmu_queries_for_late_check': 0 if action == 'baseline' else 2,
                                       'geometry_gate_detector_calls': 3 if eligible and action != 'baseline' else 0,
                                       'teacher_encoding_calls': int(desired is not None),
                                       'teacher_roundtrip_decode_calls': int(desired is not None)})
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(json.dumps({'state_id': state['state_id'], 'action': action, 'applied': desired is not None,
                                  'geometry_valid': gate.get(sid, False), 'status': 'ok'}), flush=True)


if __name__ == '__main__':
    main()
