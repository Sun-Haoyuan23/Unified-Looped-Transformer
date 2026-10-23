"""Validate object identity and geometry before any terminal flow experiment.

All localized TRAIN states from an explicit frozen expansion cohort; original/reencoded controls and
target/reverse pixel compositions before/after deterministic WanVAE encoding.
External masks/pixel edits are offline teacher diagnostics, not a loop method.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from scene_mirror_teacher import compose_instances, image_to_vae
from candidate_provenance import file_hash, freeze_config
from runtime import tensor_hash

ACTIONS = ('original_clean', 'original_roundtrip', 'target_pixels',
           'target_roundtrip', 'opposite_pixels', 'opposite_roundtrip')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--middle', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--expected-states', type=int, required=True)
    args = p.parse_args()
    root = args.work / 'stage3_grounded_repair'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'diagnostic_status.json'

    def update(phase, **kw):
        status.write_text(json.dumps({'phase': phase, 'time': time.time(), **kw}, indent=2))

    try:
        if (out / 'records.jsonl').exists():
            raise ValueError('Use a new experiment directory; do not overwrite a frozen diagnostic')
        manifest = args.masks / 'mask_manifest.json'
        capture = json.loads((args.pilot / 'capture_status.json').read_text())
        if not capture['success'] or capture['captured_states'] != args.expected_states:
            raise ValueError('Incomplete explicit TRAIN capture')
        tasks = {q['prompt_id']: q for q in json.loads((args.pilot / 'train_prompts.json').read_text())}
        if sum(len(q['seeds']) for q in tasks.values()) != args.expected_states:
            raise ValueError('Unexpected frozen prompt/seed count')
        if any(q['split'] != 'train' for q in tasks.values()):
            raise ValueError('Frozen cohort contains a non-TRAIN prompt')
        selected = [r for r in json.loads(manifest.read_text())['rows'] if r['eligible']]
        source = args.middle / 'clean_records.jsonl'
        images = {r['state_id']: r for r in map(json.loads, source.read_text().splitlines())
                  if r['flow_index'] == 35}
        paths = sorted(args.states.glob('shard*/*_i35.pt'))
        state_paths = {q.stem: q for q in paths}
        all_images = list(map(json.loads, source.read_text().splitlines()))
        all_masks = json.loads(manifest.read_text())['rows']
        expected = args.expected_states
        if (len(all_images) != expected or len(images) != expected or
                len(paths) != expected or len(state_paths) != expected or
                len(all_masks) != expected or len({q['state_id'] for q in all_masks}) != expected or
                set(images) != set(state_paths) or set(images) != {q['state_id'] for q in all_masks}):
            raise ValueError('Incomplete/duplicate explicit TRAIN cohort')
        frozen_pairs = {(q['prompt_id'], seed) for q in tasks.values() for seed in q['seeds']}
        if {(q['prompt_id'], q['seed']) for q in images.values()} != frozen_pairs:
            raise ValueError('Unexpected frozen TRAIN prompt/seed pairs')
        for row in images.values():
            if (row['split'] != 'train' or row['terminal_outcomes_used'] or row['flow_index'] != 35 or
                    row['prompt'] != tasks[row['prompt_id']]['prompt']):
                raise ValueError('TRAIN-only intermediate provenance mismatch')
        if any(q['split'] != 'train' or q['state_fingerprint'] != images[q['state_id']]['state_fingerprint']
               for q in all_masks):
            raise ValueError('Mask cohort/split/fingerprint mismatch')
        if not selected or any(r['split']!='train' for r in selected):
            raise ValueError('No eligible train states or split leakage')
        evaluation=root/'evaluator_runtime_v1/venv'
        model_config=evaluation/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
        detector=args.work/'stage2b/geneval_models'
        recovery=root/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1'
        verified=json.loads((recovery/'compatibility_report.json').read_text())
        frozen=json.loads((recovery/'config.json').read_text())['configuration']
        if verified!={'n_anchors':144,'all_correctness_and_reasons_exact':True} or file_hash(model_config)!=frozen['model_config_sha256'] or file_hash(detector/'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth')!=frozen['checkpoint_sha256']:
            raise ValueError('Official evaluator differs from frozen compatibility verification')
        checkpoint = Path('/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth')
        freeze_config(out / 'config.json', {
            'source_sha256': {n: file_hash(root / n) for n in
                              ('diagnose_train_scene_mirror_expansion.py', 'scene_mirror_teacher.py', 'pixel_transport_teacher.py',
                               'object_transport.py', 'runtime.py')},
            'pilot_prompts_sha256': file_hash(args.pilot / 'train_prompts.json'),
            'expected_states': args.expected_states,
            'mask_manifest_sha256': file_hash(manifest),
            'grounding_records_sha256': file_hash(source),
            'vae_checkpoint_sha256': file_hash(checkpoint),
            'official_detector_sha256':frozen['checkpoint_sha256'],'official_config_sha256':file_hash(model_config),
            'split':'train',
            'states_sha256': {r['state_id']: file_hash(state_paths[r['state_id']]) for r in selected},
            'selection': 'all eligible TRAIN intermediate states from explicit frozen cohort; no terminal outcome',
            'action_semantics':{'target':'horizontal scene reflection, left/right only','opposite':'vertical wrong-axis reflection for horizontal tasks, not opposite evidence','vertical_tasks':'identity and excluded by intermediate correctness gate'},
            'actions': ACTIONS, 'geometry': 'scene-preserving horizontal reflection for left/right only; vertical reflection wrong-axis control; vertical tasks identity/nontrigger',
            'source_removal': 'none; no cropping, resizing or inpainting',
            'encoder': 'official WanVAE.sample(videos, deterministic=True), float input [-1,1]',
            'optimizer_steps': 0, 'terminal_outcome_used_by_action': False})
        sys.path.insert(0, str(args.work / 'Show-o/show-o2'))
        from models import WanVAE
        from utils import denorm
        if torch.cuda.device_count() != 1:
            raise ValueError('Expose exactly one healthy GPU by UUID')
        vae = WanVAE(vae_pth=str(checkpoint), dtype=torch.bfloat16, device=torch.device('cuda:0'))
        update('generating_diagnostic', states=len(selected))
        rows, transfers, audit = [], [], []
        for record in selected:
            sid = record['state_id']
            state = torch.load(state_paths[sid], map_location='cpu', weights_only=False)
            original_record = images[sid]
            if (state['split']!='train' or not state['zero_intervention_test']['passed'] or
                    state['flow_index'] != 35 or state['prompt'] != tasks[state['prompt_id']]['prompt'] or
                    state['seed'] not in tasks[state['prompt_id']]['seeds'] or
                    tensor_hash(state['latent']) != state['state_fingerprint']):
                raise ValueError('Invalid frozen TRAIN state')
            if (state['state_fingerprint'] != record['state_fingerprint'] or
                    file_hash(record['mask_path']) != record['mask_sha256'] or
                    file_hash(original_record['clean_image']) != record['clean_image_sha256']):
                raise ValueError('Intermediate state/image/mask provenance changed')
            image = np.asarray(Image.open(original_record['clean_image']).convert('RGB'))
            with np.load(record['mask_path'], allow_pickle=False) as archive:
                masks = {role: archive[role].copy() for role in ('subject', 'reference')}
            zero, _ = compose_instances(image, masks, record['boxes'], state['evidence']['relation'], strength=0)
            if not np.array_equal(image, zero):
                raise RuntimeError('Pixel zero control is not exact')
            overlay = image.astype(np.float32)
            for role, color in [('reference', np.array([0, 200, 255])), ('subject', np.array([255, 70, 70]))]:
                overlay = overlay * (1 - .35 * masks[role][..., None]) + color * (.35 * masks[role][..., None])
            overlay_image = Image.fromarray(overlay.astype(np.uint8))
            drawer = ImageDraw.Draw(overlay_image)
            for role in ('subject', 'reference'):
                drawer.rectangle(record['boxes'][role][:4], outline='red' if role == 'subject' else 'cyan', width=2)
            (out / 'mask_audit').mkdir(exist_ok=True)
            overlay_image.save(out / 'mask_audit' / (sid + '.png'))
            audit.append({'state_id': sid, 'relation': state['evidence']['relation'],
                          'objects': original_record['objects'], 'boxes': record['boxes'],
                          'mask_overlap_pixels': int(((masks['subject'] > .5) & (masks['reference'] > .5)).sum())})
            clean = state['latent'][:1].float() + (1 - state['true_t']) * state['baseline_prediction'][:1]
            payload = {'state_id': sid, 'state_fingerprint': state['state_fingerprint'],
                       'mask_sha256': record['mask_sha256'], 'source_image_sha256': record['clean_image_sha256'],
                       'relation': state['evidence']['relation'], 'latents': {}, 'geometry': {}}
            for direction in ('original', 'target', 'opposite'):
                pixels, diag = (image, {'zero': True}) if direction == 'original' else compose_instances(
                    image, masks, record['boxes'], state['evidence']['relation'], opposite=direction == 'opposite')
                started = time.perf_counter()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    encoded = vae.sample(image_to_vae(pixels, 'cuda:0'), deterministic=True)
                    decoded = vae.batch_decode(encoded).squeeze(2)
                torch.cuda.synchronize()
                if encoded.shape != (1, 16, 1, 54, 54) or not torch.isfinite(encoded).all():
                    raise ValueError('Incorrect/nonfinite normalized WanVAE latent')
                roundtrip = denorm(decoded)[0]
                latent = encoded.squeeze(2).float().cpu()
                if sid == selected[0]['state_id'] and direction == 'original':
                    with torch.no_grad():
                        duplicate = vae.sample(image_to_vae(pixels, 'cuda:0'), deterministic=True)
                    if not torch.equal(encoded, duplicate):
                        raise RuntimeError('Deterministic WanVAE encoder/cache not reproducible')
                payload['latents'][direction] = latent
                payload['geometry'][direction] = diag
                for phase, result in [('pixels', pixels), ('roundtrip', roundtrip)]:
                    action = 'original_clean' if direction == 'original' and phase == 'pixels' else direction + '_' + phase
                    directory = out / 'official_layout' / f'{len(rows):06d}'
                    (directory / 'samples').mkdir(parents=True)
                    image_path = directory / 'samples/0000.png'
                    Image.fromarray(result).save(image_path)
                    (directory / 'metadata.jsonl').write_text(json.dumps(state['geneval_metadata']) + '\n')
                    rows.append({'state_id': sid, 'prompt_id': state['prompt_id'], 'prompt': state['prompt'],
                                 'action': action, 'image_path': str(image_path), 'image_sha256': file_hash(image_path),
                                 'mask_sha256': record['mask_sha256'], 'state_fingerprint': record['state_fingerprint'],
                                 'pixel_zero_exact': True, 'roundtrip_pixel_mae': float(abs(roundtrip.astype(float)-pixels).mean()),
                                 'encoded_vs_original_clean_relative_L2': float((latent-clean).norm()/clean.norm()),
                                 'encoded_latent_sha256': tensor_hash(latent), 'encode_decode_time_s': time.perf_counter()-started,
                                 'composition': diag})
            (out / 'teacher_latents').mkdir(exist_ok=True)
            latent_path = out / 'teacher_latents' / (sid + '.pt')
            torch.save(payload, latent_path)
            transfers.append({'state_id': sid, 'state_fingerprint': record['state_fingerprint'],
                              'teacher_path': str(latent_path.resolve()), 'teacher_sha256': file_hash(latent_path),
                              'mask_sha256': record['mask_sha256']})
            update('generating_diagnostic', states_completed=len(transfers), states=len(selected), images=len(rows))
        (out / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (out / 'teacher_manifest.json').write_text(json.dumps({'rows': transfers, 'terminal_labels_used': False}, indent=2))
        (out / 'mask_audit.json').write_text(json.dumps(audit, indent=2))
        del vae
        gc.collect()
        torch.cuda.empty_cache()
        update('evaluating', images=len(rows))
        command = [str(evaluation/'bin/python'),
                   str(args.work/'stage2b/geneval/evaluation/evaluate_images.py'),str(out/'official_layout'),
                   '--outfile',str(out/'official_results.jsonl'),'--model-path',str(detector),
                   '--model-config',str(model_config)]
        with (out / 'detector.log').open('w') as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
        if proc.returncode:
            raise RuntimeError('Official intermediate geometry evaluation failed')
        official = list(map(json.loads, (out / 'official_results.jsonl').read_text().splitlines()))
        lookup = {str(Path(r['image_path']).resolve()): r for r in rows}
        keyed = []
        for result in official:
            record = lookup.pop(str(Path(result['filename']).resolve()))
            if result['prompt'] != record['prompt']:
                raise ValueError('Wrong prompt matched to intermediate image')
            keyed.append(dict(result, state_id=record['state_id'], action=record['action']))
        if lookup:
            raise ValueError('Missing intermediate evaluations')
        (out / 'official_results_keyed.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in keyed))
        report = {'interpretation': 'Intermediate geometry/encoder diagnostic only, not terminal/shared-loop success',
                  'states': len(selected), 'prompts': len({r['prompt_id'] for r in selected}),
                  'by_action': {a: {'n': sum(r['action'] == a for r in keyed),
                                    'correct': sum(bool(r['correct']) for r in keyed if r['action'] == a)} for a in ACTIONS},
                  'rows': keyed, 'optimizer_steps': 0}
        (out / 'geometry_report.json').write_text(json.dumps(report, indent=2))
        update('complete', images=len(rows), states=len(selected), by_action=report['by_action'], optimizer_steps=0)
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
