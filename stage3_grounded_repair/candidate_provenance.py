"""Immutable continuation configuration and same-time evidence pairing."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze_config(path, config, has_existing_records=False):
    """Refuse to append candidates generated with a different implementation."""
    path = Path(path)
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False)
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
    payload = {'configuration': config, 'configuration_sha256': fingerprint}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise RuntimeError('Candidate configuration changed; use a new experiment directory')
    elif has_existing_records:
        raise RuntimeError('Existing candidates have no immutable configuration; use a new directory')
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return fingerprint


def shuffled_source(metadata, state_id):
    """Different prompt, identical split and flow time; deterministic cyclic choice."""
    state = metadata[state_id]
    eligible = sorted(key for key, row in metadata.items()
        if row['split'] == state['split'] and row['flow_index'] == state['flow_index']
        and row['prompt_id'] != state['prompt_id'])
    if not eligible:
        raise RuntimeError('Shuffled evidence needs another prompt in the same split and flow step')
    offset = int(hashlib.sha256(state_id.encode()).hexdigest()[:16], 16) % len(eligible)
    return eligible[offset]


def verify_completed(row, fingerprint):
    if row.get('configuration_sha256') != fingerprint:
        raise RuntimeError('Completed candidate configuration differs')
    for name in ('image_path', 'final_latent'):
        path = Path(row[name])
        if not path.is_file() or file_hash(path) != row.get(name + '_sha256'):
            raise RuntimeError('Completed candidate file missing or changed: ' + str(path))
