"""TRAIN preparation only: freeze v55 probes as new, explicitly derived heads."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from analytic_readout_operator import converted_readouts, READOUT_KEYS
from candidate_provenance import file_hash, freeze_config
from runtime import tensor_hash, write_json
from training_pairs import selected_pairs, validate_training_pairs

SOURCE_NAMES = ('analytic_readout_operator.py', 'prepare_analytic_readout.py',
    'validate_analytic_readout.py', 'build_analytic_readout_candidates.py',
    'run_analytic_readout_trainfit.py', 'test_analytic_readout_operator.py',
    'trainable_recurrent_state.py', 'recurrent_training_protocol.py', 'state_space_revisit.py',
    'runtime.py', 'evidence_encoder.py', 'candidate_provenance.py', 'training_pairs.py',
    'evaluate_oracle_pipeline.py', 'evaluate_oracle_pipeline_available.py',
    'evaluate_repair_pairwise.py', 'export_results.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.work/'stage3_grounded_repair'
    args.output.mkdir(exist_ok=False)
    checkpoint = root/'process_recurrent_train_v52/joint_process/operator_step100.pt'
    cohort_file = root/'process_recurrent_trainfit_v53/cohort.json'
    probe_root = root/'process_readout_probe_v55'
    if json.loads((probe_root/'pipeline_status.json').read_text())['phase'] != 'complete':
        raise ValueError('Require completed v55 fixed probe protocol')
    source = json.loads((probe_root/'config.json').read_text())['configuration']
    if source['checkpoint_sha256'] != file_hash(checkpoint) or source['cohort_sha256'] != file_hash(cohort_file):
        raise ValueError('v55 checkpoint/cohort source changed')
    teacher_root = root/'repair_train_targets_v36'
    report = json.loads((teacher_root/'paired_report.json').read_text())
    records = [json.loads(line) for line in (teacher_root/'all_actions.jsonl').read_text().splitlines()]
    gate = validate_training_pairs(report, records)
    pairs = selected_pairs(report, records)
    if gate['selected_teacher_states'] != 3 or gate['preservation_states'] != 35:
        raise ValueError('Strict TRAIN teacher/preservation gate changed')
    cohort = json.loads(cohort_file.read_text())
    if (cohort['split'] != 'train' or len(cohort['state_ids']) != 6 or
            cohort['teacher_state_ids'] != sorted(pairs) or
            cohort['teacher_report_sha256'] != file_hash(teacher_root/'paired_report.json')):
        raise ValueError('Frozen six TRAIN diagnostic cohort changed')
    baselines = {row['state_id']: row for row in records if row['action'] == 'baseline'}
    files = {path.stem: path for path in (root/'states/repair_train_targets_v36').glob('*_i35.pt')}
    if len(files) != 112 or set(files) != set(baselines):
        raise ValueError('Expected exactly the original 112 TRAIN evidence states')
    metadata = {}
    for sid, row in baselines.items():
        if row['split'] != 'train' or row['flow_index'] != 35:
            raise ValueError('Extra split/time in TRAIN evidence')
        metadata[sid] = {key: row[key] for key in ('split', 'flow_index', 'prompt_id', 'state_fingerprint')}
    for sid in cohort['state_ids']:
        if file_hash(files[sid]) != cohort['state_sha256'][sid]:
            raise ValueError('Frozen state bytes changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (payload['step'] != 100 or payload['level'] != 2 or payload['width'] != 64 or
            payload['metadata']['operator_kind'] != 'recurrent_joint_ilsd'):
        raise ValueError('Require original fixed final step100 joint_process operator')
    for name, digest in payload['metadata']['training_code_sha256'].items():
        if file_hash(root/name) != digest:
            raise ValueError('Frozen training interface changed: '+name)
    source_hashes = {name: file_hash(root/name) for name in SOURCE_NAMES}
    (args.output/'source').mkdir()
    for name in SOURCE_NAMES:
        shutil.copy2(root/name, args.output/'source'/name)
    shutil.copy2(cohort_file, args.output/'cohort.json')
    results = json.loads((probe_root/'results.json').read_text())
    derivations = {}
    for mode, probe in (('loop2', 'trained_loop2'), ('no_loop', 'trained_no_loop')):
        probe_file = probe_root/(probe+'_probe.pt')
        if file_hash(probe_file) != results[probe]['coefficient_sha256']:
            raise ValueError('Fixed analytical probe bytes changed')
        fitted = torch.load(probe_file, map_location='cpu', weights_only=False)
        if fitted['ridge'] != 1e-4 or not fitted['diagnostic_only']:
            raise ValueError('Unexpected analytical fit source')
        adapter, dimensions = converted_readouts(payload['operator'], fitted['coefficient'])
        changed = [key for key in adapter if not torch.equal(adapter[key], payload['operator'][key])]
        if set(changed) != set(READOUT_KEYS):
            raise ValueError('Only the four existing readout tensors may change')
        directory = args.output/'prepared'/mode
        directory.mkdir(parents=True)
        derived = dict(operator=adapter, metadata=payload['metadata'], step=100, level=2, width=64,
            derivation=dict(kind='optimizer_step100_then_analytical_fit', mode=mode,
                optimizer_updates_after_source=0, source_checkpoint=str(checkpoint),
                source_checkpoint_sha256=file_hash(checkpoint), source_probe=str(probe_file),
                source_probe_sha256=file_hash(probe_file), ridge=1e-4,
                changed_tensors=changed, dimensions=dimensions,
                coefficient_conversion='FP64 to existing FP32 readouts; common bias outer, inner bias zero'))
        derived_file = directory/'operator_step100_analytical_fit.pt'
        torch.save(derived, derived_file)
        derivations[mode] = dict(derived['derivation'], checkpoint=str(derived_file),
            checkpoint_sha256=file_hash(derived_file),
            adapter_tensor_hashes={key: tensor_hash(value) for key, value in adapter.items()})
    manifest = dict(source_checkpoint=str(checkpoint), source_checkpoint_sha256=file_hash(checkpoint),
        cohort_sha256=file_hash(args.output/'cohort.json'), state_metadata=metadata,
        evidence_state_sha256={sid: file_hash(path) for sid, path in files.items()},
        derivations=derivations, strict_training_gate=gate, source_sha256=source_hashes,
        scoring_rows=66, optimizer_updates=0, analytical_deployments=2,
        deployment_target_reads=False, controller=False, validation_images=0)
    write_json(args.output/'preparation_manifest.json', manifest)
    freeze_config(args.output/'config.json', dict(manifest,
        actions={'loop2': ['baseline', 'conditional_zero', 'conditional_revisit', 'shuffled_evidence',
                           'opposite_evidence', 'conditional_budget1'],
                 'no_loop': ['baseline', 'conditional_zero', 'conditional_revisit', 'shuffled_evidence', 'opposite_evidence']},
        protocol='Fixed v55 trained_loop2 and trained_no_loop; no score/val-selected fit',
        no_loop_definition='Same feedback, evidence, head; all shared Transformer re-visits skipped',
        conversion_tolerance=dict(rtol=1e-4, atol=1e-5), FLOPs=None))
    write_json(args.output/'preparation_status.json', dict(phase='complete', optimizer_updates=0,
        derived_checkpoints=2, only_existing_readout_tensors_changed=True))


if __name__ == '__main__':
    main()
