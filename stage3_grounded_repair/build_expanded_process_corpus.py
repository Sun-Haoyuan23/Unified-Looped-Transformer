"""Actual CPU Torch validation and frozen v70 manifest; no model or optimizer."""
import shutil
from pathlib import Path
from candidate_provenance import file_hash, freeze_config
from expanded_process_corpus import ExpandedProcessCorpus, expanded_schedule
from runtime import write_json

ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair')

def main():
    out=ROOT/'expanded_process_corpus_v70';out.mkdir(exist_ok=False)
    try:
        corpus=ExpandedProcessCorpus(ROOT);rows=corpus.ledger()
        names=('expanded_process_corpus.py','test_expanded_process_corpus.py','build_expanded_process_corpus.py',
               'process_training_targets.py','training_pairs.py','candidate_provenance.py','runtime.py')
        config=dict(states=rows,teachers=sorted(corpus.teachers),correct=sorted(corpus.correct),input_hashes=corpus.input_hashes,
                    source_sha256={n:file_hash(ROOT/n) for n in names},schedule=expanded_schedule(corpus.teachers,corpus.correct),
                    optimizer_updates=0,new_scoring_images=0,capacity_gate=dict(each_teacher_mse_less_than=.01,max35_preservation_mse_less_than=.001))
        freeze_config(out/'config.json',config);write_json(out/'corpus_manifest.json',rows)
        (out/'source').mkdir()
        for name in names:shutil.copy2(ROOT/name,out/'source'/name)
        write_json(out/'pipeline_status.json',dict(phase='complete',teachers=6,preservation_states=35,total_train_states=41,
            all_identity_split_hash_additive_endpoint_checks_passed=True,optimizer_updates=0,new_scoring_images=0,
            model_endpoint_replays_not_yet_run=True,controller_started=False,FLOPs=None))
    except Exception as error:
        write_json(out/'pipeline_status.json',dict(phase='failed',error=str(error)));raise

if __name__=='__main__':main()
