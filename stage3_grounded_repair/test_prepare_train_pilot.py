import json
import tempfile
import unittest
from pathlib import Path
from prepare_train_pilot import prepare


class PilotTests(unittest.TestCase):
    def rows(self):
        return [{'prompt_id':f'train_{i:03d}_{j}','split':'train','prompt':f'prompt {i} {j}',
            'metadata':{'prompt':f'prompt {i} {j}','include':[{'class':'ref'},
                {'class':'subject','position':[relation,0]}]}}
            for i,relation in enumerate(['left of','right of','above','below']) for j in range(3)]

    def run_pilot(self,rows,count=8):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name)/'train.jsonl'
        self.output=Path(self.temp.name)/'pilot.json'
        self.source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return prepare(self.source,self.output,count)

    def test_balanced_selection_is_independent_of_source_order(self):
        rows=self.rows();chosen=self.run_pilot(list(reversed(rows)))
        self.assertEqual(len(chosen),8)
        self.assertTrue(all(row['seeds']==[101,202] for row in chosen))
        self.assertTrue(all(row['prompt_id'].endswith(('_0','_1')) for row in chosen))
        self.assertEqual(chosen,prepare(self.source,self.output,8))

    def test_validation_contamination_rejected(self):
        rows=self.rows();rows[0]['split']='val'
        with self.assertRaises(ValueError):self.run_pilot(rows)

    def test_metadata_prompt_disagreement_rejected(self):
        rows=self.rows();rows[0]['metadata']['prompt']='another prompt'
        with self.assertRaises(ValueError):self.run_pilot(rows)

    def test_changed_source_cannot_overwrite_frozen_pilot(self):
        self.run_pilot(self.rows())
        self.source.write_text(self.source.read_text()+'\n')
        with self.assertRaises(RuntimeError):prepare(self.source,self.output,8)


if __name__=='__main__':unittest.main()
