import json,tempfile,unittest
from pathlib import Path
from prepare_train_expansion import prepare


class ExpansionTests(unittest.TestCase):
    def make(self,root):
        rows=[]
        for rel in ('left of','right of','above','below'):
            for i in range(5):
                prompt=f'object{i} {rel} reference';rows.append(dict(prompt_id=rel+str(i),prompt=prompt,split='train',metadata=dict(prompt=prompt,include=[{},dict(position=[rel,0])])) )
        source=root/'source.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        exclude=root/'exclude.json';exclude.write_text(json.dumps([r for r in rows if r['prompt_id'].endswith('0')]))
        return source,exclude,rows

    def test_balanced_disjoint_frozen_without_outcomes(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);source,exclude,rows=self.make(root);output=root/'tasks.json'
            chosen=prepare(source,exclude,output,count=8)
            self.assertEqual(len(chosen),8)
            self.assertTrue(all(r['split']=='train' and r['seeds']==[101,202] for r in chosen))
            self.assertTrue(all(not r['prompt_id'].endswith('0') for r in chosen))
            self.assertEqual(chosen,prepare(source,exclude,output,count=8))
            self.assertFalse(json.loads(output.with_suffix('.config.json').read_text())['configuration']['terminal_outcomes_used'])

    def test_split_and_source_identity_guard(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);source,exclude,rows=self.make(root)
            bad=json.loads(exclude.read_text());bad[0]['prompt']='different';exclude.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):prepare(source,exclude,root/'tasks.json',count=8)
            exclude.write_text('[]');rows[0]['split']='val';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):prepare(source,exclude,root/'tasks.json',count=8)


if __name__=='__main__':unittest.main()
