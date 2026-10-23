import unittest
from train_geometry_gate import ACTIONS,geometry_gate
class TrainGeometryTests(unittest.TestCase):
    def test_complete_target_and_roundtrip_required(self):
        rows=[dict(state_id=s,action=a,correct=a.startswith('target')) for s in ('t1','t2') for a in ACTIONS]
        self.assertEqual(geometry_gate(dict(rows=rows),['t1','t2']),{'t1':True,'t2':True})
        changed=[dict(r,correct=False) if (r['state_id'],r['action'])==('t1','target_roundtrip') else r for r in rows]
        self.assertEqual(geometry_gate(dict(rows=changed),['t1','t2']),{'t1':False,'t2':True})
    def test_incomplete_duplicate_or_missing_rejected(self):
        rows=[dict(state_id='t',action=a,correct=True) for a in ACTIONS]
        for invalid in (rows[:-1],rows+[rows[0]],[dict(r,correct=None) if i==0 else r for i,r in enumerate(rows)]):
            with self.assertRaises(ValueError):geometry_gate(dict(rows=invalid),['t'])
        with self.assertRaises(ValueError):geometry_gate(dict(rows=rows),['dev'])
if __name__=='__main__':unittest.main()
