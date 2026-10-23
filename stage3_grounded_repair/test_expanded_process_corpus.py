import unittest
from collections import Counter
import torch
from expanded_process_corpus import check_quality, checked_additive, expanded_schedule


class ExpandedTests(unittest.TestCase):
    def test_schedule(self):
        teachers=[f't{i}' for i in range(6)]; correct=[f'p{i}' for i in range(35)]
        rows=expanded_schedule(teachers,correct)
        self.assertEqual(len(rows),420)
        self.assertEqual(Counter(s for s,p in rows if not p),Counter({s:35 for s in teachers}))
        self.assertEqual(Counter(s for s,p in rows if p),Counter({s:6 for s in correct}))
        self.assertEqual(rows,expanded_schedule(teachers,correct))
        self.assertTrue(all(p==(i%2==1) for i,(s,p) in enumerate(rows)))

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):expanded_schedule(range(6),range(35))

    def test_additive_identity(self):
        original=torch.ones(2,1,2,2);delta=torch.full_like(original,.3)
        self.assertTrue(torch.equal(checked_additive(original,delta,original+delta),original+delta))
        with self.assertRaises(ValueError):checked_additive(original,delta,original)

    def test_cfg_mismatch(self):
        original=torch.ones(2,1,2,2);delta=torch.zeros_like(original);delta[1]=1
        with self.assertRaises(ValueError):checked_additive(original,delta,original+delta)

    def test_train_and_quality_gate(self):
        row=dict(state_id='x',split='train',prompt='p',prompt_id='p',seed=101,flow_index=35,true_t=.4,state_fingerprint='h')
        pair=dict(baseline_correct=0,candidate_correct=1,baseline_metrics=dict(hps=.2,pickscore=20.),candidate_metrics=dict(hps=.21,pickscore=21.))
        self.assertGreater(check_quality(row,pair,row,pair)['hps'],0)
        for key,value in [('split','val'),('seed',202)]:
            altered=dict(row,**{key:value})
            with self.assertRaises(ValueError):check_quality(altered,pair,row,pair)
        for value in (.1,float('nan')):
            changed=dict(pair,candidate_metrics=dict(hps=value,pickscore=21.))
            with self.assertRaises(ValueError):check_quality(row,changed,row,pair)

if __name__=='__main__':unittest.main()
