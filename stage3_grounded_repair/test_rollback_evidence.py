import unittest
import torch
from runtime import tensor_hash
from rollback_evidence import checked_then_rewound


def state(index):
    latent=torch.ones(2,1,2,2)*index;prefix=torch.ones(2,4,3)*index
    return {'state_id':f'p_seed101_i{index}','prompt_id':'p','prompt':'a photo of a cat left of a dog',
        'seed':101,'split':'architecture_dev','CFG':5,'prefix_cache_version':'v1',
        'time_grid':torch.linspace(0,1,40),'flow_index':index,'true_t':float(torch.linspace(0,1,40)[index]),
        'runtime':{'base_model_hash':'frozen'},'latent':latent,'prefix_cache':prefix,
        'state_fingerprint':tensor_hash(latent),'prefix_cache_hash':tensor_hash(prefix),
        'zero_intervention_test':{'passed':True},
        'evidence':{'subject':'cat','reference':'dog','relation':'left of','evidence':index},
        'baseline_final_latent':f'EARLY_TERMINAL_{index}'}


class RollbackTests(unittest.TestCase):
    def test_only_evidence_imported_and_extra_cost_explicit(self):
        early,late=state(20),state(35)
        result=checked_then_rewound(early,late)
        self.assertIs(result['latent'],early['latent']);self.assertIs(result['prefix_cache'],early['prefix_cache'])
        self.assertEqual(result['baseline_final_latent'],early['baseline_final_latent'])
        self.assertEqual(result['evidence']['evidence'],35)
        self.assertEqual(early['evidence']['evidence'],20)
        self.assertEqual(result['rollback']['extra_discarded_velocity_calls_vs_no_rewind'],16)
        self.assertFalse(result['rollback']['later_terminal_outcome_used'])

    def test_other_seed_or_split_rejected(self):
        for key,value in [('seed',202),('split','val'),('prompt_id','other')]:
            early,late=state(20),state(35);late[key]=value
            with self.assertRaisesRegex(ValueError,'mismatch'):checked_then_rewound(early,late)

    def test_not_later_or_changed_grid_rejected(self):
        with self.assertRaisesRegex(ValueError,'strictly later'):checked_then_rewound(state(20),state(20))
        early,late=state(20),state(35);late['time_grid']=late['time_grid']+1
        with self.assertRaisesRegex(ValueError,'grid mismatch'):checked_then_rewound(early,late)

    def test_corrupt_state_or_time_rejected(self):
        early,late=state(20),state(35);late['latent']+=1
        with self.assertRaisesRegex(ValueError,'Corrupt'):checked_then_rewound(early,late)
        early,late=state(20),state(35);late['true_t']=.9
        with self.assertRaisesRegex(ValueError,'flow time'):checked_then_rewound(early,late)


if __name__=='__main__':unittest.main()
