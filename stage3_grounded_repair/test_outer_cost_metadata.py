import unittest
from recover_outer_pairwise import normalize_cost_metadata


class CostMetadataTests(unittest.TestCase):
    def test_only_boolean_metadata_moves(self):
        row=dict(state_id='s',action='a',image_path_sha256='image',final_latent_sha256='latent',
            cost=dict(network_velocity_calls=14,outer_shared_revisit_calls=16,cached_inner_prefix_used=False,FLOPs=None),
            cost_notes=dict(scope='remaining flow'))
        fixed=normalize_cost_metadata(row)
        self.assertIs(row['cost']['cached_inner_prefix_used'],False)
        self.assertNotIn('cached_inner_prefix_used',fixed['cost'])
        self.assertIs(fixed['cost_notes']['cached_inner_prefix_used'],False)
        for key in ('state_id','action','image_path_sha256','final_latent_sha256'):
            self.assertEqual(row[key],fixed[key])
        for key in ('network_velocity_calls','outer_shared_revisit_calls','FLOPs'):
            self.assertEqual(row['cost'][key],fixed['cost'][key])

    def test_invalid_outer_prefix_reuse_rejected(self):
        with self.assertRaises(ValueError):
            normalize_cost_metadata(dict(cost=dict(cached_inner_prefix_used=True),cost_notes={}))


if __name__=='__main__':unittest.main()
