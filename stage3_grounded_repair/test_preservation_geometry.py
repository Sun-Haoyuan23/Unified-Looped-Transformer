import unittest
import torch
from preservation_geometry import preservation_geometry


class PreservationGeometryTests(unittest.TestCase):
    def check_geometry(self, p, t, y):
        result, artifact = preservation_geometry(torch.tensor(p,dtype=torch.float64),
            torch.tensor(t,dtype=torch.float64),torch.tensor(y,dtype=torch.float64),['teacher'],[len(t)])
        self.assertLess(result['qr_relative_residual'],1e-12)
        self.assertLess(result['svd_relative_residual'],1e-12)
        return result['precision_conventions']

    def test_full_rank_preservation_leaves_no_repair_direction(self):
        result=self.check_geometry([[1,0],[0,1],[1,1]],[[0,1],[1,0]],[[1],[2]])
        for row in result.values():
            self.assertEqual(row['rank'],2)
            self.assertEqual(row['null_dimensions'],0)
            self.assertEqual(row['teacher_target_projection_residual_energy_ratio'],1)

    def test_true_null_direction_separates_target(self):
        result=self.check_geometry([[1,0,0],[0,1,0],[1,1,0]],[[0,0,1],[0,0,2]],[[1],[2]])
        for row in result.values():
            self.assertEqual(row['null_dimensions'],1)
            self.assertEqual(row['projected_teacher_resolvable_rank'],1)
            self.assertLess(row['teacher_target_projection_residual_energy_ratio'],1e-25)
            self.assertLess(row['preservation_feature_energy_in_selected_null_fraction'],1e-25)

    def test_null_feature_energy_does_not_imply_target_separability(self):
        result=self.check_geometry([[1,0,0],[0,1,0],[1,1,0]],[[0,0,1],[0,0,1]],[[1],[-1]])
        for row in result.values():
            self.assertAlmostEqual(row['teacher_feature_energy_in_preservation_null_fraction'],1)
            self.assertAlmostEqual(row['teacher_target_projection_residual_energy_ratio'],1)

    def test_precision_convention_is_explicit(self):
        result=self.check_geometry([[1,0,0],[0,1e-8,0],[0,0,0]],[[0,1,0]],[[1]])
        self.assertEqual(result['fp64_arithmetic']['rank'],2)
        self.assertEqual(result['fp32_feature_resolution']['rank'],1)
        self.assertEqual(result['fp64_arithmetic']['teacher_target_projection_residual_energy_ratio'],1)
        self.assertLess(result['fp32_feature_resolution']['teacher_target_projection_residual_energy_ratio'],1e-25)


if __name__=='__main__':unittest.main()
