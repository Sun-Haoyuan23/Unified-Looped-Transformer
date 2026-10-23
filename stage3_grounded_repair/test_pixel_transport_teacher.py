import unittest
import numpy as np
from pixel_transport_teacher import compose_instances, image_to_vae


class PixelTeacherTests(unittest.TestCase):
    def scene(self):
        image = np.zeros((432, 432, 3), np.uint8) + 50
        subject = np.zeros((432, 432), np.float32)
        reference = subject.copy()
        subject[150:210, 220:280] = 1
        reference[150:210, 60:120] = 1
        image[subject > 0] = [240, 10, 10]
        image[reference > 0] = [10, 220, 10]
        return image, {'subject': subject, 'reference': reference}, {
            'subject': [220, 150, 280, 210], 'reference': [60, 150, 120, 210]}

    def test_zero_exact_and_no_input_mutation(self):
        image, masks, boxes = self.scene()
        before = image.copy()
        zero, _ = compose_instances(image, masks, boxes, 'left of', strength=0)
        np.testing.assert_array_equal(zero, image)
        compose_instances(image, masks, boxes, 'left of')
        np.testing.assert_array_equal(before, image)

    def test_identity_color_and_source_removal(self):
        image, masks, boxes = self.scene()
        moved, _ = compose_instances(image, masks, boxes, 'left of')
        np.testing.assert_array_equal(moved[180, 108], [240, 10, 10])
        np.testing.assert_array_equal(moved[180, 324], [10, 220, 10])
        # Both old centers are background rather than a blurred original object.
        self.assertLess(np.max(abs(moved[180, 250].astype(float) - 50)), 10)
        self.assertLess(np.max(abs(moved[180, 65].astype(float) - 50)), 10)

    def test_target_reverse_geometry_and_encoder_normalization(self):
        image, masks, boxes = self.scene()
        reverse, diag = compose_instances(image, masks, boxes, 'left of', opposite=True)
        np.testing.assert_array_equal(reverse[180, 324], [240, 10, 10])
        np.testing.assert_array_equal(reverse[180, 108], [10, 220, 10])
        self.assertGreater(diag['roles']['subject']['target_box'][0],
                           diag['roles']['reference']['target_box'][2])
        value = image_to_vae(image, 'cpu')
        self.assertEqual(tuple(value.shape), (1, 3, 1, 432, 432))
        self.assertAlmostEqual(float(value[0, 0, 0, 180, 250]), 240 / 127.5 - 1, places=6)


if __name__ == '__main__':
    unittest.main()
