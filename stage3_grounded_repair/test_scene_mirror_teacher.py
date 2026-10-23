import unittest
import numpy as np
from scene_mirror_teacher import compose_instances


class SceneMirrorTests(unittest.TestCase):
    def setUp(self):
        self.image=np.zeros((432,432,3),np.uint8)
        self.image[80:140,30:90]=[255,30,10]
        self.image[200:300,300:360]=[10,20,255]

    def test_identity_and_vertical_exclusion(self):
        for relation,strength in [('left of',0.),('above',1.),('below',1.)]:
            result,diag=compose_instances(self.image,{}, {},relation,strength=strength)
            self.assertTrue(np.array_equal(result,self.image))
            self.assertTrue(diag['zero'])

    def test_target_reverses_horizontal_geometry_and_preserves_pixels(self):
        result,diag=compose_instances(self.image,{}, {},'right of')
        red=np.where(result[...,0]==255);blue=np.where(result[...,2]==255)
        self.assertGreater(red[1].mean(),blue[1].mean())
        self.assertTrue(np.array_equal(np.sort(result.reshape(-1,3),axis=0),np.sort(self.image.reshape(-1,3),axis=0)))
        self.assertEqual(diag['source_erased_pixels'],0)
        restored,_=compose_instances(result,{}, {},'right of')
        self.assertTrue(np.array_equal(restored,self.image))
        self.assertTrue(result.flags.c_contiguous)

    def test_wrong_axis_does_not_reverse_horizontal_geometry(self):
        result,diag=compose_instances(self.image,{}, {},'right of',opposite=True)
        red=np.where(result[...,0]==255);blue=np.where(result[...,2]==255)
        self.assertLess(red[1].mean(),blue[1].mean())
        self.assertEqual(diag['scene_reflection_axis'],'vertical_wrong_axis')
        with self.assertRaises(ValueError):
            compose_instances(self.image,{}, {},'right of',strength=.5)


if __name__=='__main__':unittest.main()
