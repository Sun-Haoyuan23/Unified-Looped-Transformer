import unittest
import numpy as np
from minimal_pixel_teacher import minimum_geometry, compose_instances


class MinimalTests(unittest.TestCase):
    def scene(self):
        boxes = {'subject':[84,272,345,391], 'reference':[90,169,160,339]}
        masks = {r:np.zeros((432,432),np.float32) for r in boxes}
        for r,b in boxes.items():masks[r][b[1]:b[3],b[0]:b[2]]=1
        image=np.zeros((432,432,3),np.uint8)+50
        image[masks['subject']>0]=[230,150,0];image[masks['reference']>0]=[255,230,0]
        return image,masks,boxes

    def test_large_anchor_and_full_size_when_feasible(self):
        _,m,b=self.scene();target,scale,plan=minimum_geometry(b,m,'left of')
        np.testing.assert_array_equal(target['subject'],b['subject'])
        self.assertEqual(scale,1)
        self.assertGreaterEqual(target['reference'][0]-target['subject'][2],12)
        self.assertEqual(target['reference'][2]-target['reference'][0],70)

    def test_only_necessary_scale_and_frame_bounds(self):
        _,m,b=self.scene();b={'subject':[65,266,404,405],'reference':[226,163,379,212]}
        for opposite in (False,True):
            target,scale,plan=minimum_geometry(b,m,'right of',opposite)
            self.assertAlmostEqual(scale,420/492)
            for value in target.values():
                self.assertGreaterEqual(min(value),-1e-6);self.assertLessEqual(max(value),432+1e-6)
            low='subject' if opposite else 'reference';high='reference' if opposite else 'subject'
            self.assertAlmostEqual(target[high][0]-target[low][2],12)

    def test_zero_and_already_correct_scene_exact(self):
        image,m,b=self.scene();zero,_=compose_instances(image,m,b,'left of',strength=0)
        np.testing.assert_array_equal(zero,image)
        b={'subject':[20,100,80,200],'reference':[300,100,360,200]}
        stable,_=compose_instances(image,m,b,'left of')
        np.testing.assert_array_equal(stable,image)


if __name__=='__main__':unittest.main()
