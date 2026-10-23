"""Offline scene-consistent horizontal teacher; never a final pixel-edit method.

Keep every instance, background and support surface together. Horizontal
reflection targets left/right errors. The orthogonal reflection is explicitly
a wrong-axis control, not opposite evidence. Vertical tasks do not trigger.
"""
import numpy as np


def image_to_vae(image, device):
    from pixel_transport_teacher import image_to_vae as convert
    return convert(image,device)


def compose_instances(image, masks, boxes, relation, opposite=False, strength=1.):
    image=np.asarray(image)
    if image.dtype!=np.uint8 or image.shape!=(432,432,3):
        raise ValueError('Expected original432 RGB intermediate')
    if strength not in (0.,1.):
        raise ValueError('Reflection is discrete; no strength sweep')
    if relation not in ('left of','right of','above','below'):
        raise ValueError('Unknown relation')
    if strength==0 or relation in ('above','below'):
        return image.copy(),dict(zero=True,vertical_task_not_supported=relation in ('above','below'))
    axis=0 if opposite else 1
    result=np.flip(image,axis=axis).copy()
    return result,dict(zero=False,scene_reflection_axis='vertical_wrong_axis' if opposite else 'horizontal_target',
                       source_erased_pixels=0,resize=False,all_scene_pixels_preserved=True,
                       teacher_only=True,terminal_outcomes_used=False)
