"""Offline teacher that preserves object size and the larger instance first.

Closed geometry projection, not a gain/threshold/position sweep. Uniform scaling
occurs only when BOTH axis extents plus a fixed12px gap cannot fit in the frame.
The larger mask is anchored if feasible, else displaced only to feasibility.
"""
import numpy as np
from pixel_transport_teacher import image_to_vae


def minimum_geometry(boxes, masks, relation, opposite=False, gap=12.):
    if relation not in ('left of', 'right of', 'above', 'below'):
        raise ValueError('Unknown relation')
    axis = 0 if relation in ('left of', 'right of') else 1
    subject_low = relation in ('left of', 'above')
    if opposite:
        subject_low = not subject_low
    low, high = ('subject', 'reference') if subject_low else ('reference', 'subject')
    sizes = {}
    centers = {}
    for role in ('subject', 'reference'):
        b = np.asarray(boxes[role][:4], float)
        size = b[2:]-b[:2]
        if not np.isfinite(b).all() or (size <= 0).any() or (b < 0).any() or (b > 432).any():
            raise ValueError('Invalid source geometry')
        sizes[role] = size
        centers[role] = (b[:2]+b[2:])/2
    scale = min(1., (432-gap)/(sizes[low][axis]+sizes[high][axis]))
    sizes = {r:v*scale for r,v in sizes.items()}
    distance = (sizes[low][axis]+sizes[high][axis])/2+gap
    minimum_low = sizes[low][axis]/2
    maximum_low = 432-sizes[high][axis]/2-distance
    # Mask area measures how many source pixels would need to be destroyed.
    anchor = max(('subject', 'reference'), key=lambda r: float(np.asarray(masks[r]).sum()))
    original_low, original_high = centers[low][axis], centers[high][axis]
    if (scale == 1 and original_high-original_low >= distance):
        new_low, new_high = original_low, original_high
    elif anchor == low:
        new_low = float(np.clip(original_low, minimum_low, maximum_low))
        new_high = float(np.clip(original_high, new_low+distance, 432-sizes[high][axis]/2))
    else:
        new_high = float(np.clip(original_high, minimum_low+distance, 432-sizes[high][axis]/2))
        new_low = float(np.clip(original_low, minimum_low, new_high-distance))
    centers[low][axis] = new_low
    centers[high][axis] = new_high
    targets = {r: np.concatenate((centers[r]-sizes[r]/2, centers[r]+sizes[r]/2)).tolist()
               for r in ('subject', 'reference')}
    return targets, scale, {'anchor_role': anchor, 'shared_scale': scale, 'axis': axis, 'gap_pixels': gap}


def compose_instances(image, masks, boxes, relation, opposite=False, strength=1.):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != (432,432,3) or not 0 <= strength <= 1:
        raise ValueError('Unexpected intermediate image/strength')
    if strength == 0:
        return image.copy(), {'zero': True}
    import cv2
    alpha = {r: np.asarray(masks[r], np.float32) for r in ('subject','reference')}
    for value in alpha.values():
        if value.shape != (432,432) or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1 or not value.any():
            raise ValueError('Invalid source mask')
    targets, scale, plan = minimum_geometry(boxes, alpha, relation, opposite)
    matrices = {}
    moved_roles = []
    for role in ('subject','reference'):
        old_center = (np.asarray(boxes[role][:2])+np.asarray(boxes[role][2:4]))/2
        new_center = (np.asarray(targets[role][:2])+np.asarray(targets[role][2:]))/2
        shift = new_center-scale*old_center
        matrices[role] = np.array([[scale,0,shift[0]],[0,scale,shift[1]]],np.float32)
        if scale != 1 or np.max(abs(shift)) > 1e-5:
            moved_roles.append(role)
    if not moved_roles:
        return image.copy(), dict(plan, zero=True, moved_roles=[])
    union = np.maximum.reduce([alpha[r] for r in moved_roles])
    erase = cv2.dilate((union > .5).astype(np.uint8),np.ones((5,5),np.uint8))
    filled = cv2.inpaint(image,erase,5,cv2.INPAINT_TELEA).astype(np.float32)
    diagnostics = dict(plan, zero=False, moved_roles=moved_roles, source_erased_pixels=int(erase.sum()),
                       erase_fraction=float(erase.mean()), roles={})
    for role in ('reference','subject'):
        matrix = matrices[role]
        moved_alpha = cv2.warpAffine(alpha[role],matrix,(432,432),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        moved_color = cv2.warpAffine(image.astype(np.float32)*alpha[role][...,None],matrix,(432,432),
                                     flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        filled = filled*(1-moved_alpha[...,None])+moved_color
        diagnostics['roles'][role] = {'source_box':[float(v) for v in boxes[role][:4]],
                                     'target_box':targets[role], 'scale':scale,
                                     'source_mask_pixels':float(alpha[role].sum()),
                                     'moved_mask_pixels':float(moved_alpha.sum())}
    return np.clip(np.rint(image.astype(np.float32)+strength*(filled-image)),0,255).astype(np.uint8), diagnostics
