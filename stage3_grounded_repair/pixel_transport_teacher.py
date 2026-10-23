"""Object-preserving pixel composition for an OFFLINE teacher interface only.

Unlike a latent blur, explicit source removal cannot retain an object's latent
ghost. Original instance pixels are warped with premultiplied alpha. No terminal
image or correctness label enters this operation. It is not the final method.
"""
import numpy as np
from object_transport import target_box


def compose_instances(image, masks, boxes, relation, opposite=False, strength=1.):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != (432, 432, 3):
        raise ValueError('Expected the intermediate RGB uint8 image at 432x432')
    if not 0 <= strength <= 1:
        raise ValueError('Invalid composition strength')
    if strength == 0:
        return image.copy(), {'zero': True}
    import cv2
    alpha = {role: np.asarray(masks[role], dtype=np.float32)
             for role in ('subject', 'reference')}
    for role, value in alpha.items():
        if value.shape != (432, 432) or not np.isfinite(value).all():
            raise ValueError('Invalid instance mask')
        if value.min() < 0 or value.max() > 1 or not value.any():
            raise ValueError('Empty or unnormalized instance mask')
        x1, y1, x2, y2 = boxes[role][:4]
        if not (0 <= x1 < x2 <= 432 and 0 <= y1 < y2 <= 432):
            raise ValueError('Invalid source box')
    union = np.maximum(alpha['subject'], alpha['reference'])
    # Fixed five-pixel dilation removes antialiasing/fringes of the OLD objects.
    erase = cv2.dilate((union > .5).astype(np.uint8), np.ones((5, 5), np.uint8))
    filled = cv2.inpaint(image, erase, 5, cv2.INPAINT_TELEA).astype(np.float32)
    diagnostics = {'zero': False, 'source_erased_pixels': int(erase.sum()),
                   'erase_fraction': float(erase.mean()), 'roles': {}}
    # Fixed reference-then-subject order; no semantic evaluator chooses layering.
    for role in ('reference', 'subject'):
        box = boxes[role]
        target, scale = target_box(box, relation, role, opposite)
        old_cx, old_cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        new_cx, new_cy = (target[0] + target[2]) * 216, (target[1] + target[3]) * 216
        matrix = np.array([[scale, 0, new_cx - scale * old_cx],
                           [0, scale, new_cy - scale * old_cy]], np.float32)
        moved_alpha = cv2.warpAffine(alpha[role], matrix, (432, 432),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        color = image.astype(np.float32) * alpha[role][..., None]
        moved_color = cv2.warpAffine(color, matrix, (432, 432),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        filled = filled * (1 - moved_alpha[..., None]) + moved_color
        diagnostics['roles'][role] = {
            'source_box': [float(v) for v in box[:4]],
            'target_box': [v * 432 for v in target], 'scale': scale,
            'source_mask_pixels': float(alpha[role].sum()),
            'moved_mask_pixels': float(moved_alpha.sum()),
            'source_mask_sha256': __import__('hashlib').sha256(alpha[role].tobytes()).hexdigest()}
    result = image.astype(np.float32) + strength * (filled - image)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8), diagnostics


def image_to_vae(image, device):
    """Official WanVAE deterministic encoder input: B,C,T,H,W in [-1,1]."""
    import torch
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.shape != (432, 432, 3):
        raise ValueError('Unexpected encoder image')
    return torch.from_numpy(value.copy()).permute(2, 0, 1)[None, :, None].to(device).float() / 127.5 - 1
