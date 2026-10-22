import math

import torch
import torch.nn.functional as F

from lib.utils.tir_stack import stack_pair_gpu  # noqa: F401  re-export (see tir_stack contract)


# NOTE: grid_sample bilinear differs from the CPU path's cv2.resize by sub-pixel
# interpolation details (sampling-center convention, border blending with zeros
# instead of crop-then-resize); acceptable for tracking.
# Parity with the CPU path (lib.train.data.processing_utils.sample_target) includes
# its drop-last-row/column quirk at the frame edges: x2_pad = max(x2 - W + 1, 0)
# (and the y counterpart) discards the last real frame column/row whenever the crop
# reaches past it, so the crop then sees an effectively (W-1)/(H-1)-sized frame.


def gpu_sample_target(frame, target_bb, search_area_factor, output_sz, min_crop_size, mean, std):
    """GPU equivalent of lib.train.data.processing_utils.sample_target.

    args:
        frame - CUDA float tensor (C,H,W) in 0..255
        target_bb - target box [x, y, w, h]
        search_area_factor - ratio of crop size to target size
        output_sz - size to which the extracted crop is resized (always square)
        min_crop_size - lower clamp on the crop side in pixels (tiny targets)
        mean/std - per-channel normalization values

    returns:
        img - normalized crop (1,C,output_sz,output_sz) on CUDA
        resize_factor - output_sz / crop_sz, same semantics as the CPU path
        att_mask - bool (output_sz,output_sz) on CUDA, True where the crop
            sources from outside the frame (zero padding)
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    C, H, W = frame.shape
    crop_sz = max(min_crop_size, math.ceil(math.sqrt(w * h) * search_area_factor))

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    y1 = round(y + 0.5 * h - crop_sz * 0.5)

    # effective frame size: when the crop reaches past the right/bottom edge the CPU
    # path's x2_pad/y2_pad (+1) drops the last real column/row, so the sampling grid
    # must treat the frame as one pixel smaller there (interior geometry is unchanged)
    W_eff = W - 1 if x1 + crop_sz >= W else W
    H_eff = H - 1 if y1 + crop_sz >= H else H

    # affine theta mapping output normalized coords to frame normalized coords:
    # output pixel j samples frame coordinate x1 + (j + 0.5) * crop_sz / output_sz - 0.5
    theta = frame.new_tensor([[crop_sz / W_eff, 0.0, (2.0 * x1 + crop_sz) / W_eff - 1.0],
                              [0.0, crop_sz / H_eff, (2.0 * y1 + crop_sz) / H_eff - 1.0]]).unsqueeze(0)
    grid = F.affine_grid(theta, (1, C, output_sz, output_sz), align_corners=False)
    img = F.grid_sample(frame.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # padding mask from which output pixels source from inside the frame:
    # grid coords in (-1, 1) hit real pixels, coords outside get zeros (BORDER_CONSTANT)
    outside = (grid[0, ..., 0] < -1.0) | (grid[0, ..., 0] > 1.0) | \
              (grid[0, ..., 1] < -1.0) | (grid[0, ..., 1] > 1.0)
    att_mask = outside

    mean_t = frame.new_tensor(mean).view(1, C, 1, 1)
    std_t = frame.new_tensor(std).view(1, C, 1, 1)
    img = ((img / 255.0) - mean_t) / std_t

    return img, output_sz / crop_sz, att_mask
