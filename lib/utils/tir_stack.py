import cv2 as cv
import numpy as np
import torch
from collections import deque


def stack_pair(cur, prev):
    """ Stack a thermal frame pair into a 3-channel image: [I_t, I_prev, |I_t - I_prev| normalized].

    Each input is HxW or HxWx3 uint8 (thermal loads as color-replicated gray); channel 0 is used when 3D.
    Pass prev=cur for the first frame (zero diff channel).
    """
    cur_g = cur[:, :, 0] if cur.ndim == 3 else cur
    prev_g = prev[:, :, 0] if prev.ndim == 3 else prev
    diff = cv.absdiff(cur_g, prev_g)
    diff_norm = (diff * (255.0 / max(int(diff.max()), 1))).astype(np.uint8)
    return np.stack([cur_g, prev_g, diff_norm], axis=-1)


def stack_pair_gpu(cur, prev):
    """ GPU counterpart of stack_pair for (C,H,W) float frames in 0..255; returns a (3,H,W) float tensor. """
    # accept (3,H,W) color-replicated gray, (1,H,W) gray, or (H,W); use channel 0 when 3D
    cur_g = cur[0] if cur.dim() == 3 else cur
    prev_g = prev[0] if prev.dim() == 3 else prev
    diff = (cur_g - prev_g).abs()
    diff = diff * (255.0 / diff.max().clamp(min=1.0))
    return torch.stack([cur_g, prev_g, diff], dim=0)


class TIRFrameStacker:
    """ Stateful stacking for TIR inference: motion cues survive thermal crossover at zero extra learned parameters. """

    def __init__(self, delta=1):
        self.delta = delta
        self._history = deque(maxlen=delta)

    def reset(self):
        self._history.clear()

    def stack(self, frame_np):
        prev = self._history[0] if len(self._history) == self.delta else frame_np
        self._history.append(frame_np)
        return stack_pair(frame_np, prev)

    def stack_gpu(self, frame):
        prev = self._history[0] if len(self._history) == self.delta else frame
        self._history.append(frame)
        return stack_pair_gpu(frame, prev)
