"""Smoke/regression tests for the runway (approaching-aircraft) adaptation of FocusTrack.

Run:  PYTHONPATH=/home/pc_9211/Desktop/FocusTrack python tests/test_runway_adaptation.py
Plain asserts only, no pytest. Prints PASS/FAIL per case.
"""
import math
import sys
import traceback

import torch  # noqa: E402  (used by module-level helper and most cases)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
YAML_DIR = REPO / "experiments" / "focustrack"


class D(dict):
    """Minimal attribute-dict stand-in for EasyDict (avoids an extra dependency)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def fresh_cfg(yaml_name):
    """Reload the config module singleton and apply one yaml to the fresh copy."""
    import importlib

    import lib.config.focustrack.config as config_mod

    config_mod = importlib.reload(config_mod)
    config_mod.update_config_from_file(str(YAML_DIR / yaml_name))
    return config_mod.cfg


# ---------------------------------------------------------------- case 1
def test_01_config_update():
    for name in ("runway_sra.yaml", "focustrack_ftir_stage1.yaml", "focustrack_ftir_stage2.yaml"):
        cfg = fresh_cfg(name)  # raises ValueError if a yaml key is missing from defaults
        assert cfg is not None
    cfg = fresh_cfg("runway_sra.yaml")
    assert cfg.MODEL.DECODER.ADD_DECODER is False
    assert cfg.MODEL.BACKBONE.CE_LOC == []
    assert float(cfg.TRAIN.L1_WEIGHT) == 7.0
    assert cfg.TRAIN.BUFFERED_IOU is True
    assert float(cfg.TRAIN.BUFFER_RATIO) == 0.2
    assert float(cfg.TRAIN.MIN_BUFFER) == 0.015
    assert float(cfg.DATA.SEARCH.MIN_CROP_SIZE) == 64.0
    assert float(cfg.DATA.TEMPLATE.MIN_CROP_SIZE) == 32.0
    assert float(cfg.TEST.SEARCH.MIN_CROP_SIZE) == 64.0
    assert float(cfg.TEST.TEMPLATE.MIN_CROP_SIZE) == 32.0
    assert cfg.TEST.DYNAMIC_TEMPLATE is True
    assert cfg.TEST.GPU_CROP is True


# ---------------------------------------------------------------- case 2
def test_02_sample_target_clamp():
    import numpy as np
    import torch

    from lib.train.data.processing_utils import sample_target, transform_image_to_crop

    im = (np.random.rand(500, 640, 3) * 255).astype(np.uint8)
    bb = torch.tensor([10.0, 10.0, 3.0, 5.0])  # 3x5 px target

    # search crop, min 64 px: crop_sz = max(64, ceil(sqrt(15)*6)=24) = 64
    crop, rf, att = sample_target(im, bb, 6.0, output_sz=256, min_crop_size=64.0)
    assert crop.shape[:2] == (256, 256)
    assert abs(rf - 256.0 / 64.0) < 1e-9, rf
    assert att.dtype == np.bool_ and att.shape == (256, 256)

    # template crop, min 32 px: crop_sz = max(32, ceil(sqrt(15)*2)=8) = 32
    _, rf_t, _ = sample_target(im, bb, 2.0, output_sz=256, min_crop_size=32.0)
    assert abs(rf_t - 256.0 / 32.0) < 1e-9, rf_t

    # unclamped large box keeps the legacy formula
    bb_big = torch.tensor([200.0, 150.0, 200.0, 100.0])
    _, rf_big, _ = sample_target(im, bb_big, 6.0, output_sz=256, min_crop_size=None)
    expected = 256.0 / math.ceil(6.0 * math.sqrt(200.0 * 100.0))
    assert abs(rf_big - expected) < 1e-9, (rf_big, expected)

    # gt box transformed into crop coordinates lands inside the crop
    for bb_i, rf_i in ((bb, rf), (bb_big, rf_big)):
        out = transform_image_to_crop(bb_i, bb_i, rf_i, torch.tensor([256.0, 256.0]))
        x, y, w, h = out.tolist()
        assert x >= 0.0 and y >= 0.0, out
        assert x + w <= 256.0 + 1e-4 and y + h <= 256.0 + 1e-4, out


# ---------------------------------------------------------------- case 3
def _legacy_jittered_box(box, mode, center_jitter_factor, scale_jitter_factor):
    """Original OSTrack/FocusTrack formula, computed inline for exact comparison."""
    jittered_size = box[2:4] * torch.exp(torch.randn(2) * scale_jitter_factor[mode])
    max_offset = jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor[mode]).float()
    jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)


def test_03_get_jittered_box():
    import torch

    from lib.train.data.processing import QpProcessing, STARKProcessing

    def make(cls):
        return cls(
            search_area_factor={"template": 2.0, "search": 6.0},
            output_sz=128,
            center_jitter_factor={"template": 3.0, "search": 4.5},
            scale_jitter_factor={"template": 0.3, "search": 0.0},
            mode="pair",
            settings=None,
            min_crop_size=64.0,
        )

    for cls in (STARKProcessing, QpProcessing):
        proc = make(cls)
        tiny = torch.tensor([100.0, 100.0, 3.0, 5.0])
        big = torch.tensor([500.0, 300.0, 500.0, 300.0])

        # search mode, tiny 3x5 box: floor 10.0 px effective scale, factor 2.5
        center = tiny[0:2] + 0.5 * tiny[2:4]
        offs = []
        for _ in range(200):
            jb = proc._get_jittered_box(tiny, "search")
            offs.append((jb[0:2] + 0.5 * jb[2:4] - center).abs())
        max_abs = torch.stack(offs).max().item()
        assert max_abs <= 0.5 * 2.5 * 10.0 + 1e-4, max_abs  # tight bound 12.5
        assert max_abs <= 2.5 * 10.0 + 1e-4, max_abs  # task-stated loose bound
        assert max_abs > 0.5 * 2.5 * math.sqrt(15.0) + 1.0, max_abs  # floor engaged (legacy cap ~4.84)

        # search mode, 500x300 box: reduced factor 1.8, never the 2.5 branch
        center_b = big[0:2] + 0.5 * big[2:4]
        scale_b = math.sqrt(500.0 * 300.0)
        offs_b = []
        for _ in range(200):
            jb = proc._get_jittered_box(big, "search")
            offs_b.append((jb[0:2] + 0.5 * jb[2:4] - center_b).abs())
        max_abs_b = torch.stack(offs_b).max().item()
        assert max_abs_b <= 0.5 * 1.8 * scale_b + 1e-3, max_abs_b  # any 2.5-branch draw near the edge violates this
        assert max_abs_b > 100.0, max_abs_b  # meaningful spread -> jitter really active
        assert max_abs_b >= 0.5 * 10.0, max_abs_b  # sanity: floor lower bound branch

        # template mode == legacy formula exactly (same RNG stream, same seed)
        for seed in (0, 7, 1234):
            box = torch.tensor([42.0, 24.0, 37.0, 80.0])
            torch.manual_seed(seed)
            adapted = proc._get_jittered_box(box, "template")
            torch.manual_seed(seed)
            legacy = _legacy_jittered_box(
                box, "template",
                {"template": proc.center_jitter_factor["template"]},
                {"template": proc.scale_jitter_factor["template"]},
            )
            assert torch.allclose(adapted, legacy, atol=1e-6), (adapted, legacy)


# ---------------------------------------------------------------- case 4
def test_04_buffered_giou():
    import torch

    from lib.utils.box_ops import (
        _buffer_boxes,
        buffered_giou_loss,
        generalized_box_iou,
        generalized_buffered_box_iou,
    )

    # 3x5-px-equivalent normalized target
    a = torch.tensor([[0.40, 0.40, 0.40 + 3.0 / 256.0, 0.40 + 5.0 / 256.0]])
    giou_self, iou_self = generalized_buffered_box_iou(a, a)
    assert abs(iou_self.item() - 1.0) < 1e-6
    assert abs(giou_self.item() - 1.0) < 1e-6

    # 1-px shift in x (1/256 normalized)
    b = a.clone()
    b[0, 0] += 1.0 / 256.0
    b[0, 2] += 1.0 / 256.0
    std_giou, _ = generalized_box_iou(a, b)
    buf_giou, _ = generalized_buffered_box_iou(a, b)
    std_drop = 1.0 - std_giou.item()
    buf_drop = 1.0 - buf_giou.item()
    assert buf_drop < 0.5 * std_drop, (buf_drop, std_drop)

    # invariants: x2 >= x1 kept, centers preserved, pads respect ratio/min
    buf = _buffer_boxes(a, 0.2, 0.015)
    assert (buf[:, 2:] >= buf[:, :2]).all()
    c_orig = (a[:, :2] + a[:, 2:]) / 2.0
    c_buf = (buf[:, :2] + buf[:, 2:]) / 2.0
    assert torch.allclose(c_orig, c_buf, atol=1e-7)
    pad_w = a[0, 0].item() - buf[0, 0].item()
    pad_h = a[0, 1].item() - buf[0, 1].item()
    assert abs(pad_w - 0.015) < 1e-6 and abs(pad_h - 0.015) < 1e-6  # min_buffer dominates tiny boxes (f32 tol)
    big = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    buf_big = _buffer_boxes(big, 0.2, 0.015)
    assert abs((buf_big[0, 2] - buf_big[0, 0]).item() - 0.4 * (1.0 + 2 * 0.2)) < 1e-6  # ratio dominates big boxes (f32 tol)
    # also non-degenerate self-consistency: buffered iou == plain iou for a big box vs itself
    g_self, _ = generalized_buffered_box_iou(big, big)
    assert abs(g_self.item() - 1.0) < 1e-6
    # loss API sanity
    loss, iou = buffered_giou_loss(a, b, 0.2, 0.015)
    assert 0.0 <= iou.item() <= 1.0 and torch.isfinite(loss).all()
    assert abs(loss.item() - buf_drop) < 1e-6


# ---------------------------------------------------------------- case 5
def test_05_patch_embed_overlapping():
    import torch

    from lib.models.focustrack.base_backbone import BaseBackbone
    from lib.models.layers.patch_embed import PatchEmbed

    x = torch.zeros(1, 3, 256, 256)
    pe = PatchEmbed(patch_size=16, stride=12, padding=2, embed_dim=64)
    assert pe(x).shape == (1, 21 * 21, 64), pe(x).shape
    pe = PatchEmbed(patch_size=16, stride=8, padding=4, embed_dim=64)
    assert pe(x).shape == (1, 32 * 32, 64), pe(x).shape
    pe = PatchEmbed(patch_size=16, embed_dim=64)  # default stride 16
    assert pe(x).shape == (1, 16 * 16, 64), pe(x).shape

    def make_bb():
        bb = BaseBackbone()
        bb.embed_dim = 768
        bb.img_size = (224, 224)
        bb.patch_size = 16
        bb.add_cls_token = True
        bb.patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        bb.pos_embed = torch.nn.Parameter(torch.randn(1, 197, 768))
        return bb

    def stub_cfg(stride):
        return D(
            DATA=D(SEARCH=D(SIZE=256), TEMPLATE=D(SIZE=128)),
            MODEL=D(
                BACKBONE=D(PATCH_SIZE=16, STRIDE=stride, CAT_MODE="direct", SEP_SEG=False),
                RETURN_INTER=False,
                RETURN_STAGES=[11],
            ),
        )

    # STRIDE 12: 256//12 = 21 -> 441 search tokens; 128//12 = 10 -> 100 template tokens
    bb = make_bb()
    bb.finetune_track(stub_cfg(12))  # conv-grid assert inside must pass
    assert tuple(bb.pos_embed_x.shape) == (1, 441, 768), bb.pos_embed_x.shape
    assert tuple(bb.pos_embed_z.shape) == (1, 100, 768), bb.pos_embed_z.shape
    assert bb.num_tokens_all == [1, 100, 441]

    # STRIDE 16 (default): unchanged behaviour, 256 tokens
    bb16 = make_bb()
    bb16.finetune_track(stub_cfg(16))
    assert tuple(bb16.pos_embed_x.shape) == (1, 256, 768), bb16.pos_embed_x.shape
    assert tuple(bb16.pos_embed_z.shape) == (1, 64, 768), bb16.pos_embed_z.shape
    assert bb16.num_tokens_all == [1, 64, 256]


# ---------------------------------------------------------------- case 6
def test_06_model_build_forward():
    import torch

    from lib.models.focustrack.focustrack import build_focustrack

    assert torch.cuda.is_available(), "CUDA required for this case"
    dev = "cuda"

    cfg = fresh_cfg("focustrack_ftir_stage2.yaml")  # stride 12 -> 21x21 map
    model = build_focustrack(cfg, training=False).to(dev).eval()
    with torch.no_grad():
        out = model(
            torch.randn(1, 3, 128, 128, device=dev),
            torch.randn(1, 3, 256, 256, device=dev),
        )
    assert out["score_map"].shape[-2:] == (21, 21), out["score_map"].shape
    assert tuple(out["pred_boxes"].shape) == (1, 1, 4), out["pred_boxes"].shape
    assert tuple(out["logits"].shape) == (1, 2), out["logits"].shape
    assert torch.isfinite(out["pred_boxes"]).all()
    del model, out

    cfg = fresh_cfg("runway_sra.yaml")  # stride 16 -> 16x16 map
    model = build_focustrack(cfg, training=False).to(dev).eval()
    with torch.no_grad():
        out = model(
            torch.randn(1, 3, 128, 128, device=dev),
            torch.randn(1, 3, 256, 256, device=dev),
        )
    assert out["score_map"].shape[-2:] == (16, 16), out["score_map"].shape
    assert tuple(out["pred_boxes"].shape) == (1, 1, 4), out["pred_boxes"].shape
    assert tuple(out["logits"].shape) == (1, 2), out["logits"].shape


# ---------------------------------------------------------------- case 7
def test_07_tir_stacking():
    import numpy as np
    import torch

    from lib.utils.tir_stack import TIRFrameStacker, stack_pair, stack_pair_gpu

    rng = np.random.RandomState(0)
    a = rng.randint(0, 256, (64, 80)).astype(np.uint8)
    b = rng.randint(0, 256, (64, 80)).astype(np.uint8)

    s = stack_pair(a, b)
    assert s.shape == (64, 80, 3) and s.dtype == np.uint8
    assert np.array_equal(s[:, :, 0], a) and np.array_equal(s[:, :, 1], b)
    assert int(s[:, :, 2].max()) == 255  # diff channel peak-normalized
    s0 = stack_pair(a, a)
    assert s0[:, :, 2].max() == 0  # identical pair -> zero diff
    a3 = np.stack([a] * 3, axis=-1)
    s3 = stack_pair(a3, b)
    assert np.array_equal(s3[:, :, 0], a)  # 3-channel input uses channel 0

    assert torch.cuda.is_available()
    cur = torch.from_numpy(a.astype(np.float32))[None].cuda()   # (1,H,W)
    prev = torch.from_numpy(b.astype(np.float32))[None].cuda()
    g = stack_pair_gpu(cur, prev)
    assert g.shape == (3, 64, 80) and g.is_cuda
    ref = torch.from_numpy(s.astype(np.float32)).permute(2, 0, 1)  # stack_pair is HWC, stack_pair_gpu is CHW
    assert torch.allclose(g.cpu(), ref, atol=1.5), "GPU/CPU stacking parity"
    g0 = stack_pair_gpu(cur, cur)
    assert g0[2].abs().max().item() == 0.0

    st = TIRFrameStacker(delta=1)
    first = st.stack(a)
    assert first[:, :, 2].max() == 0  # first call: prev == cur -> zero diff
    second = st.stack(b)
    assert second[:, :, 2].max() > 0
    st.reset()
    assert len(st._history) == 0


# ---------------------------------------------------------------- case 8
def test_08_gpu_sample_target():
    import numpy as np
    import torch

    from lib.test.tracker.gpu_crop import gpu_sample_target
    from lib.train.data.processing_utils import sample_target

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    frame = torch.rand(3, 1080, 1920, device="cuda") * 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    bb = [960.0, 540.0, 3.0, 5.0]

    img, rf, att = gpu_sample_target(frame, bb, 6.0, 256, 64.0, mean, std)
    assert tuple(img.shape) == (1, 3, 256, 256), img.shape
    assert img.is_cuda and img.dtype == torch.float32
    assert abs(rf - 4.0) < 1e-9, rf  # 256 / max(64, ceil(sqrt(15)*6))
    assert att.dtype == torch.bool and att.shape == (256, 256)
    assert not att.any()  # box well inside the frame -> no padding region
    assert torch.isfinite(img).all()

    im_np = frame.permute(1, 2, 0).cpu().numpy()  # HxWx3 float32, 0..255
    crop, rf_cpu, att_cpu = sample_target(im_np, bb, 6.0, output_sz=256, min_crop_size=64.0)
    assert abs(rf_cpu - 4.0) < 1e-9, rf_cpu
    assert abs(float(rf) - float(rf_cpu)) < 1e-9  # CPU/GPU resize_factor agreement
    assert att_cpu.dtype == np.bool_ and att_cpu.shape == (256, 256)
    assert not att_cpu.any()


# ---------------------------------------------------------------- case 9
def test_09_import_surface():
    import lib.test.tracker.focustrack as tracker_mod  # module import only, no checkpoint
    import lib.train.data.sampler  # noqa: F401
    import lib.train.dataset as ds

    assert tracker_mod is not None
    assert hasattr(ds, "LSOTB_TIR") and hasattr(ds, "RunwayTIR")

    from lib.train.base_functions import names2datasets

    settings = D(
        use_lmdb=False,
        env=D(
            lsotb_tir_dir="/tmp/__nonexistent_lsotb_dir",
            runway_tir_dir="/tmp/__nonexistent_runway_dir",
        ),
    )
    for name in ("LSOTB_TIR", "RunwayTIR"):
        raised = None
        try:
            names2datasets([name], settings, lambda p: None)
        except Exception as e:  # noqa: BLE001
            raised = e
        # A clean missing-directory error (or no error at all) is acceptable;
        # NameError/ImportError would mean a broken wiring of the new datasets.
        if raised is not None:
            assert not isinstance(raised, (NameError, ImportError)), repr(raised)
            assert isinstance(raised, OSError), (
                f"{name}: expected clean missing-dir error, got {type(raised).__name__}: {raised}"
            )


# ---------------------------------------------------------------- case 10
def test_10_actor_buffered_loss_stub():
    import torch

    from lib.utils.box_ops import box_cxcywh_to_xyxy, buffered_giou_loss, giou_loss

    # Minimal stub mirroring FocusTrackActor.compute_losses (actors/focustrack.py,
    # buffered branch ~lines 94-99): predictions come as cxcywh and are compared
    # in xyxy with buffered_giou_loss(buffer_ratio, min_buffer) from the config.
    cfg = fresh_cfg("runway_sra.yaml")
    assert cfg.TRAIN.BUFFERED_IOU is True

    pred_cxcywh = torch.tensor([[0.51, 0.51, 3.0 / 256.0, 5.0 / 256.0]], requires_grad=True)
    gt_cxcywh = torch.tensor([[0.511, 0.511, 3.0 / 256.0, 5.0 / 256.0]])
    pred_vec = box_cxcywh_to_xyxy(pred_cxcywh)
    gt_vec = box_cxcywh_to_xyxy(gt_cxcywh)

    if cfg.TRAIN.BUFFERED_IOU:
        giou_loss_vec, iou = buffered_giou_loss(
            pred_vec, gt_vec,
            buffer_ratio=cfg.TRAIN.BUFFER_RATIO, min_buffer=cfg.TRAIN.MIN_BUFFER,
        )
    else:  # pragma: no cover
        giou_loss_vec, iou = giou_loss(pred_vec, gt_vec)
    positive_mask = torch.ones(1)
    loss = (giou_loss_vec * positive_mask).sum() / positive_mask.sum()
    weighted = loss * cfg.TRAIN.GIOU_WEIGHT

    assert torch.isfinite(weighted).all() and weighted.item() > 0.0
    assert 0.0 <= iou.item() <= 1.0
    weighted.backward()
    assert pred_cxcywh.grad is not None
    assert torch.isfinite(pred_cxcywh.grad).all()
    assert pred_cxcywh.grad.abs().sum().item() > 0.0  # gradient actually flows


# ---------------------------------------------------------------- case 11
def test_11_full_ckpt_stride_transition():
    import tempfile
    import torch

    from lib.models.focustrack.focustrack import build_focustrack

    # synthetic stage1 (STRIDE 16) checkpoint in the trainer's {"net": ...} format
    cfg1 = fresh_cfg("focustrack_ftir_stage1.yaml")
    m1 = build_focustrack(cfg1, training=False)
    sd1 = m1.state_dict()
    assert tuple(sd1["backbone.pos_embed_z"].shape) == (1, 64, 768)
    assert tuple(sd1["backbone.pos_embed_x"].shape) == (1, 256, 768)
    assert tuple(sd1["cls_head.avgpool.positional_embedding"].shape) == (257, 768)

    # pre-fix FULL_CKPT behaviour: strict=False still raises on the size mismatch
    cfg2 = fresh_cfg("focustrack_ftir_stage2.yaml")
    m2_ref = build_focustrack(cfg2, training=False)
    raised = None
    try:
        m2_ref.load_state_dict(sd1, strict=False)
    except RuntimeError as e:  # noqa: BLE001
        raised = e
    assert raised is not None and "size mismatch" in str(raised), repr(raised)

    # full-ckpt init across the STRIDE 16 -> 12 stage transition must build cleanly
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "FocusTrack_ep0015.pth")
        torch.save({"net": sd1}, ckpt_path)
        cfg2.TRAIN.FULL_CKPT = ckpt_path
        m2 = build_focustrack(cfg2, training=True)

    sd2 = m2.state_dict()
    assert tuple(sd2["backbone.pos_embed_z"].shape) == (1, 100, 768)
    assert tuple(sd2["backbone.pos_embed_x"].shape) == (1, 441, 768)
    assert tuple(sd2["cls_head.avgpool.positional_embedding"].shape) == (442, 768)
    assert torch.isfinite(sd2["backbone.pos_embed_x"]).all()

    # inline golden reference: bicubic 16x16 -> 21x21 resample of the search grid
    ref = sd1["backbone.pos_embed_x"].reshape(1, 16, 16, 768).permute(0, 3, 1, 2)
    ref = torch.nn.functional.interpolate(ref, size=(21, 21), mode="bicubic", align_corners=False)
    ref = ref.flatten(2).permute(0, 2, 1)
    assert torch.allclose(sd2["backbone.pos_embed_x"], ref, atol=1e-5)
    # cls-head embedding: cls row copied verbatim, grid part resampled 16x16 -> 21x21
    assert torch.equal(sd2["cls_head.avgpool.positional_embedding"][0],
                       sd1["cls_head.avgpool.positional_embedding"][0])
    cls_ref = sd1["cls_head.avgpool.positional_embedding"][1:].reshape(16, 16, 768).permute(2, 0, 1)
    cls_ref = torch.nn.functional.interpolate(cls_ref[None], size=(21, 21), mode="bicubic",
                                              align_corners=False)
    cls_ref = cls_ref.flatten(2).permute(0, 2, 1).reshape(441, 768)
    assert torch.allclose(sd2["cls_head.avgpool.positional_embedding"][1:], cls_ref, atol=1e-5)

    # grid pos embeds were resampled, not dropped; every non-grid weight copied verbatim
    for key in ("backbone.patch_embed.proj.weight",
                "backbone.blocks.0.attn.qkv.weight",
                "cls_head.mlp_head.0.weight",
                "box_head.conv1_ctr.0.weight"):
        assert torch.equal(sd1[key], sd2[key]), key


CASES = [
    ("01_config_update", test_01_config_update),
    ("02_sample_target_clamp", test_02_sample_target_clamp),
    ("03_get_jittered_box", test_03_get_jittered_box),
    ("04_buffered_giou", test_04_buffered_giou),
    ("05_patch_embed_overlapping", test_05_patch_embed_overlapping),
    ("06_model_build_forward", test_06_model_build_forward),
    ("07_tir_stacking", test_07_tir_stacking),
    ("08_gpu_sample_target", test_08_gpu_sample_target),
    ("09_import_surface", test_09_import_surface),
    ("10_actor_buffered_loss_stub", test_10_actor_buffered_loss_stub),
    ("11_full_ckpt_stride_transition", test_11_full_ckpt_stride_transition),
]


def main():
    failures = []
    for name, fn in CASES:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception:
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
            failures.append(name)
    print(flush=True)
    if failures:
        print(f"SUMMARY: {len(failures)} FAILED: {failures}", flush=True)
        return 1
    print("SUMMARY: ALL {} CASES PASS".format(len(CASES)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
