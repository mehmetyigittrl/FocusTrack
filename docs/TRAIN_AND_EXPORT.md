# Training & TensorRT Export Guide

How to train the runway-adapted FocusTrack (SRA-only / FTIR) and export a
trained checkpoint to a TensorRT FP16 engine via NVIDIA ModelOpt.
Every command here was run and verified on this machine (RTX 5060 8 GB,
conda env `gzs_env_2404`, TensorRT 11.2).

## Contents
1. [Prerequisites](#1-prerequisites)
2. [Starting a training](#2-starting-a-training)
3. [Exporting to TensorRT (NVIDIA ModelOpt)](#3-exporting-to-tensorrt-nvidia-modelopt)
4. [Deploying on the offline / target machine](#4-deploying-on-the-offline--target-machine)

---

## 1. Prerequisites

| Item | Where / how |
|---|---|
| conda env `gzs_env_2404` | Python 3.13, PyTorch + CUDA, TensorRT already installed |
| DropMAE backbone weights | `pretrained_models/dropmae_pretrained_vit-B-16_k700_800E.pth` (referenced by `MODEL.PRETRAIN_FILE` in the yamls) |
| AntiUAV410-style dataset | `data/antiuav410/train/<seq>/` frames + `IR_label.json` (`gt_rect`, `exist`). Convert from COCO with `tools/coco2antiuav410.py` |
| `libturbojpeg` for the data loader | exported via `LD_LIBRARY_PATH` (see commands below) |
| FTIR stages only | fill `lsotb_tir_dir` / `runway_tir_dir` in `lib/train/admin/local.py`, and point `TRAIN.FULL_CKPT` in `experiments/focustrack/focustrack_ftir_stage{1,2}.yaml` at real checkpoints |

Run everything from the repo root.

## 2. Starting a training

### 2.1 Smoke test first (recommended after any env change)

```bash
PYTHONPATH=. LD_LIBRARY_PATH=/home/pc_9211/.local/lib timeout 900 \
conda run --no-capture-output -n gzs_env_2404 \
python tracking/train.py --script focustrack --config runway_sra_smoke \
  --save_dir /tmp/focustrack_smoke_$$ --mode single --use_wandb 0
```

`runway_sra_smoke.yaml` = 1 epoch, batch 2, 20 samples/epoch — finishes in a
few minutes and exercises the whole pipeline (data load, SRA losses,
checkpoint save).

### 2.2 Real training (RGB, SRA-only)

```bash
PYTHONPATH=. LD_LIBRARY_PATH=/home/pc_9211/.local/lib \
conda run --no-capture-output -n gzs_env_2404 \
python tracking/train.py --script focustrack --config runway_sra \
  --save_dir ./output --mode single --use_wandb 0
```

- Config: `experiments/focustrack/runway_sra.yaml` — 30 epochs, batch 64,
  AntiUAV410 train/val. Lower `TRAIN.BATCH_SIZE` first if the GPU runs out of
  memory (8 GB needs a small batch + gradient accumulation or a swap).
- Checkpoints land in `output/checkpoints/train/focustrack/<config>/` as
  `FocusTrack_ep0030.pth.tar`. Every epoch is written, but older ones are
  deleted as training progresses: only `TRAIN.SAVE_INTERVAL` multiples (10,
  20), `TRAIN.EXTRA_SAVE_EPOTH` (29, 30) and the final epochs survive — for
  runway_sra that's ep0010, ep0020, ep0025–ep0030 (8 files; each holds `net`,
  `optimizer`, `stats`).
- TensorBoard logs: `tensorboard/` (env `tensorboard_dir` in `local.py`).
- **Resume**: the trainer automatically resumes from the newest
  `FocusTrack_ep*.pth.tar` it finds in the save dir. Point `--save_dir` at a
  **fresh** directory when you want a clean run (that's why the smoke command
  uses `/tmp/focustrack_smoke_$$`).
- Multi-GPU: replace `--mode single` with `--mode multiple --nproc_per_node 4`
  (see `tracking/train_ftir.sh`).

### 2.3 Thermal fine-tune (two-stage FTIR)

1. Edit `experiments/focustrack/focustrack_ftir_stage1.yaml` → set
   `TRAIN.FULL_CKPT` to your trained runway_sra checkpoint (a placeholder path
   is pre-filled), and likewise in `focustrack_ftir_stage2.yaml` for the
   stage-1 result.
2. Make sure `lsotb_tir_dir` / `runway_tir_dir` in `lib/train/admin/local.py`
   point at the datasets.
3. Run both stages:

```bash
NGPU=1 bash tracking/train_ftir.sh
```

Stage 2 switches the backbone to an overlapping patch embedding
(`STRIDE 12`); `tools/export_trt.py` handles the changed feature-map size
automatically (16×16 maps become 21×21).

### 2.4 Known gotchas

- `tracking/train.py` launches the child with `os.system`, so the shell **exit
  code can be 0 even when training crashed** — always check the tail of the
  log before trusting a run.
- Yaml keys are strictly whitelisted by `lib/config/focustrack/config.py` —
  unknown keys raise. Spec-style names like `USE_ATM` don't exist; the real
  keys are e.g. `MODEL.DECODER.ADD_DECODER`, `TRAIN.L1_WEIGHT`.
- Regression suite for all runway adaptations:
  `PYTHONPATH=. conda run -n gzs_env_2404 python tests/test_runway_adaptation.py`

## 3. Exporting to TensorRT (NVIDIA ModelOpt)

`tools/export_trt.py` turns a checkpoint into an FP16 engine:

```
checkpoint → torch eval model → ONNX opset 19 (fp32)
           → NVIDIA ModelOpt convert_to_f16 (fp16 weights, fp32 IO)
           → TensorRT strongly-typed engine
           → parity check vs torch fp32 → FPS benchmark
```

### 3.1 Run it

```bash
# FP16 (default): picks the newest checkpoint for the config automatically
PYTHONPATH=. conda run --no-capture-output -n gzs_env_2404 \
python tools/export_trt.py --config runway_sra

# explicit checkpoint / fp32 / custom output dir
python tools/export_trt.py --config runway_sra \
  --checkpoint output/checkpoints/train/focustrack/runway_sra/FocusTrack_ep0030.pth.tar
python tools/export_trt.py --config runway_sra --dtype fp32
python tools/export_trt.py --config focustrack_ftir_stage2 --out-dir /tmp/exports
```

Artifacts land in `output/export/focustrack/<config>/`:

| File | Meaning |
|---|---|
| `<config>_fp32.onnx` | base graph, opset 19 (~382 MB) |
| `<config>_fp16.onnx` | ModelOpt FP16, fp32 IO (~192 MB) — portable across machines |
| `<config>_fp16.engine` | TensorRT engine (~195 MB) — **specific to the build GPU + TRT version** |

Useful flags: `--skip-bench`, `--skip-verify`, `--verify-seed`, `--opset`
(keep 19 — see below). If no checkpoint is found the tool warns and exports
random weights, so you can test the pipeline before any training.

### 3.2 Verified results (RTX 5060 8 GB, TensorRT 11.2, torch 2.x)

| Runtime | Latency | FPS | Parity vs torch fp32 (worst output) |
|---|---|---|---|
| TRT FP16 (ModelOpt) | 1.83 ms | **547** | 1.4e-2 (`logits`; maps ≤ 8e-3) |
| TRT FP32 | 5.21 ms | 192 | 2.6e-3 |
| torch tracker, fp16 (`TEST.FP16: True`) | — | ~270 | reference |
| torch tracker, fp32 | — | ~88 | reference |

The tool prints per-output `max_abs`/`max_rel` and a verdict; anything under
2 % worst relative error is `OK` (tracking accuracy is insensitive at this
level — verify end-to-end if you see `WARN`).

### 3.3 Engine input/output contract

The engine is the **network forward only** — stateless, one call per frame.

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `template` (in) | `1×3×128×128` | fp32 | current template crop (initial box or refreshed dynamic template) |
| `search` (in) | `1×3×256×256` | fp32 | search-region crop |
| `score_map` (out) | `1×1×16×16` | fp32 | confidence map → `cal_bbox` decode |
| `size_map` (out) | `1×2×16×16` | fp32 | w/h regression |
| `offset_map` (out) | `1×2×16×16` | fp32 | center offset |
| `logits` (out) | `1×2` | fp32 | presence logits → sigmoid feeds the SRA escalation gate |

- Inputs must be **normalized with the training statistics**
  (`DATA.MEAN [0.485, 0.456, 0.406]`, `DATA.STD [0.229, 0.224, 0.225]`)
  *before* being fed to the engine — the engine does not normalize.
- The tracker's `pred_boxes` output is not exported: the host decodes boxes
  from the maps (`cal_bbox`), same as the PyTorch tracker.
- Everything stateful stays host-side: search/template cropping (GPU crop via
  `grid_sample` in the tracker), TIR temporal stacking, hann windowing,
  SRA escalation, dynamic-template refresh.
- FTIR stage-2 engines have `21×21` maps instead of `16×16` — read shapes
  from the engine context (`get_tensor_shape`), don't hardcode.

Minimal inference:

```python
import tensorrt as trt, torch
engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(open("runway_sra_fp16.engine", "rb").read())
ctx = engine.create_execution_context()
z, x = template_crop.cuda(), search_crop.cuda()          # normalized, fp32
ctx.set_input_shape("template", tuple(z.shape)); ctx.set_input_shape("search", tuple(x.shape))
outs = {n: torch.empty(tuple(ctx.get_tensor_shape(n)), device="cuda") for n in
        ("score_map", "size_map", "offset_map", "logits")}
for name, t in {"template": z, "search": x, **outs}.items():
    ctx.set_tensor_address(name, t.data_ptr())
ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
```

### 3.4 Why opset 19 and "strongly typed" (context, not required reading)

- TensorRT ≥ 11 removed `BuilderFlag.FP16`; engines are built as
  **strongly-typed** networks and the precision comes from the ONNX tensor
  dtypes — which is exactly what ModelOpt's `convert_to_f16(..., keep_io_types=True)`
  produces.
- ModelOpt raises older opsets through the onnx C-API version converter, and
  that converter **silently corrupts this particular graph** (17→19 gave ~50 %
  output deviation). Exporting the base ONNX at opset 19 directly bypasses the
  conversion entirely. Don't lower `--opset`.
- `onnxconverter-common`'s float16 pass is *not* used — its mixed Half/Float
  graph fails TRT parsing on the cls-head avgpool `Div`.

## 4. Deploying on the offline / target machine

The export tool is part of the repo, so the whole pipeline travels with it.

1. **Copy over**: the repo (at least `lib/`, `experiments/`, `tools/`) + the
   checkpoint(s) you want, placed at
   `output/checkpoints/train/focustrack/<config>/FocusTrack_ep*.pth.tar` —
   that's where `--config`-only invocations look for them. Checkpoints kept
   anywhere else need an explicit `--checkpoint` (otherwise the tool warns and
   exports random weights). The portable artifact is the `.onnx`; the
   `.engine` should be built on the machine/GPU that will run it.
2. **Install export deps**. The target env needs the same runtime the model
   code imports (torch + CUDA, torchvision, timm, matplotlib, pyyaml,
   easydict, numpy — a clone of the training env is the easy way) plus
   TensorRT (≥ 10.4; 11.x tested) and the ModelOpt/ONNX stack. On a networked
   machine:

   ```bash
   uv pip install "nvidia-modelopt[onnx]" onnx onnxscript
   # if modelopt imports still fail, add its pulled-in deps explicitly:
   uv pip install onnx_graphsurgeon onnxslim onnxruntime-gpu onnxconverter-common
   ```

   On the offline box itself `uv pip install` can't reach PyPI — download
   wheels on a networked machine and carry the directory over:

   ```bash
   # networked machine (copy this dir alongside the repo):
   python -m pip download -d wheels/ "nvidia-modelopt[onnx]" onnx onnxscript
   # offline machine:
   uv pip install --no-index --find-links wheels/ "nvidia-modelopt[onnx]" onnx onnxscript
   ```
3. **Export there** (same commands as §3.1 — watch for the random-weights
   warning if the checkpoint isn't found). If the GPU differs from the
   training box, that's fine — the engine is built on the machine you run the
   export on, which is the point.
4. **Sanity-check** before integrating: the tool's built-in parity check
   (`--skip-verify` off) compares the engine against the torch model on the
   same GPU; a quick FPS read comes from the built-in bench.

> INT8/FP8 quantization is not wired up yet. ModelOpt supports it
> (`modelopt.onnx.quantization.quantize` with calibration data) if you need to
> go below FP16 latency later.
