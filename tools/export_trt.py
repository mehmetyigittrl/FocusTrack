"""Export a FocusTrack (SRA-only) eval model to TensorRT.

Pipeline (NVIDIA ModelOpt + TensorRT, both installed in the training env):

  checkpoint --> torch eval model --> ONNX (opset 19)  \
      dtype=fp16: ModelOpt autocast convert_to_f16      --> TRT strongly-typed engine
                                                        --> parity check vs torch fp32
                                                        --> speed benchmark

Usage (from the repo root):

  python tools/export_trt.py --config runway_sra                       # fp16, latest checkpoint
  python tools/export_trt.py --config runway_sra --dtype fp32
  python tools/export_trt.py --config focustrack_ftir_stage2 --checkpoint output/.../FocusTrack_ep0010.pth.tar
  PYTHONPATH=. conda run -n gzs_env_2404 python tools/export_trt.py --config runway_sra

Notes
-----
* TensorRT >= 11 builds only strongly-typed networks; FP16 therefore comes from
  the ONNX tensor dtypes (ModelOpt's convert_to_f16), not from a builder flag.
* The base ONNX is exported at opset 19 on purpose: ModelOpt raises older
  opsets via the onnx C-API version converter, and the 17->19 conversion of
  THIS graph silently corrupts it (outputs deviate ~50% from torch). Exporting
  at opset 19 directly avoids the rewrite. Do not lower --opset for fp16.
* Engines are specific to the GPU and TensorRT version they were built on.
  Build the engine on the deployment machine; carry the ONNX across machines.
* The engine covers only the network forward. Cropping, TIR stacking, hann
  windowing, cal_bbox decode, SRA escalation and template refresh stay in the
  tracker (host side).
"""
import argparse
import glob
import os
import sys
import time

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

OUTPUT_NAMES = ("score_map", "size_map", "offset_map", "logits")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="yaml name under experiments/focustrack (e.g. runway_sra)")
    p.add_argument("--checkpoint", default=None,
                   help="FocusTrack_ep*.pth.tar; default: latest under output/checkpoints/train/focustrack/<config>/")
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--opset", type=int, default=19,
                   help="ONNX opset of the base export (keep 19: see module docstring)")
    p.add_argument("--out-dir", default=None,
                   help="default: output/export/focustrack/<config>")
    p.add_argument("--skip-verify", action="store_true", help="skip the parity check")
    p.add_argument("--skip-bench", action="store_true", help="skip the speed benchmark")
    p.add_argument("--verify-seed", type=int, default=7)
    return p.parse_args()


def find_latest_checkpoint(config):
    pattern = os.path.join(REPO_ROOT, "output", "checkpoints", "train", "focustrack", config, "FocusTrack_ep*.pth.tar")
    ckpts = sorted(glob.glob(pattern))
    return ckpts[-1] if ckpts else None


def build_model(config, checkpoint):
    import lib.config.focustrack.config as config_mod
    config_mod.update_config_from_file(os.path.join(REPO_ROOT, "experiments", "focustrack", config + ".yaml"))
    cfg = config_mod.cfg
    cfg.MODEL.PRETRAIN_FILE = ""
    from lib.models.focustrack import build_focustrack
    net = build_focustrack(cfg, training=False).cuda().eval()
    if checkpoint:
        # build_focustrack only consumes FULL_CKPT while training; load it here.
        from lib.models.focustrack.focustrack import _adapt_full_ckpt
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        state = _adapt_full_ckpt(net, state)  # resample pos embeds if stride differs (FTIR stage2)
        missing, unexpected = net.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith(("cls_token",))]
        if missing or unexpected:
            print(f"[model] WARNING: missing={missing} unexpected={unexpected}")
        print(f"[model] loaded weights from {checkpoint}")
    return net, cfg


class ExportWrapper(nn.Module):
    """Tracker-facing output contract: the four tensors consumed per frame."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, template, search):
        out = self.m(template, search, training=False)
        return tuple(out[n] for n in OUTPUT_NAMES)


def export_onnx(net, template, search, path, opset):
    torch.onnx.export(
        ExportWrapper(net).eval(), (template, search), path,
        input_names=["template", "search"], output_names=list(OUTPUT_NAMES),
        opset_version=opset, dynamo=False,
    )
    print(f"[export] fp32 ONNX (opset {opset}) -> {path}")


def modelopt_fp16(onnx_path, out_path):
    import onnx
    from modelopt.onnx.autocast import convert_to_f16
    model = convert_to_f16(onnx.load(onnx_path), keep_io_types=True)
    onnx.save(model, out_path)
    print(f"[modelopt] fp16 ONNX (fp32 IO) -> {out_path}")


def build_engine(onnx_path, out_path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT >= 11: strongly-typed only; precision follows the ONNX tensor dtypes.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(open(onnx_path, "rb").read()):
        for i in range(parser.num_errors):
            print("[trt] parse error:", parser.get_error(i))
        raise SystemExit("TensorRT failed to parse the ONNX model")
    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 5 << 30)
    except Exception:
        pass
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("TensorRT engine build failed")
    open(out_path, "wb").write(engine)
    print(f"[trt] engine built in {time.time() - t0:.1f}s -> {out_path} ({engine.nbytes / 1e6:.1f} MB)")


def make_context(engine_path, template, search):
    """Deserialize the engine, allocate torch-backed IO buffers, bind addresses."""
    import tensorrt as trt
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(open(engine_path, "rb").read())
    ctx = engine.create_execution_context()
    ctx.set_input_shape("template", tuple(template.shape))
    ctx.set_input_shape("search", tuple(search.shape))
    tensors = {"template": template.contiguous(), "search": search.contiguous()}
    for name in OUTPUT_NAMES:
        tensors[name] = torch.empty(tuple(ctx.get_tensor_shape(name)), device="cuda", dtype=torch.float32)
    for name, t in tensors.items():
        ctx.set_tensor_address(name, t.data_ptr())
    return ctx, tensors


def main():
    args = parse_args()

    checkpoint = args.checkpoint or find_latest_checkpoint(args.config)
    if checkpoint:
        print(f"[model] checkpoint: {checkpoint}")
    else:
        print("[model] WARNING: no checkpoint found, exporting RANDOM weights (pipeline test only)")

    net, cfg = build_model(args.config, checkpoint)
    t_size, s_size = cfg.TEST.TEMPLATE.SIZE, cfg.TEST.SEARCH.SIZE
    template = torch.randn(1, 3, t_size, t_size, device="cuda")
    search = torch.randn(1, 3, s_size, s_size, device="cuda")

    out_dir = args.out_dir or os.path.join(REPO_ROOT, "output", "export", "focustrack", args.config)
    os.makedirs(out_dir, exist_ok=True)
    onnx_fp32 = os.path.join(out_dir, f"{args.config}_fp32.onnx")
    onnx_dtype = os.path.join(out_dir, f"{args.config}_{args.dtype}.onnx")
    engine_path = os.path.join(out_dir, f"{args.config}_{args.dtype}.engine")

    export_onnx(net, template, search, onnx_fp32, args.opset)
    if args.dtype == "fp16":
        try:
            modelopt_fp16(onnx_fp32, onnx_dtype)
        except ImportError as e:
            raise SystemExit(
                f"fp16 export needs NVIDIA ModelOpt ({e}).\n"
                'Install with:  uv pip install "nvidia-modelopt[onnx]" onnx onnxscript'
            )
    build_engine(onnx_dtype, engine_path)

    if not args.skip_verify:
        torch.manual_seed(args.verify_seed)
        vt = torch.randn(1, 3, t_size, t_size, device="cuda")
        vs = torch.randn(1, 3, s_size, s_size, device="cuda")
        with torch.no_grad():
            ref = net(vt, vs, training=False)
        ctx, tensors = make_context(engine_path, vt, vs)
        stream = torch.cuda.current_stream()
        ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        print(f"[verify] engine({args.dtype}) vs torch fp32 (seed {args.verify_seed}):")
        worst = 0.0
        for name in OUTPUT_NAMES:
            d = (tensors[name] - ref[name]).abs()
            rel = d.max().item() / max(ref[name].abs().max().item(), 1e-9)
            worst = max(worst, rel)
            print(f"   {name:10s} max_abs={d.max().item():.3e}  max_rel={rel:.3e}")
        verdict = "OK" if worst < 0.02 else ("WARN (check end-to-end tracking accuracy)" if worst < 0.05 else "FAIL")
        print(f"[verify] {verdict} (worst max_rel={worst:.3e})")

    if not args.skip_bench:
        bt = torch.randn(1, 3, t_size, t_size, device="cuda")
        bs = torch.randn(1, 3, s_size, s_size, device="cuda")
        ctx, _ = make_context(engine_path, bt, bs)
        stream = torch.cuda.current_stream()
        for _ in range(50):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(300):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 300 * 1000
        print(f"[bench] TRT {args.dtype}: {ms:.2f} ms/frame = {1000.0 / ms:.1f} FPS (GPU-only)")

    print("[done] artifacts:")
    for p in (onnx_fp32, onnx_dtype, engine_path):
        print(f"   {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
