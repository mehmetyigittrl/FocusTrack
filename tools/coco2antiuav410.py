#!/usr/bin/env python3
"""Convert COCO-format detection datasets into the AntiUAV410 layout used by
FocusTrack (lib/train/dataset/antiuav410.py).

Target layout (verified against the repo's own loaders):

    {output}/{split}/{seq_name}/IR_label.json
        {"gt_rect": [[x, y, w, h], ...], "exist": [0/1, ...]}   (same length N)
    {output}/{split}/{seq_name}/000001.jpg ... 00000N.jpg
        6-digit zero-padded, 1-based, contiguous, real JPEG files

* Boxes are COCO [x, y, w, h], top-left, 0-indexed; floats are fine.
* Absent target -> exist 0 with box [0, 0, 0, 0] (loader validity test is
  w > 0 and h > 0).
* Sequences sit exactly one level under the split root (the train loader globs
  '{root}/*/IR_label.json'; the eval loader sorts *.jpg by int(stem)).
* meta_info.ini is optional and is NOT written.
* The trainer never samples a sequence with < 20 visible frames
  (lib/train/data/sampler.py: sample_seq_from_dataset requires
  len(visible) >= 20 and visible.sum() > 2*(search+template)); the converter
  warns about such sequences and --min-visible can drop them outright.

Two input modes are auto-detected from --input:

  per_video  Each subfolder of --input is one continuous video holding its own
             COCO json (any *.json with top-level "images" and "annotations",
             found recursively) plus its frames.  One json -> one sequence,
             named after the subfolder ({subfolder}_{jsonstem} when a
             subfolder holds several jsons).  Disk frames with no json entry
             become absent frames (exist 0) at their sorted position; they
             are merged into the frame list BEFORE the labels are computed,
             so gt_rect/exist can never desync (a warning fires whenever
             such frames interleave with json frames).
  global     Global COCO instances json(s) (e.g. stock COCO 2017:
             annotations/instances_train2017.json + train2017/) with no video
             structure: images are natural-sorted by file_name and chunked into
             --seq-len pseudo-sequences named {jsonbase}_seqNNNN (the
             'instances_' prefix of the json stem is dropped).

If both structures are present, per-video mode wins (noted in the manifest).

Robustness and re-run behavior:

* Frame decode verification: .jpg/.jpeg sources are decoded (cv2 preferred,
  PIL fallback, lazy imports) before being copied/linked, because those are
  transferred byte-for-byte.  ON by default; --no-verify-frames trusts the
  file extension instead (saves one decode per frame of I/O at the cost of
  possibly emitting frames the trainer cannot read).  An undecodable frame is
  dropped entirely — image file AND label row, keeping jpg count ==
  len(gt_rect) — with a warning listing the first few and a per-sequence
  manifest count (n_undecodable); --strict aborts on it instead.  Non-JPEG
  sources are always re-encoded to real JPEG, so a bad file fails loudly
  there regardless.
* json image width/height may be numbers or numeric strings ("640"); anything
  unparseable falls back to decoding the image, and an undecodable file is
  then treated like a missing frame (skip + warning) instead of aborting.
* --category resolves per json: a json that HAS categories but matches none
  of the given values is a hard error listing the valid ids/names (an empty
  filter would silently mark every frame exist=0 and hang the trainer's
  sampler); a json without a categories list warns ("filter not applied")
  and behaves as no filter; when several values are given, unmatched ones
  are warned about and ignored while valid ones apply.
* Corrupt/unparseable jsons, jsons that are not COCO instances files, and
  image-bearing subfolders without any readable COCO json are skipped with
  warnings and recorded in the manifest under "skipped_sources" (--strict
  aborts on the bad jsons instead), so whole videos can no longer vanish
  silently.
* Re-running into an existing --output reconciles it first: sequence dirs
  that the current run does not keep (dropped by --min-visible, renamed or
  deleted sources, re-filtered) and same-name dirs sitting in a split other
  than the planned one are deleted (recorded in the manifest under "pruned").
  The output root therefore always mirrors the latest run and a sequence can
  never exist in both train and val.
* The --output tree is excluded from ALL input scanning, so it may safely be
  nested inside --input (an upfront warning notes the exclusion); using the
  same directory for --input and --output is refused.
* --jobs N (default: min(16, cpu_count)) plans and writes sequences on a
  thread pool.  The dominant costs — per-frame decode verification, image
  file resolution and frame copying — release the GIL, so threads scale;
  cv2 is pinned to one internal thread per worker (many small decodes beat
  cv2's own all-core threading inside each call).  Sequence names are
  allocated serially and task results/warnings merge strictly in submission
  order, so the output tree, manifest and warning order are identical to
  --jobs 1; --jobs 1 restores the fully serial behavior.  One deliberate
  difference on the FAILURE path only: with --jobs > 1 an aborted run
  (--strict error, failed hardlink/re-encode) may already have written
  sequences a serial run would never have reached — the exit code and the
  first reported error are identical, no manifest is written, and the next
  run's reconcile pass prunes the leftovers.

The tool is stdlib-first (json/argparse/pathlib/shutil/re).  cv2/PIL are
imported lazily only when an image actually has to be decoded: verification
of .jpg/.jpeg sources, the size fallback for json entries without usable
width/height, or re-encoding non-JPEG frames to real JPEG.

Examples:
    python3 tools/coco2antiuav410.py --input /data/my_videos --output /data/antiuav410
    python3 tools/coco2antiuav410.py --input /data/coco --output /tmp/smoke \
        --seq-len 50 --limit 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

VERSION = "1.1.0"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
JPEG_EXTS = {".jpg", ".jpeg"}
IMAGE_DIR_NAMES = {"images", "img", "frames"}
FRAME_FIELDS = ("frame_index", "frame_id", "frame")
MIN_TRAINER_VISIBLE = 20
ABSENT_BOX = [0.0, 0.0, 0.0, 0.0]


class ConverterError(Exception):
    """Fatal conversion error (exit code 2)."""


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def natural_key(s):
    """Natural sort key: digit runs compared numerically, rest case-insensitive."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def sanitize_name(name):
    """Restrict a sequence name to [A-Za-z0-9_-]."""
    s = re.sub(r"[^A-Za-z0-9_-]", "_", str(name))
    return s if s else "seq"


class NameAllocator:
    """Hands out unique sequence names, deduping with a numeric suffix."""

    def __init__(self):
        self.used = set()

    def alloc(self, base):
        name, k = base, 1
        while name in self.used:
            k += 1
            name = "{}_{}".format(base, k)
        self.used.add(name)
        return name


class Frame:
    """One output frame: a source image plus the annotations attached to it."""

    __slots__ = ("src", "anns", "width", "height", "merge_key")

    def __init__(self, src, anns, width, height, merge_key):
        self.src = src
        self.anns = anns
        self.width = width
        self.height = height
        self.merge_key = merge_key


# ---------------------------------------------------------------------------
# lazy image codecs (only touched for non-JPEG sources / size fallback)
# ---------------------------------------------------------------------------

def is_inside_real(path, root_real):
    """True when path's realpath is root_real itself or lies underneath it."""
    rp = os.path.realpath(str(path))
    return rp == root_real or rp.startswith(root_real + os.sep)


def decode_image_size(path):
    """Decode the image at path; return (width, height) or None when it cannot
    be decoded (truncated / corrupt / not an image at all).

    cv2 is preferred, PIL is the fallback (both lazy imports).  Raises
    ConverterError only when neither cv2 nor PIL is importable."""
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                return int(img.shape[1]), int(img.shape[0])
        except Exception:
            pass
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        raise ConverterError(
            "cannot decode {}: neither cv2 nor PIL is importable; install "
            "opencv-python-headless or pillow".format(path))
    try:
        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return None


def reencode_to_jpeg(src, dst):
    """Re-encode a non-JPEG frame to a real JPEG (cv2 preferred, PIL fallback)."""
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is not None:
            if cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                return
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        raise ConverterError(
            "cannot re-encode non-JPEG frame {} to JPEG: neither cv2 nor PIL "
            "is importable; install opencv-python-headless or pillow".format(src))
    try:
        with Image.open(src) as im:
            im.convert("RGB").save(str(dst), "JPEG", quality=95)
    except Exception as e:
        raise ConverterError("failed to re-encode {} to JPEG: {}".format(src, e))


# ---------------------------------------------------------------------------
# COCO json discovery / mode detection
# ---------------------------------------------------------------------------

def looks_like_coco(doc):
    return (isinstance(doc, dict)
            and isinstance(doc.get("images"), list)
            and isinstance(doc.get("annotations"), list))


def find_coco_jsons(root, exclude_real=None):
    """All *.json under root (recursive) that look like COCO instances files.

    Returns (found, problems): found is a list of (path, doc); problems is a
    list of (path, reason) for jsons that failed to parse or are not COCO
    instances files (they are reported by the caller instead of vanishing
    silently).  exclude_real (a realpath string) prunes any output tree that
    lives inside the scanned root."""
    found, problems = [], []
    for p in sorted(root.rglob("*.json")):
        if exclude_real is not None and is_inside_real(p, exclude_real):
            continue
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                doc = json.load(f)
        except Exception as e:
            problems.append((p, "failed to parse ({})".format(e)))
            continue
        if looks_like_coco(doc):
            found.append((p, doc))
        else:
            problems.append((p, "not a COCO instances file "
                                "(no top-level 'images' and 'annotations' lists)"))
    return found, problems


def count_images(root, exclude_real=None):
    """Number of image files under root (output tree excluded)."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if exclude_real is not None:
            dirnames[:] = [d for d in dirnames
                           if not is_inside_real(Path(dirpath) / d, exclude_real)]
        n += sum(1 for f in filenames
                 if os.path.splitext(f)[1].lower() in IMG_EXTS)
    return n


def detect_structure(input_dir, annotations_dir, warnings=None,
                     exclude_real=None, strict=False):
    """Classify the input tree.

    Returns (mode, per_video_map, global_jsons, skipped_sources) where
    per_video_map maps subfolder -> [(json_path, doc), ...].  A json is
    treated as GLOBAL when it sits directly under the input root or inside a
    directory named 'annotations'; every other top-level subfolder containing
    a COCO json is a per-video subfolder.  --annotations-dir forces global
    mode.  Unparseable / non-COCO jsons and image-bearing subfolders without
    a readable COCO json are warned about and recorded in skipped_sources
    (under --strict they abort the run)."""
    if warnings is None:
        warnings = []
    skipped_sources = []

    def record_problem(p, reason):
        skipped_sources.append({"path": str(p), "reason": reason})
        if strict:
            raise ConverterError("{}: {} [--strict]".format(p, reason))
        warn("skipping {}: {}".format(p, reason), warnings)

    if annotations_dir is not None:
        if not annotations_dir.is_dir():
            raise ConverterError("--annotations-dir {} is not a directory".format(annotations_dir))
        found, problems = find_coco_jsons(annotations_dir, exclude_real)
        for p, reason in problems:
            record_problem(p, reason)
        return "global", {}, found, skipped_sources

    found, problems = find_coco_jsons(input_dir, exclude_real)
    for p, reason in problems:
        record_problem(p, reason)
    per_video, global_jsons = {}, []
    for p, doc in found:
        try:
            rel = p.relative_to(input_dir)
        except ValueError:
            global_jsons.append((p, doc))
            continue
        if len(rel.parts) == 1 or rel.parts[0].lower() == "annotations":
            global_jsons.append((p, doc))
        else:
            per_video.setdefault(input_dir / rel.parts[0], []).append((p, doc))
    if per_video:
        # Per-video mode: an image-bearing subfolder without a readable COCO
        # json would otherwise vanish from the dataset completely silently.
        for sub in sorted(input_dir.iterdir()):
            if not sub.is_dir() or sub in per_video:
                continue
            if sub.name.lower() == "annotations":
                continue
            if exclude_real is not None and is_inside_real(sub, exclude_real):
                continue
            n = count_images(sub, exclude_real)
            if n:
                reason = ("holds {} image file(s) but no readable COCO "
                          "json".format(n))
                skipped_sources.append({"path": str(sub), "reason": reason})
                warn("subfolder {} {} — skipped".format(sub, reason), warnings)
        return "per_video", per_video, global_jsons, skipped_sources
    if global_jsons:
        return "global", {}, global_jsons, skipped_sources
    raise ConverterError(
        "no COCO json (top-level 'images' and 'annotations' lists) found under "
        "{}".format(input_dir))


# ---------------------------------------------------------------------------
# per-json helpers
# ---------------------------------------------------------------------------

def dedupe_images(images, ctx, warnings):
    """Drop duplicate image ids / file_names (keep first occurrence), warn."""
    seen_ids, seen_names, out = set(), set(), []
    n_id = n_name = 0
    for e in images:
        eid, fn = e.get("id"), e.get("file_name")
        if eid is not None and eid in seen_ids:
            n_id += 1
            continue
        if fn is not None and str(fn) in seen_names:
            n_name += 1
            continue
        if eid is not None:
            seen_ids.add(eid)
        if fn is not None:
            seen_names.add(str(fn))
        out.append(e)
    if n_id:
        warn("{}: dropped {} duplicate image id(s)".format(ctx, n_id), warnings)
    if n_name:
        warn("{}: dropped {} duplicate file_name(s)".format(ctx, n_name), warnings)
    return out


def frame_value(entry):
    """Explicit int-able frame field (frame_index/frame_id/frame) or None."""
    for name in FRAME_FIELDS:
        if name in entry:
            try:
                f = float(entry[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                return int(f)
    return None


def coerce_size(v):
    """Numeric image size (width/height) or None.

    Dirty exporters sometimes quote numbers ("width": "640"); coerce via
    float() and treat anything unparseable / non-finite / non-positive as
    missing so the image itself is decoded as the size fallback."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def order_images(images):
    """Order json image entries: explicit frame field if any entry has one,
    else natural sort of file_name.  Returns (ordered_list, has_explicit)."""
    has_explicit = any(frame_value(e) is not None for e in images)

    def key(e):
        nat = natural_key(str(e.get("file_name", "")))
        fv = frame_value(e)
        if has_explicit and fv is not None:
            return (0, fv, nat)
        return (1, 0, nat)

    return sorted(images, key=key), has_explicit


def scan_subfolder(subfolder, exclude_real=None):
    """Walk a per-video subfolder: dirs named images/img/frames plus a
    basename -> sorted paths index of all image files.  Any output tree nested
    inside the subfolder is pruned from the walk."""
    named, bindex = [], {}
    for dirpath, dirnames, filenames in os.walk(subfolder):
        if exclude_real is not None:
            dirnames[:] = [d for d in dirnames
                           if not is_inside_real(Path(dirpath) / d, exclude_real)]
        for d in dirnames:
            if d.lower() in IMAGE_DIR_NAMES:
                named.append(Path(dirpath) / d)
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                bindex.setdefault(f, []).append(Path(dirpath) / f)
    for v in bindex.values():
        v.sort()
    return sorted(set(named)), bindex


def resolve_image(file_name, roots, basename_index):
    """Resolve a COCO file_name against candidate roots, then by basename.

    file_name may contain subpath components (they are honored); absolute
    paths also work because Path joining honours them.
    """
    rel = str(file_name).replace("\\", "/")
    for r in roots:
        cand = Path(r) / rel
        if cand.is_file():
            return cand
    hits = basename_index.get(os.path.basename(rel))
    if hits:
        return hits[0]
    return None


def resolve_categories(category_args, doc, ctx, warnings):
    """Turn --category values (numeric id or case-insensitive name) into a set
    of category ids for this json.  None means 'no filter'.

    A json that HAS a categories list but matches none of the given values is
    a hard error (an empty filter would silently drop every annotation — the
    classic typo case); a json with no usable categories list warns and
    applies no filter.  Mixed valid+invalid values keep only the valid ones
    (each invalid value is warned about and ignored)."""
    if not category_args:
        return None
    cats = doc.get("categories")
    if not isinstance(cats, list):
        warn("{}: json has no categories key; --category filter not applied".format(ctx),
             warnings)
        return None
    known_ids = {c.get("id") for c in cats}
    name_to_id = {}
    for c in cats:
        name = c.get("name")
        if name is not None:
            name_to_id[str(name).lower()] = c.get("id")
    allowed = set()
    for val in category_args:
        v = str(val).strip()
        as_int = None
        try:
            as_int = int(v)
        except ValueError:
            pass
        if as_int is not None and as_int in known_ids:
            allowed.add(as_int)
            continue
        if v.lower() in name_to_id:
            allowed.add(name_to_id[v.lower()])
            continue
        warn("{}: --category '{}' matches no category id or name in this json "
             "(value ignored here)".format(ctx, val), warnings)
    if not allowed:
        valid = ", ".join("id={}/name={!r}".format(c.get("id"), c.get("name"))
                          for c in cats) or "json has an empty categories list"
        raise ConverterError(
            "{}: none of the --category value(s) [{}] match any category in "
            "this json (valid: {}); refusing to convert with an empty filter "
            "— a typo here would otherwise silently drop every annotation "
            "(all frames exist=0)".format(
                ctx, ", ".join(str(v) for v in category_args), valid))
    return allowed


def pick_target(anns, allowed_cats, pick, include_crowd):
    """Choose the single antiuav410 target box among candidate annotations."""
    cands = []
    for a in anns:
        if a.get("iscrowd", 0) and not include_crowd:
            continue
        if allowed_cats is not None and a.get("category_id") not in allowed_cats:
            continue
        b = a.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            cands.append(b)
    if not cands:
        return None
    if pick == "first":
        return list(cands[0])

    def area(b):
        try:
            w, h = float(b[2]), float(b[3])
            a = w * h
            return a if math.isfinite(a) else float("-inf")
        except (TypeError, ValueError):
            return float("-inf")

    best = max(cands, key=area) if pick == "largest" else min(cands, key=area)
    return list(best)


def sanitize_box(bbox, width, height):
    """Return ([x, y, w, h], clipped) or (None, clipped) for an absent box.

    NaN / non-numeric / w<=0 / h<=0 -> absent.  x, y clipped to >= 0 and
    x+w / y+h clipped to the json image size; a box that becomes degenerate
    after clipping counts as clipped AND absent.
    """
    try:
        x, y, w, h = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None, False
    if any(math.isnan(v) for v in (x, y, w, h)):
        return None, False
    if w <= 0 or h <= 0:
        return None, False
    clipped = False
    if x < 0:
        x, clipped = 0.0, True
    if y < 0:
        y, clipped = 0.0, True
    x2, y2 = x + w, y + h
    if width is not None and x2 > width:
        x2, clipped = float(width), True
    if height is not None and y2 > height:
        y2, clipped = float(height), True
    w2, h2 = x2 - x, y2 - y
    if w2 <= 0 or h2 <= 0:
        return None, True
    return [x, y, w2, h2], clipped


# ---------------------------------------------------------------------------
# sequence planning
# ---------------------------------------------------------------------------

def finalize_frames(frames, doc, opts, ctx, warnings):
    """Target selection + box sanitization -> (gt_rect, exist, n_clipped)."""
    allowed = resolve_categories(opts.category, doc, ctx, warnings)
    gt, exist = [], []
    n_clipped = 0
    for fr in frames:
        box = pick_target(fr.anns, allowed, opts.pick, opts.include_crowd)
        if box is None:
            gt.append(list(ABSENT_BOX))
            exist.append(0)
            continue
        sbox, clipped = sanitize_box(box, fr.width, fr.height)
        if clipped:
            n_clipped += 1
        if sbox is None:
            gt.append(list(ABSENT_BOX))
            exist.append(0)
        else:
            gt.append(sbox)
            exist.append(1)
    return gt, exist, n_clipped


def group_annotations(doc, entries, ctx, warnings):
    """Group annotations by image_id for the given entries; ignore + warn about
    annotations that reference image ids not present in the json at all."""
    known_ids = {e.get("id") for e in entries}
    anns_by_img = {}
    orphan = 0
    for a in doc.get("annotations", []):
        iid = a.get("image_id")
        if iid not in known_ids:
            orphan += 1
            continue
        anns_by_img.setdefault(iid, []).append(a)
    if orphan:
        warn("{}: {} annotation(s) reference unknown image ids (ignored)".format(ctx, orphan), warnings)
    return anns_by_img


def merge_disk_extras(frames, extras_fn, has_explicit, ctx, warnings, opts,
                      undecodable, dropped_real):
    """Merge disk-only frames (image files with no json entry at all) into the
    frame list BEFORE gt_rect/exist are computed, so the label rows can never
    desync from the frames.  Warn whenever extras interleave before the tail
    (or are appended, for explicit-frame-field jsons which cannot interleave).
    Undecodable extras are dropped like undecodable json frames (files already
    reported as undecodable json frames are not re-reported).  Returns the new
    full frame list."""
    claimed = {os.path.realpath(str(f.src)) for f in frames}
    extras = [p for p in extras_fn(claimed)
              if os.path.realpath(str(p)) not in dropped_real]
    if not extras:
        return frames
    kept = []
    for p in extras:
        if opts.verify_frames and p.suffix.lower() in JPEG_EXTS \
                and decode_image_size(p) is None:
            note_undecodable(p, p.name, opts, ctx, undecodable, dropped_real)
            continue
        kept.append(p)
    if not kept:
        return frames
    extra_frames = [Frame(p, [], None, None, natural_key(p.name)) for p in kept]
    if has_explicit:
        warn("{}: {} disk-only frame(s) without json entries appended at the "
             "end (json uses explicit frame fields, so they cannot be "
             "interleaved)".format(ctx, len(kept)), warnings)
        return frames + extra_frames
    merged = sorted(frames + extra_frames, key=lambda f: f.merge_key)
    n_extra = len(extra_frames)
    tail = {id(f) for f in merged[len(merged) - n_extra:]}
    interleaved = any(id(f) not in tail for f in extra_frames)
    warn("{}: {} disk-only frame(s) without json entries merged as absent "
         "frames{}".format(ctx, len(kept),
                           " (they INTERLEAVE with json frames)" if interleaved
                           else ""), warnings)
    return merged


def note_undecodable(src, fn, opts, ctx, undecodable, dropped_real):
    """Record (or, under --strict, abort on) a frame that cannot be decoded."""
    msg = "{}: frame '{}' ({}) is not a decodable image".format(ctx, fn, src)
    if opts.strict:
        raise ConverterError(msg + " [--strict]")
    undecodable.append(str(fn))
    dropped_real.add(os.path.realpath(str(src)))


def plan_from_entries(name, entries, anns_by_img, resolve, doc, opts, ctx, warnings,
                      extras_fn=None, has_explicit=False):
    """Shared per-sequence planning for already-ordered json image entries.

    resolve(file_name) -> Path or None; anns_by_img maps image_id -> anns.
    extras_fn(claimed_realpaths) -> disk-only frame Paths to merge into the
    sequence (per-video mode, single-json subfolders) before labels are
    computed.  Returns a plan dict.
    """
    frames, skipped, undecodable, dropped_real = [], 0, [], set()

    for e in entries:
        fn = e.get("file_name")
        src = resolve(fn) if fn is not None else None
        if src is None:
            skipped += 1
            msg = "{}: image file not found for '{}' (frame skipped)".format(ctx, fn)
            if opts.strict:
                raise ConverterError(msg + " [--strict]")
            warn(msg, warnings)
            continue
        # .jpg/.jpeg sources are copied/linked byte-for-byte, so verify they
        # actually decode (--no-verify-frames restores trusting the suffix);
        # non-JPEG sources are always re-encoded and fail loudly there anyway.
        decoded = None
        if opts.verify_frames and src.suffix.lower() in JPEG_EXTS:
            decoded = decode_image_size(src)
            if decoded is None:
                note_undecodable(src, fn, opts, ctx, undecodable, dropped_real)
                continue
        w, h = coerce_size(e.get("width")), coerce_size(e.get("height"))
        if w is None or h is None:
            # json lacks a usable size -> read it from the image itself
            size = decoded if decoded is not None else decode_image_size(src)
            if size is None:
                note_undecodable(src, fn, opts, ctx, undecodable, dropped_real)
                continue
            w, h = size
        frames.append(Frame(src, anns_by_img.get(e.get("id"), []),
                            w, h,
                            natural_key(os.path.basename(str(fn)))))

    # Disk frames present on disk but with NO json entry at all become absent
    # frames at their sorted position; they are merged BEFORE finalize_frames
    # so gt_rect/exist stay aligned with the frame order.
    if extras_fn is not None:
        frames = merge_disk_extras(frames, extras_fn, has_explicit, ctx,
                                   warnings, opts, undecodable, dropped_real)

    undecodable = list(dict.fromkeys(undecodable))  # a file may appear twice
    if undecodable:
        shown = ", ".join(undecodable[:3])
        if len(undecodable) > 3:
            shown += ", ... (+{} more)".format(len(undecodable) - 3)
        warn("{}: {} undecodable frame(s) skipped entirely (image file and "
             "label row dropped, keeping jpg count == len(gt_rect); first: "
             "{}); use --strict to fail hard or --no-verify-frames to trust "
             "file extensions".format(ctx, len(undecodable), shown), warnings)

    gt, exist, n_clipped = finalize_frames(frames, doc, opts, ctx, warnings)
    n_visible = sum(exist)
    return {
        "name": name,
        "source_json": None,       # filled by caller
        "source_subfolder": None,  # filled by caller
        "frames": frames,
        "gt_rect": gt,
        "exist": exist,
        "n_frames": len(frames),
        "n_visible": n_visible,
        "n_absent": len(frames) - n_visible,
        "n_clipped_boxes": n_clipped,
        "n_skipped_missing_files": skipped,
        "n_undecodable": len(undecodable),
    }


def plan_per_video(input_dir, per_video_map, opts, allocator, warnings,
                   exclude_real=None, jobs=1):
    """Plan every per-video sequence.

    Sequence names are allocated in a serial pass first (the allocator is not
    thread-safe and name assignment must not depend on thread scheduling);
    then each subfolder — an independent unit holding its own jsons, frame
    scan and decode verification — is planned as one task on the pool.
    Results and warnings merge in sorted-subfolder order, byte-identical to
    the serial run."""
    grouped = {}
    for subfolder in sorted(per_video_map):
        jsons = sorted(per_video_map[subfolder])
        multi = len(jsons) > 1
        grouped[subfolder] = []
        for json_path, doc in jsons:
            stem_base = subfolder.name if not multi else "{}_{}".format(subfolder.name, json_path.stem)
            grouped[subfolder].append((json_path, doc,
                                       allocator.alloc(sanitize_name(stem_base))))

    def plan_subfolder(item):
        subfolder, group = item
        local = []
        plans_here = []
        named_dirs, bindex = scan_subfolder(subfolder, exclude_real)
        for json_path, doc, name in group:
            ctx = "sequence '{}' ({})".format(name, json_path)
            images = dedupe_images(doc.get("images", []), ctx, local)
            entries, has_explicit = order_images(images)
            roots = [json_path.parent] + named_dirs + [subfolder]
            anns_by_img = group_annotations(doc, entries, ctx, local)

            def resolve(fn, roots=roots, bindex=bindex):
                return resolve_image(fn, roots, bindex)

            # Disk frames with no json entry at all become absent frames at
            # their sorted position (per-video mode only, and only when the
            # subfolder holds a single json, so frames cannot be
            # mis-attributed between sibling jsons).  They are merged inside
            # plan_from_entries BEFORE gt/exist are computed.
            def extras_fn(claimed, subfolder=subfolder):
                return [p for p in _iter_images(subfolder, exclude_real)
                        if os.path.realpath(str(p)) not in claimed]

            plan = plan_from_entries(name, entries, anns_by_img, resolve, doc,
                                     opts, ctx, local,
                                     extras_fn=extras_fn if len(group) == 1 else None,
                                     has_explicit=has_explicit)
            plan["source_json"] = str(json_path)
            plan["source_subfolder"] = str(subfolder)
            plans_here.append(plan)
        return plans_here, local

    tasks = [partial(plan_subfolder, (sub, grouped[sub]))
             for sub in sorted(grouped)]
    # run_plan_tasks merges each task's local warnings in order already
    plans = []
    for plans_here in run_plan_tasks(tasks, jobs, warnings):
        plans.extend(plans_here)
    return plans


def _iter_images(root, exclude_real=None):
    for dirpath, dirnames, filenames in os.walk(root):
        if exclude_real is not None:
            dirnames[:] = [d for d in dirnames
                           if not is_inside_real(Path(dirpath) / d, exclude_real)]
        for f in sorted(filenames):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                yield Path(dirpath) / f


def global_image_roots(input_dir, json_path, base, images_subdir, exclude_real=None):
    roots = []

    def add(p):
        if p is None or not p.is_dir() or p in roots:
            return
        if exclude_real is not None and is_inside_real(p, exclude_real):
            return  # never read frames out of the converter's own output tree
        roots.append(p)

    if images_subdir:
        add(input_dir / images_subdir)
        add(json_path.parent / images_subdir)
    add(json_path.parent)
    add(json_path.parent.parent / base)
    add(input_dir / base)
    add(input_dir)
    if input_dir.is_dir():
        for d in sorted(input_dir.iterdir()):
            if d.is_dir():
                add(d)
    return roots


def plan_global(input_dir, global_jsons, opts, allocator, warnings,
                exclude_real=None, jobs=1):
    plans = []
    remaining = opts.limit
    if opts.limit is not None and opts.limit <= 0:
        warn("--limit {} caps the run at zero images; nothing to convert".format(opts.limit), warnings)
        return plans

    def plan_chunk(spec):
        name, chunk, anns_by_img, resolve, doc, ctx, json_path = spec
        local = []
        plan = plan_from_entries(name, chunk, anns_by_img, resolve, doc, opts,
                                 ctx, local)
        plan["source_json"] = str(json_path)
        plan["source_subfolder"] = None
        return plan, local

    # Serial pass over the jsons: dedupe/order/group, apply --limit slicing,
    # build the per-json resolve state (read-only at task runtime), and
    # allocate chunk names deterministically.  The chunks themselves — the
    # decode-verify/resolve-heavy part — then run as pool tasks.
    tasks = []
    for json_path, doc in sorted(global_jsons):
        if remaining is not None and remaining <= 0:
            break
        ctx0 = "json {}".format(json_path)
        # json-level warnings (duplicate ids/names, orphan annotations) must
        # land at this json's serial position, not all before every chunk's:
        # a pass-through task placed ahead of the chunk tasks merges them
        # there, keeping the warning order identical to the serial tool.
        prep_local = []
        images = dedupe_images(doc.get("images", []), ctx0, prep_local)
        entries, _has_explicit = order_images(images)
        anns_by_img = group_annotations(doc, entries, ctx0, prep_local)
        if prep_local:
            def emit_prep(preps):
                return None, list(preps)
            tasks.append(partial(emit_prep, prep_local))
        stem = json_path.stem
        base = stem[len("instances_"):] if stem.lower().startswith("instances_") else stem
        base = sanitize_name(base)
        if remaining is not None:
            entries = entries[:remaining]
        roots = global_image_roots(input_dir, json_path, base, opts.images_subdir,
                                   exclude_real)
        bindex = {}
        for r in roots:
            try:
                for f in sorted(r.iterdir()):
                    if f.is_file() and f.suffix.lower() in IMG_EXTS:
                        bindex.setdefault(f.name, []).append(f)
            except OSError:
                continue

        def resolve(fn, roots=roots, bindex=bindex):
            return resolve_image(fn, roots, bindex)

        for ci in range(0, len(entries), opts.seq_len):
            chunk = entries[ci:ci + opts.seq_len]
            name = allocator.alloc("{}_seq{:04d}".format(base, ci // opts.seq_len + 1))
            ctx = "sequence '{}' ({})".format(name, json_path)
            spec = (name, chunk, anns_by_img, resolve, doc, ctx, json_path)
            tasks.append(partial(plan_chunk, spec))
        if remaining is not None:
            remaining -= len(entries)
    # pass-through warning-merge tasks yield None; keep only the chunk plans
    plans.extend(r for r in run_plan_tasks(tasks, jobs, warnings) if r is not None)
    return plans


# ---------------------------------------------------------------------------
# materialization
# ---------------------------------------------------------------------------

def place_frame(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in JPEG_EXTS:
        if mode == "copy":
            shutil.copyfile(str(src), str(dst))          # byte-for-byte
        elif mode == "link":
            try:
                os.link(str(src), str(dst))              # hardlink, same fs only
            except OSError as e:
                raise ConverterError(
                    "hardlink failed for {} -> {} ({}); hardlinks require the "
                    "same filesystem — drop --link or use --symlink".format(src, dst, e))
        else:  # symlink
            os.symlink(os.path.relpath(str(src), start=str(dst.parent)), str(dst))
    else:
        reencode_to_jpeg(src, dst)                       # non-jpg -> real JPEG q95


def write_sequence(output_dir, plan, split, mode):
    seq_dir = output_dir / split / plan["name"]
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True)
    for i, fr in enumerate(plan["frames"], start=1):
        place_frame(fr.src, seq_dir / "{:06d}.jpg".format(i), mode)
    with (seq_dir / "IR_label.json").open("w", encoding="utf-8") as f:
        json.dump({"gt_rect": plan["gt_rect"], "exist": plan["exist"]}, f)
    return seq_dir


def reconcile_output(output_dir, kept_plans, warnings):
    """Prune stale output from previous runs before writing.

    Any sequence dir under {output} holding an IR_label.json is deleted when
    its name is not in the current kept plan (dropped/renamed/re-filtered
    sequences) or when it lives in a split other than the planned one (a
    sequence must never exist in two splits).  Only first-level sequence dirs
    with an IR_label.json are ever touched.  Returns the pruned
    (split, name) records."""
    planned = {p["name"]: p["split"] for p in kept_plans}
    pruned = []
    if not output_dir.is_dir():
        return pruned
    for split_dir in sorted(output_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        removed_here = False
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir() or not (seq_dir / "IR_label.json").is_file():
                continue
            want = planned.get(seq_dir.name)
            if want is None or want != split_dir.name:
                shutil.rmtree(seq_dir)
                pruned.append({"split": split_dir.name, "name": seq_dir.name})
                removed_here = True
        if removed_here and not list(split_dir.iterdir()):
            try:
                split_dir.rmdir()   # drop split dirs this run emptied
            except OSError:
                pass
    if pruned:
        warn("pruned {} stale sequence dir(s) from earlier runs: {}".format(
            len(pruned),
            ", ".join("{}/{}".format(p["split"], p["name"]) for p in pruned)),
            warnings)
    return pruned


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def warn(msg, warnings):
    warnings.append(msg)
    print("WARNING: {}".format(msg), file=sys.stderr)


def _limit_cv2_threads():
    """Pin cv2 to one internal thread (used when --jobs > 1: many small
    per-frame decodes/encodes across workers beat cv2's own all-core
    threading inside each call).  No-op when cv2 is not installed."""
    try:
        import cv2  # type: ignore
    except Exception:
        return
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def run_plan_tasks(tasks, jobs, warnings):
    """Run zero-arg planning/writing tasks, preserving serial semantics.

    Each task returns (result, local_warnings).  With jobs > 1 the tasks run
    on a thread pool — the dominant work (cv2/PIL decode verification,
    stat-heavy file resolution, frame copying, re-encoding) releases the GIL,
    so threads scale.  Results are collected and local warnings merged
    strictly in task order regardless of completion order, keeping the output
    tree, manifest and warning sequence identical to --jobs 1.  A
    ConverterError raised inside a task re-raises from the main thread at the
    first offending task in order (executor.map semantics), like serial."""
    results = []
    if jobs <= 1 or len(tasks) <= 1:
        for task in tasks:
            result, local = task()
            warnings.extend(local)
            results.append(result)
        return results
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for result, local in pool.map(lambda t: t(), tasks):
            warnings.extend(local)
            results.append(result)
    return results


def build_argparser():
    p = argparse.ArgumentParser(
        prog="coco2antiuav410",
        description="Convert COCO-format datasets to the AntiUAV410 training layout "
                    "({output}/{split}/{seq}/IR_label.json + 000001.jpg..00000N.jpg).")
    p.add_argument("--input", required=True,
                   help="dataset root: per-video subfolders (each with its own COCO "
                        "json + frames) OR a global COCO tree (annotations/ + train2017/)")
    p.add_argument("--output", required=True,
                   help="output root; sequences land in "
                        "{output}/{split}/{seq_name}/ (the output tree is "
                        "excluded from input scanning and may live inside "
                        "--input; re-runs prune stale sequence dirs from "
                        "earlier runs)")
    p.add_argument("--seq-len", type=int, default=50, metavar="N",
                   help="global mode: pseudo-video chunk size (default: 50)")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="global mode: cap on total images converted, applied before chunking")
    p.add_argument("--images-subdir", default=None, metavar="DIR",
                   help="global mode: image directory under --input (e.g. train2017) "
                        "when the json sits in annotations/")
    p.add_argument("--annotations-dir", default=None, metavar="DIR",
                   help="look for global COCO json(s) here instead of under --input "
                        "(forces global mode)")
    p.add_argument("--category", action="append", default=None, metavar="ID_OR_NAME",
                   help="restrict annotations to this category (repeatable; numeric id "
                        "or case-insensitive name, resolved via the json's categories)")
    p.add_argument("--pick", choices=["largest", "first", "smallest"], default="largest",
                   help="which candidate annotation becomes the tracked target "
                        "(default: largest by bbox area)")
    p.add_argument("--include-crowd", action="store_true",
                   help="do NOT skip iscrowd annotations (skipped by default)")
    p.add_argument("--val-ratio", type=float, default=None, metavar="R",
                   help="deterministic train/val split: of the sorted sequence names, "
                        "every max(1, round(1/R))-th goes to val (0 < R < 1)")
    p.add_argument("--split", default=None, metavar="NAME",
                   help="send every sequence to {output}/NAME (overrides --val-ratio)")
    p.add_argument("--min-visible", type=int, default=None, metavar="N",
                   help="drop sequences with fewer than N visible frames "
                        "(default: keep, but warn below 20)")
    p.add_argument("--link", action="store_true",
                   help="hardlink jpg frames instead of copying (same filesystem only)")
    p.add_argument("--symlink", action="store_true",
                   help="relative-symlink jpg frames instead of copying")
    p.add_argument("--strict", action="store_true",
                   help="fail hard on json entries whose image file is missing, "
                        "on frames that cannot be decoded, and on corrupt/"
                        "unparseable COCO jsons")
    p.add_argument("--no-verify-frames", dest="verify_frames",
                   action="store_false", default=True,
                   help="do NOT decode-verify .jpg/.jpeg sources before "
                        "copying/linking them (default: verify; undecodable "
                        "frames are skipped with a warning — trades extra I/O "
                        "for guaranteed-readable output frames)")
    p.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 4),
                   metavar="N",
                   help="number of worker threads for planning (decode "
                        "verification, file resolution) and writing frames "
                        "(default: min(16, cpu_count)); --jobs 1 is fully "
                        "serial.  Output, manifest and warning order are "
                        "identical for any N")
    p.add_argument("--version", action="version", version="%(prog)s " + VERSION)
    return p


def main(argv=None):
    opts = build_argparser().parse_args(argv)

    if opts.link and opts.symlink:
        print("ERROR: --link and --symlink are mutually exclusive", file=sys.stderr)
        return 2
    if opts.seq_len < 1:
        print("ERROR: --seq-len must be >= 1", file=sys.stderr)
        return 2
    if opts.val_ratio is not None and not (0.0 < opts.val_ratio < 1.0):
        print("ERROR: --val-ratio must satisfy 0 < R < 1", file=sys.stderr)
        return 2
    if opts.jobs < 1:
        print("ERROR: --jobs must be >= 1", file=sys.stderr)
        return 2
    if opts.jobs > 1:
        _limit_cv2_threads()  # before any pool: workers share one cv2 runtime

    input_dir = Path(opts.input).expanduser().resolve()
    output_dir = Path(opts.output).expanduser().resolve()
    if not input_dir.is_dir():
        print("ERROR: --input {} is not a directory".format(input_dir), file=sys.stderr)
        return 2
    annotations_dir = (Path(opts.annotations_dir).expanduser().resolve()
                       if opts.annotations_dir else None)

    warnings = []
    out_real = str(output_dir)
    exclude_real = None
    if out_real == str(input_dir):
        print("ERROR: --output must not be the same directory as --input",
              file=sys.stderr)
        return 2
    if is_inside_real(output_dir, str(input_dir)):
        warn("output dir is inside the input tree; it will be excluded from "
             "scanning", warnings)
        exclude_real = out_real
    try:
        mode, per_video_map, global_jsons, skipped_sources = detect_structure(
            input_dir, annotations_dir, warnings=warnings,
            exclude_real=exclude_real, strict=opts.strict)
        mode_note = None
        if mode == "per_video" and global_jsons:
            mode_note = ("both per-video and global COCO structures found under {}; "
                         "per-video mode preferred (global jsons ignored: {})".format(
                             input_dir, ", ".join(str(p) for p, _ in global_jsons)))
            warn(mode_note, warnings)
        if mode == "global" and opts.limit is None and opts.annotations_dir is None:
            pass  # full global conversion is allowed; nothing to say

        if opts.limit is not None and mode == "per_video":
            warn("--limit applies to global mode only; ignored in per-video mode", warnings)

        allocator = NameAllocator()
        if mode == "per_video":
            plans = plan_per_video(input_dir, per_video_map, opts, allocator,
                                   warnings, exclude_real, jobs=opts.jobs)
        else:
            plans = plan_global(input_dir, global_jsons, opts, allocator,
                                warnings, exclude_real, jobs=opts.jobs)

        # drop / warn pass (visible counts are known once planned)
        kept, dropped = [], []
        for plan in plans:
            if plan["n_frames"] == 0:
                dropped.append({"name": plan["name"], "n_visible": plan["n_visible"],
                                "reason": "no usable frames (all image files missing or undecodable)"})
                continue
            if opts.min_visible is not None and plan["n_visible"] < opts.min_visible:
                dropped.append({"name": plan["name"], "n_visible": plan["n_visible"],
                                "reason": "below --min-visible {}".format(opts.min_visible)})
                continue
            if plan["n_visible"] < MIN_TRAINER_VISIBLE:
                warn("sequence '{}' has only {} visible frames (< {}): the FocusTrack "
                     "trainer sampler will never pick it".format(
                         plan["name"], plan["n_visible"], MIN_TRAINER_VISIBLE), warnings)
            kept.append(plan)

        # deterministic split assignment over the sorted sequence names
        if opts.split:
            for plan in kept:
                plan["split"] = opts.split
        elif opts.val_ratio is not None:
            step = max(1, round(1.0 / opts.val_ratio))
            for i, plan in enumerate(sorted(kept, key=lambda p: p["name"])):
                plan["split"] = "val" if i % step == 0 else "train"
        else:
            for plan in kept:
                plan["split"] = "train"

        output_dir.mkdir(parents=True, exist_ok=True)
        # reconcile with previous runs: drop stale sequence dirs and any
        # same-name dir in a split other than the planned one
        pruned = reconcile_output(output_dir, kept, warnings)
        place_mode = "link" if opts.link else ("symlink" if opts.symlink else "copy")

        def write_one(plan):
            write_sequence(output_dir, plan, plan["split"], place_mode)
            return None, []

        # one task per sequence: disjoint dirs, so writes are independent
        run_plan_tasks([partial(write_one, p)
                        for p in sorted(kept, key=lambda p: p["name"])],
                       opts.jobs, warnings)

        manifest = {
            "tool": "coco2antiuav410",
            "version": VERSION,
            "input": str(input_dir),
            "output": str(output_dir),
            "mode": mode,
            "mode_note": mode_note,
            "args": {
                "seq_len": opts.seq_len,
                "limit": opts.limit,
                "images_subdir": opts.images_subdir,
                "annotations_dir": opts.annotations_dir,
                "category": opts.category,
                "pick": opts.pick,
                "include_crowd": opts.include_crowd,
                "val_ratio": opts.val_ratio,
                "split": opts.split,
                "min_visible": opts.min_visible,
                "link": opts.link,
                "symlink": opts.symlink,
                "strict": opts.strict,
                "verify_frames": opts.verify_frames,
                "jobs": opts.jobs,
            },
            "sequences": [
                {
                    "name": p["name"],
                    "split": p["split"],
                    "n_frames": p["n_frames"],
                    "n_visible": p["n_visible"],
                    "n_absent": p["n_absent"],
                    "n_clipped_boxes": p["n_clipped_boxes"],
                    "n_skipped_missing_files": p["n_skipped_missing_files"],
                    "n_undecodable": p["n_undecodable"],
                    "source_subfolder": p["source_subfolder"],
                    "source_json": p["source_json"],
                }
                for p in sorted(kept, key=lambda x: x["name"])
            ],
            "dropped_sequences": sorted(dropped, key=lambda d: d["name"]),
            "skipped_sources": skipped_sources,
            "pruned": pruned,
            "totals": {
                "n_sequences": len(kept),
                "n_dropped": len(dropped),
                "n_frames": sum(p["n_frames"] for p in kept),
                "n_visible": sum(p["n_visible"] for p in kept),
                "n_absent": sum(p["n_absent"] for p in kept),
                "n_clipped_boxes": sum(p["n_clipped_boxes"] for p in kept),
                "n_skipped_missing_files": sum(p["n_skipped_missing_files"] for p in kept),
                "n_undecodable": sum(p["n_undecodable"] for p in kept),
            },
            "warnings": warnings,
        }
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # human summary
        print("coco2antiuav410 v{}: mode={}, sequences={} (dropped {}), frames={} "
              "(visible {}, absent {})".format(
                  VERSION, mode, len(kept), len(dropped),
                  manifest["totals"]["n_frames"],
                  manifest["totals"]["n_visible"],
                  manifest["totals"]["n_absent"]))
        for p in sorted(kept, key=lambda x: x["name"]):
            print("  {:40s} split={:5s} frames={:5d} visible={:5d} absent={:5d} "
                  "clipped={:3d} skipped_missing={:3d}".format(
                      p["name"], p["split"], p["n_frames"], p["n_visible"],
                      p["n_absent"], p["n_clipped_boxes"], p["n_skipped_missing_files"]))
        if dropped:
            print("  dropped sequences:")
            for d in sorted(dropped, key=lambda x: x["name"]):
                print("    {} (visible={}, {})".format(d["name"], d["n_visible"], d["reason"]))
        if pruned:
            print("  pruned {} stale sequence dir(s) from earlier runs "
                  "(recorded in manifest 'pruned')".format(len(pruned)))
        print("manifest: {}".format(output_dir / "manifest.json"))
        if warnings:
            print("{} warning(s) recorded in manifest and stderr".format(len(warnings)))
        return 0
    except ConverterError as e:
        print("ERROR: {}".format(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
