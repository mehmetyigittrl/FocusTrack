"""Test suite for tools/coco2antiuav410.py (COCO -> AntiUAV410 converter).

Run inside the gzs_env_2404 conda environment:

    conda run -n gzs_env_2404 python -m pytest tools/tests/test_coco2antiuav410.py -v

The suite builds synthetic per-video COCO trees and a global-COCO tree in
tempfile dirs (tiny real images written with PIL, cv2 as fallback) and invokes
the converter through its CLI exactly like a user would.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "coco2antiuav410.py"

CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "human"},
    {"id": 2, "name": "car", "supercategory": "vehicle"},
    {"id": 3, "name": "uav", "supercategory": "object"},
]

JPEG_MAGIC = b"\xff\xd8\xff"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def write_image(path, size=(32, 24), color=(90, 128, 40)):
    """Write a tiny real image; format follows the extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.new("RGB", size, color).save(path)
    except ImportError:
        import cv2
        import numpy as np
        arr = np.full((size[1], size[0], 3), color[::-1], dtype=np.uint8)
        cv2.imwrite(str(path), arr)
    return path


def write_coco(path, images, annotations, categories=CATEGORIES):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"images": images, "annotations": annotations}
    if categories is not None:
        doc["categories"] = categories
    path.write_text(json.dumps(doc))
    return path


def img_entry(i, fn, w=32, h=24, **kw):
    d = {"id": i, "file_name": fn, "width": w, "height": h}
    d.update(kw)
    return d


def ann(i, image_id, bbox, cat=3, crowd=0):
    return {"id": i, "image_id": image_id, "category_id": cat,
            "iscrowd": crowd, "bbox": list(bbox)}


def run_converter(*args):
    cmd = [sys.executable, str(SCRIPT)] + [str(a) for a in args]
    return subprocess.run(cmd, capture_output=True, text=True)


def build_video(root, name, frame_specs, json_name="instances.json",
                categories=CATEGORIES):
    """frame_specs: list of (file_name, [list of ann tuples (bbox, cat, crowd)]).

    Images are written next to the json inside {root}/{name}. Returns paths.
    """
    vid = Path(root) / name
    vid.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    aid = 0
    for i, (fn, anns) in enumerate(frame_specs, start=1):
        write_image(vid / fn)
        images.append(img_entry(i, fn))
        for bbox, cat, crowd in anns:
            aid += 1
            annotations.append(ann(aid, i, bbox, cat=cat, crowd=crowd))
    j = write_coco(vid / json_name, images, annotations, categories)
    return vid, j


def read_label(seq_dir):
    return json.loads((Path(seq_dir) / "IR_label.json").read_text())


def seq_dirs(out, split="train"):
    d = Path(out) / split
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir())


# --------------------------------------------------------------------------
# per-video mode
# --------------------------------------------------------------------------

def test_per_video_happy_path(tmp_path):
    vid, _ = build_video(tmp_path / "in", "videoA", [
        ("img_0001.jpg", [([1.0, 2.0, 5.0, 6.0], 3, 0)]),
        ("img_0002.jpg", [([2.0, 3.0, 5.0, 6.0], 3, 0)]),
        ("img_0003.jpg", [([3.0, 4.0, 5.0, 6.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr

    seqs = seq_dirs(out)
    assert [p.name for p in seqs] == ["videoA"]
    seq = seqs[0]

    # contiguous renumbering 000001..N, jpg only besides IR_label.json
    files = sorted(p.name for p in seq.iterdir())
    assert files == ["000001.jpg", "000002.jpg", "000003.jpg", "IR_label.json"]

    lab = read_label(seq)
    assert set(lab.keys()) == {"gt_rect", "exist"}
    assert lab["gt_rect"] == [[1.0, 2.0, 5.0, 6.0],
                              [2.0, 3.0, 5.0, 6.0],
                              [3.0, 4.0, 5.0, 6.0]]
    assert lab["exist"] == [1, 1, 1]

    man = json.loads((out / "manifest.json").read_text())
    assert man["mode"] == "per_video"
    assert man["totals"]["n_sequences"] == 1
    assert man["totals"]["n_frames"] == 3
    assert man["totals"]["n_visible"] == 3
    s = man["sequences"][0]
    assert s["name"] == "videoA"
    assert s["split"] == "train"
    assert s["n_frames"] == 3 and s["n_visible"] == 3 and s["n_absent"] == 0
    # 3 visible < 20 -> short-sequence warning in manifest and on stderr
    assert any("visible" in w for w in man["warnings"])
    assert "WARNING" in r.stderr


def test_png_frames_reencoded_to_real_jpeg(tmp_path):
    vid, _ = build_video(tmp_path / "in", "videoB", [
        ("frame_001.png", [([1.0, 1.0, 4.0, 4.0], 3, 0)]),
        ("frame_002.png", [([2.0, 2.0, 4.0, 4.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "videoB"
    for i in (1, 2):
        dst = seq / f"{i:06d}.jpg"
        assert dst.is_file()
        data = dst.read_bytes()
        assert data[:3] == JPEG_MAGIC, "output must be a real JPEG"
        assert data != (vid / f"frame_{i:03d}.png").read_bytes()
    lab = read_label(seq)
    assert lab["exist"] == [1, 1]


def test_nonjpg_reencoded_even_with_link_and_symlink(tmp_path):
    # --link and --symlink must still re-encode non-jpg sources into real files
    for flag in ("--link", "--symlink"):
        src_in = tmp_path / f"in_{flag.strip('-')}"
        build_video(src_in, "videoP", [
            ("f1.png", [([1.0, 1.0, 4.0, 4.0], 3, 0)]),
        ])
        out = tmp_path / f"out_{flag.strip('-')}"
        r = run_converter("--input", src_in, "--output", out, flag)
        assert r.returncode == 0, r.stderr
        dst = out / "train" / "videoP" / "000001.jpg"
        assert dst.is_file()
        assert not dst.is_symlink()
        assert dst.read_bytes()[:3] == JPEG_MAGIC


def test_absent_frames_unannotated_and_disk_only(tmp_path):
    vid, _ = build_video(tmp_path / "in", "videoC", [
        ("img_0001.jpg", [([1.0, 1.0, 5.0, 5.0], 3, 0)]),
        ("img_0002.jpg", []),                # json entry without annotations
        ("img_0003.jpg", [([3.0, 3.0, 5.0, 5.0], 3, 0)]),
    ])
    write_image(vid / "img_0004.jpg")        # disk-only: no json entry at all
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr

    seq = out / "train" / "videoC"
    lab = read_label(seq)
    assert len(lab["gt_rect"]) == 4
    assert len(lab["exist"]) == 4
    assert lab["exist"] == [1, 0, 1, 0]
    assert lab["gt_rect"][1] == [0, 0, 0, 0]
    assert lab["gt_rect"][3] == [0, 0, 0, 0]
    assert (seq / "000004.jpg").is_file()    # disk-only frame copied at sorted position

    man = json.loads((out / "manifest.json").read_text())
    s = man["sequences"][0]
    assert s["n_frames"] == 4
    assert s["n_visible"] == 2
    assert s["n_absent"] == 2


def test_multi_object_frame_picks_largest_by_default(tmp_path):
    build_video(tmp_path / "in", "videoD", [
        # small box listed FIRST, large second: largest must win, not first
        ("img_0001.jpg", [([1.0, 1.0, 2.0, 2.0], 3, 0),
                          ([10.0, 10.0, 10.0, 8.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoD")
    assert lab["gt_rect"][0] == [10.0, 10.0, 10.0, 8.0]
    assert lab["exist"] == [1]


def test_pick_first_and_smallest(tmp_path):
    build_video(tmp_path / "in", "videoE", [
        ("img_0001.jpg", [([1.0, 1.0, 2.0, 2.0], 3, 0),
                          ([10.0, 10.0, 10.0, 8.0], 3, 0)]),
    ])
    out1 = tmp_path / "out_first"
    r = run_converter("--input", tmp_path / "in", "--output", out1, "--pick", "first")
    assert r.returncode == 0, r.stderr
    assert read_label(out1 / "train" / "videoE")["gt_rect"][0] == [1.0, 1.0, 2.0, 2.0]

    out2 = tmp_path / "out_smallest"
    r = run_converter("--input", tmp_path / "in", "--output", out2, "--pick", "smallest")
    assert r.returncode == 0, r.stderr
    assert read_label(out2 / "train" / "videoE")["gt_rect"][0] == [1.0, 1.0, 2.0, 2.0]


def test_category_filter_by_id_and_name(tmp_path):
    build_video(tmp_path / "in", "videoF", [
        ("img_0001.jpg", [([1.0, 1.0, 30.0, 20.0], 1, 0),   # person, big
                          ([5.0, 5.0, 3.0, 3.0], 2, 0)]),    # car, small
    ])
    # by numeric id
    out = tmp_path / "out_id"
    r = run_converter("--input", tmp_path / "in", "--output", out, "--category", "2")
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoF")
    assert lab["gt_rect"][0] == [5.0, 5.0, 3.0, 3.0]

    # by case-insensitive name
    out = tmp_path / "out_name"
    r = run_converter("--input", tmp_path / "in", "--output", out, "--category", "PeRsOn")
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoF")
    assert lab["gt_rect"][0] == [1.0, 1.0, 30.0, 20.0]

    # repeatable flag: both categories -> largest across the union wins
    out = tmp_path / "out_both"
    r = run_converter("--input", tmp_path / "in", "--output", out,
                      "--category", "person", "--category", "car")
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoF")
    assert lab["gt_rect"][0] == [1.0, 1.0, 30.0, 20.0]


def test_iscrowd_skipped_by_default_and_included_with_flag(tmp_path):
    build_video(tmp_path / "in", "videoG", [
        # frame 1: ONLY a crowd ann -> absent by default
        ("img_0001.jpg", [([1.0, 1.0, 20.0, 15.0], 3, 1)]),
        # frame 2: crowd (big) + normal (small) -> normal wins by default
        ("img_0002.jpg", [([1.0, 1.0, 25.0, 20.0], 3, 1),
                          ([4.0, 4.0, 4.0, 4.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoG")
    assert lab["exist"] == [0, 1]
    assert lab["gt_rect"][0] == [0, 0, 0, 0]
    assert lab["gt_rect"][1] == [4.0, 4.0, 4.0, 4.0]

    out2 = tmp_path / "out_crowd"
    r = run_converter("--input", tmp_path / "in", "--output", out2, "--include-crowd")
    assert r.returncode == 0, r.stderr
    lab = read_label(out2 / "train" / "videoG")
    assert lab["exist"] == [1, 1]
    assert lab["gt_rect"][0] == [1.0, 1.0, 20.0, 15.0]
    assert lab["gt_rect"][1] == [1.0, 1.0, 25.0, 20.0]


def test_out_of_bounds_clipping_and_fully_out_absent(tmp_path):
    build_video(tmp_path / "in", "videoH", [
        ("img_0001.jpg", [([-5.0, -3.0, 40.0, 30.0], 3, 0)]),  # spills all sides
        ("img_0002.jpg", [([50.0, 10.0, 5.0, 5.0], 3, 0)]),    # fully outside
        ("img_0003.jpg", [([1.0, 1.0, 5.0, 5.0], 3, 0)]),      # untouched
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoH")
    assert lab["gt_rect"][0] == [0.0, 0.0, 32.0, 24.0]  # clipped to 32x24 image
    assert lab["exist"][1] == 0 and lab["gt_rect"][1] == [0, 0, 0, 0]
    assert lab["gt_rect"][2] == [1.0, 1.0, 5.0, 5.0]

    man = json.loads((out / "manifest.json").read_text())
    assert man["sequences"][0]["n_clipped_boxes"] == 2  # clipped + fully-out


def test_nan_and_degenerate_boxes_absent(tmp_path):
    vid = tmp_path / "in" / "videoI"
    vid.mkdir(parents=True)
    images, annotations = [], []
    bboxes = [[float("nan"), 1.0, 5.0, 5.0],
              [1.0, 1.0, 0.0, 5.0],
              [1.0, 1.0, 5.0, -2.0],
              [1.0, 1.0, 5.0, 5.0]]
    for i, bbox in enumerate(bboxes, start=1):
        write_image(vid / f"img_{i:04d}.jpg")
        images.append(img_entry(i, f"img_{i:04d}.jpg"))
        annotations.append(ann(i, i, bbox))
    write_coco(vid / "instances.json", images, annotations)

    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoI")
    assert lab["exist"] == [0, 0, 0, 1]
    assert lab["gt_rect"][0] == [0, 0, 0, 0]
    assert lab["gt_rect"][3] == [1.0, 1.0, 5.0, 5.0]


def test_duplicate_file_name_and_id_deduped(tmp_path):
    vid = tmp_path / "in" / "videoJ"
    vid.mkdir(parents=True)
    write_image(vid / "dup.jpg")
    images = [
        img_entry(1, "dup.jpg"),
        img_entry(2, "dup.jpg"),          # duplicate file_name
        img_entry(3, "other.jpg"),
    ]
    write_image(vid / "other.jpg")
    images.append(img_entry(3, "third.jpg"))  # duplicate id 3
    write_image(vid / "third.jpg")
    annotations = [ann(1, 1, [1.0, 1.0, 4.0, 4.0]),
                   ann(2, 2, [2.0, 2.0, 4.0, 4.0]),
                   ann(3, 3, [3.0, 3.0, 4.0, 4.0])]
    write_coco(vid / "instances.json", images, annotations)

    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "videoJ"
    jpgs = sorted(p.name for p in seq.glob("*.jpg"))
    # dup.jpg (deduped), other.jpg, third.jpg
    assert jpgs == ["000001.jpg", "000002.jpg", "000003.jpg"]
    man = json.loads((out / "manifest.json").read_text())
    joined = " ".join(man["warnings"])
    assert "duplicate" in joined.lower()


def test_natural_sort_order(tmp_path):
    # lexicographic order would be img1, img10, img2; natural order is img1, img2, img10
    build_video(tmp_path / "in", "videoK", [
        ("img10.jpg", [([10.0, 0.0, 4.0, 4.0], 3, 0)]),
        ("img2.jpg", [([2.0, 0.0, 4.0, 4.0], 3, 0)]),
        ("img1.jpg", [([1.0, 0.0, 4.0, 4.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoK")
    assert [row[0] for row in lab["gt_rect"]] == [1.0, 2.0, 10.0]


def test_frame_index_orders_frames(tmp_path):
    vid = tmp_path / "in" / "videoL"
    vid.mkdir(parents=True)
    images, annotations = [], []
    for i, (fn, fidx) in enumerate([("c.jpg", 2), ("a.jpg", 0), ("b.jpg", 1)], start=1):
        write_image(vid / fn)
        images.append(img_entry(i, fn, frame_index=fidx))
        annotations.append(ann(i, i, [float(fidx), 0.0, 4.0, 4.0]))
    write_coco(vid / "instances.json", images, annotations)

    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoL")
    assert [row[0] for row in lab["gt_rect"]] == [0.0, 1.0, 2.0]


def test_image_path_resolution_chain(tmp_path):
    vid = tmp_path / "in" / "videoM"
    vid.mkdir(parents=True)
    write_image(vid / "images" / "a.jpg")     # file_name carries subpath -> json-dir relative
    write_image(vid / "images" / "b.jpg")     # bare file_name in images/ subdir
    write_image(vid / "more" / "c.jpg")       # bare file_name in unrelated dir (basename search)
    images = [img_entry(1, "images/a.jpg"),
              img_entry(2, "b.jpg"),
              img_entry(3, "c.jpg")]
    annotations = [ann(1, 1, [1.0, 1.0, 3.0, 3.0]),
                   ann(2, 2, [2.0, 1.0, 3.0, 3.0]),
                   ann(3, 3, [3.0, 1.0, 3.0, 3.0])]
    write_coco(vid / "instances.json", images, annotations)

    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoM")
    assert lab["exist"] == [1, 1, 1]
    # frame order is the natural sort of the FULL file_name (spec rule), so the
    # bare names come first and the subpath-prefixed one last:
    # "b.jpg" < "c.jpg" < "images/a.jpg"
    assert [row[0] for row in lab["gt_rect"]] == [2.0, 3.0, 1.0]


def test_multiple_jsons_in_one_subfolder(tmp_path):
    vid = tmp_path / "in" / "videoN"
    vid.mkdir(parents=True)
    for stem, xoff in (("partA", 1.0), ("partB", 5.0)):
        write_image(vid / f"{stem}_1.jpg")
        write_coco(vid / f"{stem}.json",
                   [img_entry(1, f"{stem}_1.jpg")],
                   [ann(1, 1, [xoff, 1.0, 3.0, 3.0])])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    names = [p.name for p in seq_dirs(out)]
    assert names == ["videoN_partA", "videoN_partB"]
    assert read_label(out / "train" / "videoN_partA")["gt_rect"][0][0] == 1.0
    assert read_label(out / "train" / "videoN_partB")["gt_rect"][0][0] == 5.0


def test_sequence_name_sanitization(tmp_path):
    build_video(tmp_path / "in", "My Video (1)!", [
        ("a.jpg", [([1.0, 1.0, 3.0, 3.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    names = [p.name for p in seq_dirs(out)]
    import re as _re
    assert names == [_re.sub(r"[^A-Za-z0-9_-]", "_", "My Video (1)!")]


def test_missing_file_skipped_and_strict_fails(tmp_path):
    vid = tmp_path / "in" / "videoO"
    vid.mkdir(parents=True)
    write_image(vid / "real_1.jpg")
    write_image(vid / "real_2.jpg")
    images = [img_entry(1, "real_1.jpg"),
              img_entry(2, "ghost.jpg"),        # json entry, no file on disk
              img_entry(3, "real_2.jpg")]
    annotations = [ann(1, 1, [1.0, 1.0, 3.0, 3.0]),
                   ann(2, 2, [2.0, 2.0, 3.0, 3.0]),
                   ann(3, 3, [3.0, 3.0, 3.0, 3.0])]
    write_coco(vid / "instances.json", images, annotations)

    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoO")
    assert len(lab["exist"]) == 2                     # ghost skipped entirely
    assert [row[0] for row in lab["gt_rect"]] == [1.0, 3.0]
    man = json.loads((out / "manifest.json").read_text())
    assert man["sequences"][0]["n_skipped_missing_files"] == 1
    assert any("ghost" in w for w in man["warnings"])

    out2 = tmp_path / "out_strict"
    r2 = run_converter("--input", tmp_path / "in", "--output", out2, "--strict")
    assert r2.returncode != 0
    assert "ghost" in (r2.stderr + r2.stdout)


def test_min_visible_drops_short_sequences(tmp_path):
    build_video(tmp_path / "in", "videoGood", [
        (f"g{i:04d}.jpg", [([1.0, 1.0, 3.0, 3.0], 3, 0)]) for i in range(1, 4)
    ])
    build_video(tmp_path / "in", "videoBad", [          # no annotations -> 0 visible
        (f"b{i:04d}.jpg", []) for i in range(1, 4)
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out, "--min-visible", "2")
    assert r.returncode == 0, r.stderr
    names = [p.name for p in seq_dirs(out)]
    assert names == ["videoGood"]
    man = json.loads((out / "manifest.json").read_text())
    assert [d["name"] for d in man["dropped_sequences"]] == ["videoBad"]
    assert man["dropped_sequences"][0]["n_visible"] == 0


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def _build_eight_videos(root):
    for i in range(1, 9):
        build_video(root, f"vid_{i:02d}", [
            (f"f{i}.jpg", [([1.0, 1.0, 3.0, 3.0], 3, 0)]),
        ])


def test_val_ratio_deterministic_split(tmp_path):
    _build_eight_videos(tmp_path / "in")
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out, "--val-ratio", "0.25")
    assert r.returncode == 0, r.stderr
    train_names = [p.name for p in seq_dirs(out, "train")]
    val_names = [p.name for p in seq_dirs(out, "val")]
    # every 4th of the sorted names (0-based idx 0 and 4) -> val, rest -> train
    assert val_names == ["vid_01", "vid_05"]
    assert train_names == ["vid_%02d" % i for i in (2, 3, 4, 6, 7, 8)]

    # determinism: identical assignment on a fresh run
    out2 = tmp_path / "out2"
    r2 = run_converter("--input", tmp_path / "in", "--output", out2, "--val-ratio", "0.25")
    assert r2.returncode == 0, r2.stderr
    assert [p.name for p in seq_dirs(out2, "val")] == val_names
    assert [p.name for p in seq_dirs(out2, "train")] == train_names


def test_split_override_sends_everything_to_named_split(tmp_path):
    _build_eight_videos(tmp_path / "in")
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out,
                      "--split", "valtest", "--val-ratio", "0.5")
    assert r.returncode == 0, r.stderr
    assert len(seq_dirs(out, "valtest")) == 8
    assert not (out / "train").exists()
    assert not (out / "val").exists()


# --------------------------------------------------------------------------
# linking modes
# --------------------------------------------------------------------------

def test_hardlink_same_inode(tmp_path):
    vid, _ = build_video(tmp_path / "in", "videoLink", [
        ("a.jpg", [([1.0, 1.0, 3.0, 3.0], 3, 0)]),
        ("b.jpg", [([2.0, 1.0, 3.0, 3.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out, "--link")
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "videoLink"
    for src_name, dst_name in (("a.jpg", "000001.jpg"), ("b.jpg", "000002.jpg")):
        dst = seq / dst_name
        assert dst.is_file() and not dst.is_symlink()
        assert os.stat(dst).st_ino == os.stat(vid / src_name).st_ino


def test_symlink_relative_and_resolves(tmp_path):
    vid, _ = build_video(tmp_path / "in", "videoSym", [
        ("a.jpg", [([1.0, 1.0, 3.0, 3.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out, "--symlink")
    assert r.returncode == 0, r.stderr
    dst = out / "train" / "videoSym" / "000001.jpg"
    assert dst.is_symlink()
    target = os.readlink(dst)
    assert not target.startswith("/")            # relative symlink
    assert os.path.realpath(dst) == str(vid / "a.jpg")
    assert dst.read_bytes() == (vid / "a.jpg").read_bytes()


# --------------------------------------------------------------------------
# global-COCO mode
# --------------------------------------------------------------------------

def _build_global(root, n_images=7, img_dir="train2017",
                  json_name="instances_train2017.json"):
    ann_dir = Path(root) / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (Path(root) / img_dir).mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i in range(1, n_images + 1):
        fn = f"{i:012d}.jpg"
        write_image(Path(root) / img_dir / fn)
        images.append(img_entry(i, fn))
        annotations.append(ann(i, i, [float(i), 1.0, 4.0, 4.0]))
    write_coco(ann_dir / json_name, images, annotations)


def test_global_mode_chunking_naming_and_limit(tmp_path):
    _build_global(tmp_path / "in", n_images=7)
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out,
                      "--seq-len", "3", "--limit", "7")
    assert r.returncode == 0, r.stderr
    names = [p.name for p in seq_dirs(out)]
    assert names == ["train2017_seq0001", "train2017_seq0002", "train2017_seq0003"]
    lab1 = read_label(out / "train" / "train2017_seq0001")
    assert len(lab1["exist"]) == 3
    assert [row[0] for row in lab1["gt_rect"]] == [1.0, 2.0, 3.0]  # natural order
    assert len(read_label(out / "train" / "train2017_seq0002")["exist"]) == 3
    lab3 = read_label(out / "train" / "train2017_seq0003")
    assert len(lab3["exist"]) == 1
    man = json.loads((out / "manifest.json").read_text())
    assert man["mode"] == "global"

    # --limit applied BEFORE chunking: 4 images -> one chunk of 3 + one of 1
    out2 = tmp_path / "out2"
    r2 = run_converter("--input", tmp_path / "in", "--output", out2,
                       "--seq-len", "3", "--limit", "4")
    assert r2.returncode == 0, r2.stderr
    names2 = [p.name for p in seq_dirs(out2)]
    assert names2 == ["train2017_seq0001", "train2017_seq0002"]
    assert len(read_label(out2 / "train" / "train2017_seq0002")["exist"]) == 1


def test_global_mode_images_subdir(tmp_path):
    _build_global(tmp_path / "in", n_images=3, img_dir="myimgs")
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out,
                      "--seq-len", "2", "--images-subdir", "myimgs")
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "train2017_seq0001"
    assert (seq / "000001.jpg").is_file()
    assert len(read_label(seq)["exist"]) == 2


def test_global_mode_missing_file_skipped(tmp_path):
    _build_global(tmp_path / "in", n_images=3)
    (tmp_path / "in" / "train2017" / f"{2:012d}.jpg").unlink()
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out, "--seq-len", "10")
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "train2017_seq0001"
    lab = read_label(seq)
    assert len(lab["exist"]) == 2
    assert [row[0] for row in lab["gt_rect"]] == [1.0, 3.0]
    man = json.loads((out / "manifest.json").read_text())
    assert man["sequences"][0]["n_skipped_missing_files"] == 1


def test_both_structures_prefers_per_video(tmp_path):
    _build_global(tmp_path / "in", n_images=3)          # global structure present
    build_video(tmp_path / "in", "videoPV", [           # AND a per-video subfolder
        ("a.jpg", [([1.0, 1.0, 3.0, 3.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    man = json.loads((out / "manifest.json").read_text())
    assert man["mode"] == "per_video"
    assert man["totals"]["n_sequences"] == 1
    assert man["sequences"][0]["name"] == "videoPV"
    assert any("per-video" in w.lower() or "per_video" in w.lower()
               for w in man["warnings"] + [man.get("mode_note", "")])


# --------------------------------------------------------------------------
# label alignment when disk-only frames interleave with json frames
# --------------------------------------------------------------------------

def build_colored_video(root, name, specs):
    """specs: list of (file_name, bbox-or-None).  Each frame gets a distinct
    solid color so per-frame alignment can be verified by image content
    (jpg sources are copied byte-for-byte, so bytes identify the source)."""
    vid = Path(root) / name
    vid.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i, (fn, box) in enumerate(specs, start=1):
        write_image(vid / fn, color=(10 * i, 40 + 10 * i, 240 - 10 * i))
        images.append(img_entry(i, fn))
        if box is not None:
            annotations.append(ann(i, i, list(box)))
    j = write_coco(vid / "instances.json", images, annotations)
    return vid, j


def test_disk_only_frame_sorting_before_json_frames_stays_aligned(tmp_path):
    vid, _ = build_colored_video(tmp_path / "in", "videoIL1", [
        ("img_0002.jpg", [2.0, 2.0, 5.0, 5.0]),
        ("img_0003.jpg", [3.0, 3.0, 5.0, 5.0]),
        ("img_0004.jpg", [4.0, 4.0, 5.0, 5.0]),
    ])
    # unannotated frame on disk that sorts BEFORE every json frame
    write_image(vid / "img_0001.jpg", color=(255, 1, 7))
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "videoIL1"
    lab = read_label(seq)
    assert lab["exist"] == [0, 1, 1, 1]
    assert lab["gt_rect"] == [[0, 0, 0, 0],
                              [2.0, 2.0, 5.0, 5.0],
                              [3.0, 3.0, 5.0, 5.0],
                              [4.0, 4.0, 5.0, 5.0]]
    # per-frame alignment verified by content: output i is byte-identical to
    # the source frame whose row sits at index i
    for i, fn in enumerate(["img_0001.jpg", "img_0002.jpg",
                            "img_0003.jpg", "img_0004.jpg"], start=1):
        assert (seq / f"{i:06d}.jpg").read_bytes() == (vid / fn).read_bytes(), fn
    man = json.loads((out / "manifest.json").read_text())
    assert any("disk-only" in w for w in man["warnings"])


def test_disk_only_frame_sorting_between_json_frames_stays_aligned(tmp_path):
    vid, _ = build_colored_video(tmp_path / "in", "videoIL2", [
        ("img_0001.jpg", [1.0, 1.0, 5.0, 5.0]),
        ("img_0003.jpg", [3.0, 3.0, 5.0, 5.0]),
        ("img_0004.jpg", [4.0, 4.0, 5.0, 5.0]),
    ])
    # unannotated frame on disk that sorts BETWEEN json frames
    write_image(vid / "img_0002.jpg", color=(255, 2, 7))
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "videoIL2"
    lab = read_label(seq)
    assert lab["exist"] == [1, 0, 1, 1]
    assert lab["gt_rect"] == [[1.0, 1.0, 5.0, 5.0],
                              [0, 0, 0, 0],
                              [3.0, 3.0, 5.0, 5.0],
                              [4.0, 4.0, 5.0, 5.0]]
    for i, fn in enumerate(["img_0001.jpg", "img_0002.jpg",
                            "img_0003.jpg", "img_0004.jpg"], start=1):
        assert (seq / f"{i:06d}.jpg").read_bytes() == (vid / fn).read_bytes(), fn
    man = json.loads((out / "manifest.json").read_text())
    assert any("disk-only" in w and "interleave" in w.lower()
               for w in man["warnings"])


# --------------------------------------------------------------------------
# --category semantics (typo protection)
# --------------------------------------------------------------------------

def test_category_no_value_matches_hard_error(tmp_path):
    build_video(tmp_path / "in", "videoCat", [
        ("img_0001.jpg", [([1.0, 1.0, 5.0, 5.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out,
                      "--category", "drone")
    assert r.returncode == 2
    assert "drone" in r.stderr
    assert "uav" in r.stderr          # valid category names are listed
    assert not out.exists()           # nothing materialized


def test_category_json_without_categories_key_no_filter(tmp_path):
    build_video(tmp_path / "in", "videoNoCat", [
        ("img_0001.jpg", [([1.0, 1.0, 5.0, 5.0], 3, 0)]),
    ], categories=None)               # json omits the categories key entirely
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out,
                      "--category", "uav")
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoNoCat")
    assert lab["exist"] == [1]
    assert lab["gt_rect"] == [[1.0, 1.0, 5.0, 5.0]]
    man = json.loads((out / "manifest.json").read_text())
    assert any("no categories key" in w for w in man["warnings"])


def test_category_mixed_valid_and_invalid_keeps_valid(tmp_path):
    build_video(tmp_path / "in", "videoMix", [
        ("img_0001.jpg", [([1.0, 1.0, 5.0, 5.0], 3, 0)]),
    ])
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out,
                      "--category", "uav", "--category", "drone")
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoMix")
    assert lab["exist"] == [1]
    assert lab["gt_rect"] == [[1.0, 1.0, 5.0, 5.0]]
    man = json.loads((out / "manifest.json").read_text())
    assert any("drone" in w and "ignored" in w for w in man["warnings"])


# --------------------------------------------------------------------------
# width/height given as JSON strings must not crash (coerced / decoded)
# --------------------------------------------------------------------------

def test_width_height_as_strings_coerced_not_crash(tmp_path):
    vid = tmp_path / "in" / "videoStr"
    vid.mkdir(parents=True)
    images, annotations = [], []
    specs = [
        ("img_0001.jpg", "32", "24", [1.0, 1.0, 5.0, 5.0]),       # plain strings
        ("img_0002.jpg", "32", "24", [30.0, 1.0, 10.0, 5.0]),     # clips at w=32
        ("img_0003.jpg", "garbage", "24", [2.0, 2.0, 4.0, 4.0]),  # bad string -> decode
    ]
    for i, (fn, w, h, box) in enumerate(specs, start=1):
        write_image(vid / fn)
        images.append(img_entry(i, fn, w=w, h=h))
        annotations.append(ann(i, i, box))
    write_coco(vid / "instances.json", images, annotations)
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    lab = read_label(out / "train" / "videoStr")
    assert lab["exist"] == [1, 1, 1]
    assert lab["gt_rect"][0] == [1.0, 1.0, 5.0, 5.0]
    assert lab["gt_rect"][1] == [30.0, 1.0, 2.0, 5.0]   # clipped via coerced width
    assert lab["gt_rect"][2] == [2.0, 2.0, 4.0, 4.0]    # size read from the image


# --------------------------------------------------------------------------
# frame decode verification (.jpg/.jpeg sources must be decodable)
# --------------------------------------------------------------------------

GARBAGE = b"THIS IS NOT A REAL JPEG IMAGE " * 12


def _build_video_with_garbage_frame(root, name, with_sizes=True):
    """3-frame video whose middle frame is garbage bytes behind a .jpg name,
    with json width/height present (the fast path that used to trust it)."""
    vid = Path(root) / name
    vid.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    boxes = [[1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 5.0, 5.0], [3.0, 3.0, 5.0, 5.0]]
    for i, box in enumerate(boxes, start=1):
        fn = f"img_{i:04d}.jpg"
        if i == 2:
            (vid / fn).write_bytes(GARBAGE)
        else:
            write_image(vid / fn, color=(10 * i, 40 + 10 * i, 240 - 10 * i))
        if with_sizes:
            images.append(img_entry(i, fn))
        else:
            images.append(img_entry(i, fn, w=None, h=None))
        annotations.append(ann(i, i, box))
    write_coco(vid / "instances.json", images, annotations)
    return vid


def test_undecodable_jpg_skipped_by_default_with_warning(tmp_path):
    vid = _build_video_with_garbage_frame(tmp_path / "in", "videoUVD")
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "videoUVD"
    # the garbage frame is dropped entirely: 2 jpgs, 2 rows, still in lockstep
    jpgs = sorted(p.name for p in seq.glob("*.jpg"))
    assert jpgs == ["000001.jpg", "000002.jpg"]
    lab = read_label(seq)
    assert lab["exist"] == [1, 1]
    assert lab["gt_rect"] == [[1.0, 1.0, 5.0, 5.0], [3.0, 3.0, 5.0, 5.0]]
    assert (seq / "000002.jpg").read_bytes() == (vid / "img_0003.jpg").read_bytes()
    man = json.loads((out / "manifest.json").read_text())
    assert man["sequences"][0]["n_undecodable"] == 1
    assert man["totals"]["n_undecodable"] == 1
    assert any("undecodable" in w and "img_0002.jpg" in w for w in man["warnings"])


def test_undecodable_jpg_strict_aborts(tmp_path):
    _build_video_with_garbage_frame(tmp_path / "in", "videoUVD")
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out, "--strict")
    assert r.returncode == 2
    assert "img_0002.jpg" in r.stderr
    assert "undecodable" in r.stderr or "not a decodable" in r.stderr


def test_no_verify_frames_flag_restores_trust_by_extension(tmp_path):
    vid = _build_video_with_garbage_frame(tmp_path / "in", "videoUVD")
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out,
                      "--no-verify-frames")
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "videoUVD"
    assert len(list(seq.glob("*.jpg"))) == 3
    assert (seq / "000002.jpg").read_bytes() == GARBAGE
    lab = read_label(seq)
    assert lab["exist"] == [1, 1, 1]      # garbage frame trusted again


def test_undecodable_without_json_size_skips_instead_of_abort(tmp_path):
    # json entries LACK width/height and one frame is garbage: previously this
    # aborted the whole run (ConverterError from the size fallback)
    _build_video_with_garbage_frame(tmp_path / "in", "videoUVN", with_sizes=False)
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "videoUVN"
    lab = read_label(seq)
    assert lab["exist"] == [1, 1]
    assert lab["gt_rect"] == [[1.0, 1.0, 5.0, 5.0], [3.0, 3.0, 5.0, 5.0]]
    man = json.loads((out / "manifest.json").read_text())
    assert man["sequences"][0]["n_undecodable"] == 1


# --------------------------------------------------------------------------
# corrupt / non-COCO jsons and json-less subfolders are reported, not silent
# --------------------------------------------------------------------------

def _build_dirty_input(root):
    build_video(root / "in", "videoGood", [
        ("g1.jpg", [([1.0, 1.0, 3.0, 3.0], 3, 0)]),
        ("g2.jpg", [([2.0, 1.0, 3.0, 3.0], 3, 0)]),
    ])
    vc = root / "in" / "videoCorrupt"
    vc.mkdir(parents=True)
    write_image(vc / "c1.jpg")
    (vc / "instances.json").write_text('{"images": [{"id": 1, "file_na')  # truncated
    vw = root / "in" / "videoWeird"
    vw.mkdir(parents=True)
    write_image(vw / "w1.jpg")
    (vw / "meta.json").write_text(json.dumps({"info": {"year": 2026}}))  # parses, not COCO
    vn = root / "in" / "videoNoJson"
    vn.mkdir(parents=True)
    write_image(vn / "n1.jpg")
    write_image(vn / "n2.jpg")


def test_corrupt_and_jsonless_subfolders_reported_not_silent(tmp_path):
    _build_dirty_input(tmp_path)
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    assert [p.name for p in seq_dirs(out)] == ["videoGood"]
    man = json.loads((out / "manifest.json").read_text())
    skipped = {s["path"]: s["reason"] for s in man["skipped_sources"]}
    assert any("videoCorrupt" in p for p in skipped)
    assert any("videoWeird" in p for p in skipped)
    assert any("videoNoJson" in p for p in skipped)
    joined = " ".join(man["warnings"])
    assert "failed to parse" in joined
    assert "not a COCO instances file" in joined
    assert "no readable COCO json" in joined and "videoNoJson" in joined


def test_strict_aborts_on_corrupt_json(tmp_path):
    _build_dirty_input(tmp_path)
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out, "--strict")
    assert r.returncode == 2
    assert "videoCorrupt" in r.stderr or "failed to parse" in r.stderr


# --------------------------------------------------------------------------
# re-runs into an existing --output reconcile (no stale dirs, no split leakage)
# --------------------------------------------------------------------------

def _build_four_videos(root, n=10):
    for v in "ABCD":
        build_video(root, "vid" + v, [
            (f"{v.lower()}{i:04d}.jpg", [([1.0, 1.0, 3.0, 3.0], 3, 0)])
            for i in range(1, n + 1)
        ])


def test_rerun_min_visible_prunes_stale_sequence_dirs(tmp_path):
    _build_four_videos(tmp_path / "in")
    out = tmp_path / "out"
    r1 = run_converter("--input", tmp_path / "in", "--output", out)
    assert r1.returncode == 0, r1.stderr
    assert len(seq_dirs(out)) == 4
    # second run drops every sequence via --min-visible: the output must not
    # keep training on the stale dirs
    r2 = run_converter("--input", tmp_path / "in", "--output", out,
                       "--min-visible", "20")
    assert r2.returncode == 0, r2.stderr
    assert seq_dirs(out) == []
    man = json.loads((out / "manifest.json").read_text())
    assert len(man["pruned"]) == 4
    assert {p["name"] for p in man["pruned"]} == {"vidA", "vidB", "vidC", "vidD"}
    assert "pruned" in r2.stdout


def test_rerun_val_ratio_change_never_leaves_sequence_in_two_splits(tmp_path):
    _build_four_videos(tmp_path / "in")
    out = tmp_path / "out"
    r1 = run_converter("--input", tmp_path / "in", "--output", out,
                       "--val-ratio", "0.5")          # val: vidA, vidC
    assert r1.returncode == 0, r1.stderr
    assert [p.name for p in seq_dirs(out, "val")] == ["vidA", "vidC"]
    r2 = run_converter("--input", tmp_path / "in", "--output", out,
                       "--val-ratio", "0.25")         # val: vidA only
    assert r2.returncode == 0, r2.stderr
    val_names = [p.name for p in seq_dirs(out, "val")]
    train_names = [p.name for p in seq_dirs(out, "train")]
    assert val_names == ["vidA"]
    assert train_names == ["vidB", "vidC", "vidD"]
    assert not (set(val_names) & set(train_names))    # no train/val leakage
    man = json.loads((out / "manifest.json").read_text())
    assert {"split": "val", "name": "vidC"} in man["pruned"]


# --------------------------------------------------------------------------
# --output nested inside the input tree must be excluded from scanning
# --------------------------------------------------------------------------

def _snapshot(out):
    seq = out / "train" / "videoA"
    snaps = {
        "label": read_label(seq),
        "jpgs": {p.name: p.read_bytes() for p in sorted(seq.glob("*.jpg"))},
    }
    return snaps


def test_output_nested_in_video_subfolder_rerun_twice_copy(tmp_path):
    # exact crash repro from the review: copy mode, output inside the
    # per-video subfolder -> run 2 used to ingest its own output and crash
    build_colored_video(tmp_path / "in", "videoA", [
        (f"frame_{i:03d}.jpg", [float(i), 1.0, 4.0, 4.0]) for i in range(1, 7)
    ])
    out = tmp_path / "in" / "videoA" / "out"
    r1 = run_converter("--input", tmp_path / "in", "--output", out)
    assert r1.returncode == 0, r1.stderr
    snap1 = _snapshot(out)
    assert len(snap1["jpgs"]) == 6
    r2 = run_converter("--input", tmp_path / "in", "--output", out)
    assert r2.returncode == 0, r2.stderr
    assert _snapshot(out) == snap1                   # identical second run
    assert any("inside the input tree" in w for w in
               json.loads((out / "manifest.json").read_text())["warnings"])


def test_output_nested_with_symlink_png_rerun_stays_clean(tmp_path):
    # the silent-corruption variant: PNG sources + --symlink
    vid = tmp_path / "in" / "videoP"
    vid.mkdir(parents=True)
    images, annotations = [], []
    for i in range(1, 5):
        fn = f"p{i}.png"
        write_image(vid / fn, color=(10 * i, 50, 200))
        images.append(img_entry(i, fn))
        annotations.append(ann(i, i, [float(i), 1.0, 4.0, 4.0]))
    write_coco(vid / "instances.json", images, annotations)
    out = tmp_path / "in" / "videoP" / "out"
    for _ in range(2):
        r = run_converter("--input", tmp_path / "in", "--output", out, "--symlink")
        assert r.returncode == 0, r.stderr
        seq = out / "train" / "videoP"
        jpgs = sorted(seq.glob("*.jpg"))
        assert len(jpgs) == 4                         # no self-ingested extras
        for p in jpgs:
            assert not p.is_symlink()                 # re-encoded, not dangling
            assert p.read_bytes()[:3] == JPEG_MAGIC   # actually readable
        lab = read_label(seq)
        assert len(lab["exist"]) == 4
        assert [row[0] for row in lab["gt_rect"]] == [1.0, 2.0, 3.0, 4.0]


def test_output_at_input_top_level_warned_and_not_scanned(tmp_path):
    build_video(tmp_path / "in", "videoA", [
        ("a.jpg", [([1.0, 1.0, 3.0, 3.0], 3, 0)]),
    ])
    out = tmp_path / "in" / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    assert "inside the input tree" in r.stderr
    man = json.loads((out / "manifest.json").read_text())
    assert any("inside the input tree" in w for w in man["warnings"])
    # the output dir itself must not be reported as a json-less video subfolder
    assert not any("no readable COCO json" in w and "out" in w
                   for w in man["warnings"])


def test_undecodable_disk_only_extra_dropped_and_warned(tmp_path):
    # a garbage file with no json entry: dropped from the extras too, counted
    # once, and actually named in a warning
    build_colored_video(tmp_path / "in", "videoUX", [
        ("img_0001.jpg", [1.0, 1.0, 5.0, 5.0]),
    ])
    (tmp_path / "in" / "videoUX" / "img_0002.jpg").write_bytes(GARBAGE)
    out = tmp_path / "out"
    r = run_converter("--input", tmp_path / "in", "--output", out)
    assert r.returncode == 0, r.stderr
    seq = out / "train" / "videoUX"
    assert len(list(seq.glob("*.jpg"))) == 1
    lab = read_label(seq)
    assert lab["exist"] == [1]
    man = json.loads((out / "manifest.json").read_text())
    assert man["sequences"][0]["n_undecodable"] == 1
    assert any("undecodable" in w and "img_0002.jpg" in w for w in man["warnings"])
