#!/bin/bash
# Fine-tune FocusTrack for thermal runway tracking (fTIR) in two stages.
# Stage 1 (focustrack_ftir_stage1): LSOTB_TIR + AntiUAV410 mixed fine-tune,
#   initialized from the trained RGB runway checkpoint (TRAIN.FULL_CKPT).
# Stage 2 (focustrack_ftir_stage2): adapt to the 1088x1088 thermal runway data
#   (RunwayTIR) with an overlapping patch embedding (STRIDE 12, PATCH_SIZE 16).
# Before running, point TRAIN.FULL_CKPT in both stage yamls at the real
# checkpoints; the paths shipped in the yamls are placeholders.
# Invocation mirrors the training command in README.md; GPU count via NGPU.

set -e

NGPU=${NGPU:-4}

# Stage 1: TIR + RGB fine-tune from the runway_sra checkpoint
python tracking/train.py --script focustrack --config focustrack_ftir_stage1 --save_dir ./output --mode multiple --nproc_per_node ${NGPU} --use_wandb 0

# Stage 2: adapt to the 1088x1088 thermal runway data
python tracking/train.py --script focustrack --config focustrack_ftir_stage2 --save_dir ./output --mode multiple --nproc_per_node ${NGPU} --use_wandb 0
