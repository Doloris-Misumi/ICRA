# Anonymous implementation: Weather-Conditioned Branch Routing

This review snapshot contains the current no-prompt detector implementation, training configuration, evaluation adapter and accelerated inference benchmark. Camera features condition the detector during both training and inference. Text prompts and contrastive alignment are disabled in the supplied configurations. Optional legacy modules remain for import/checkpoint compatibility; they are not evidence of an active prompt path.

## Environment

The experiments used Python 3.8, PyTorch 1.10.1+cu113, torchvision 0.11.2+cu113 and spconv-cu113 2.1.25. A compatible CUDA toolchain is required for the rotated-IoU extension. Install dependencies using `python -m pip install -r requirements.txt`. The requirements include runtime dependencies previously inherited from the development environment; installation in a fresh environment has not yet been certified.

Run commands from the repository root. Set `CUDA_VISIBLE_DEVICES` for your machine. Do not use a multi-GPU run to reproduce the supplied single-GPU training results without documenting the change.

## Data and weights

Obtain K-Radar independently from its official distribution. No dataset, annotations, checkpoints, training logs or compiled binaries are bundled. Place or link the dataset at `data/k_radar_dataset`, or edit `DATASET.DIR.LIST_DIR` in the configuration you use.

The shared Stage 1 checkpoint belongs at `checkpoints/stage1_weather.pth`. Its expected checksum and the known limits of its training lineage are in `configs/icra_next_260907/stage1_shared_legacy.json`. The checkpoint is not included and no download is currently provided by this snapshot. Training a replacement is possible, but does not reproduce the same fixed frontend. Detector checkpoints must likewise be supplied separately.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python main_train_0.py --config configs/train_full.yml
```

## Evaluate the main table

```bash
python tools/prepare_test_protocol.py --dataset-root data/k_radar_dataset
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_paper.py \
  --config configs/eval_full.yml \
  --checkpoint checkpoints/model_16.pt --epoch 16
```

The preparation command rebuilds annotations from the local dataset, without redistributing them. It uses the supplied 10,065-frame manifest and calibrated, inclusive Sedan ROI filtering. The generated annotations were checked against every annotation used by the main-table experiment: all 10,065 frames and 18,874 objects match exactly. See `docs/evaluation_protocol.md`.

Do not substitute the generic `main_cond_0.py` entry point for this adapter: that entry point serves other development protocols and is retained for compatibility.

## Accelerated inference benchmark

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
python tools/benchmark_wcbr_fast_post_cached.py \
  --config configs/benchmark_fast.yml \
  --checkpoint checkpoints/model_16.pt \
  --output outputs/benchmark.json --verify
```

Create `outputs/` first with `mkdir -p outputs`. Use a new output filename for each run. The benchmark samples 120 frames from the parent 17,536-frame test list: 20 warm-up and 100 timed frames. It uses batch size 1, FP32 with backend TF32 settings, includes model preprocessing/forward and postprocessing, and excludes data loading. `--verify` checks the optimized postprocessing against its original counterpart on the benchmark frames. Exact KNN does not imply bitwise equivalence of the entire accelerated backbone.

The original backbone and accelerated implementation are separate files. The main AP protocol and the timing protocol are not interchangeable; the full AP table was not regenerated with the latest cached postprocessing optimization.

## Loss ablations

`configs/ablations/` supplies 10-epoch loss-weight variants with the same 20-epoch learning-rate horizon. Run the training entry point with the selected configuration, then evaluate `model_9.pt` using `tools/evaluate_paper.py --epoch 9` and that configuration. Validation is disabled in these short runs and testing is an explicit separate command. The model module implementations have not been changed for this release.

## Review snapshot

No original Git history or author-identifying project links are included in the files. Third-party source notices and licenses are retained. This snapshot is a source release, not a claim that model weights are bundled or that a fresh-environment full training run has been verified.
