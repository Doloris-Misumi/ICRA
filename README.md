# Anonymous implementation: Weather-Conditioned Branch Routing

This review snapshot contains the current no-prompt detector implementation, training configuration, evaluation adapter and accelerated inference benchmark. Camera features condition the detector during both training and inference.
## Environment

The experiments used Python 3.8, PyTorch 1.10.1+cu113, torchvision 0.11.2+cu113 and spconv-cu113 2.1.25. A compatible CUDA toolchain is required for the rotated-IoU extension. Install dependencies using `python -m pip install -r requirements.txt`. The requirements include runtime dependencies previously inherited from the development environment; installation in a fresh environment has not yet been certified.

Run commands from the repository root. Set `CUDA_VISIBLE_DEVICES` for your machine. Do not use a multi-GPU run to reproduce the supplied single-GPU training results without documenting the change.

## Data and weights

Obtain K-Radar independently from its official distribution. No dataset, annotations, checkpoints, training logs or compiled binaries are bundled. Place or link the dataset at `data/k_radar_dataset`, or edit `DATASET.DIR.LIST_DIR` in the configuration you use.



## Train

Stage1

```bash
CUDA_VISIBLE_DEVICES=0 python models/img_cls/train_stage1_3dlrf.py \
  --config configs/train_full.yml \
  --epochs 100 --batch-size 16 \
  --output-dir checkpoints/stage1
```

Stage2
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



## Accelerated inference benchmark

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
python tools/benchmark_wcbr_fast_post_cached.py \
  --config configs/benchmark_fast.yml \
  --checkpoint checkpoints/model_16.pt \
  --output outputs/benchmark.json --verify
```




## Review snapshot

No original Git history or author-identifying project links are included in the files. Third-party source notices and licenses are retained. This snapshot is a source release, not a claim that model weights are bundled or that a fresh-environment full training run has been verified.
