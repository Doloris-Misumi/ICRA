"""Batch-one wall-clock inference benchmark; no AP evaluation or model changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
    output = Path(args.output).resolve()
    assert not output.exists()
    from models import backbone_3d
    from models.backbone_3d.rl_3df_fast_v2 import RL3DFBackbone_BranchingFastV2
    backbone_3d.__all__['RL3DFBackbone_BranchingFastV2'] = RL3DFBackbone_BranchingFastV2
    pline = PipelineDetection_v1_0(path_cfg=args.config, mode='test')
    pline.load_dict_model(args.checkpoint, is_strict=True)
    model = pline.network.eval()
    from utils.nms_fast_exact_cached import install
    old_post = install(model.list_modules[-1])
    verified = 0
    dataset = pline.dataset_test
    assert len(dataset) == 17536
    # Spread frames across the entire ordered test set, rather than one sequence.
    indices = np.linspace(0, len(dataset)-1, 120, dtype=int).tolist()
    assert len(set(indices)) == 120
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices),
        batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    iterator = iter(loader)
    rows = []
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    model_memory = torch.cuda.memory_allocated()/1024**3
    gpu_snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
    for i in range(120):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = next(iterator)
        t1 = time.perf_counter()
        events[0].record()
        with torch.no_grad():
            out = model(batch)
        events[1].record()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.no_grad():
            prediction = model.list_modules[-1].get_nms_pred_boxes_for_single_sample(out, 0.3, is_nms=True)
        if prediction is None:
            raise RuntimeError('postprocessing returned None')
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        event_ms = events[0].elapsed_time(events[1])
        if args.verify:
            import copy
            actual = copy.deepcopy({k: prediction[k] for k in ['pp_bbox','pp_cls','pp_num_bbox']})
            with torch.no_grad(): expected = old_post(out, 0.3, is_nms=True)
            for key in actual:
                assert actual[key] == expected[key], (i, key, actual[key], expected[key])
            verified += 1
        if i >= 20:
            desc = batch['meta'][0]['desc']
            rows.append(dict(dataset_index=indices[i], weather=desc['climate'],
                load_ms=(t1-t0)*1000, forward_wall_ms=(t2-t1)*1000,
                forward_cuda_event_ms=event_ms, postprocess_ms=(t3-t2)*1000,
                detector_ms=(t3-t1)*1000, pipeline_ms=(t3-t0)*1000))
        prediction = out = batch = None
        if i == 19:
            torch.cuda.reset_peak_memory_stats()
        if (i+1) % 10 == 0:
            print(f'BENCH {i+1}/120', flush=True)
    assert len(rows) == 100
    def stats(key):
        v=np.array([row[key] for row in rows])
        return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v,95)),
                    min=float(v.min()), max=float(v.max()), std=float(v.std(ddof=1)))
    summary={k:stats(k) for k in rows[0] if k.endswith('_ms')}
    result=dict(verified_frames=verified, config=args.config,config_sha256=digest(args.config),checkpoint=args.checkpoint,
        checkpoint_sha256=digest(args.checkpoint),script_sha256=digest(__file__),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),torch_num_threads=torch.get_num_threads(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        precision='FP32; no autocast; backend TF32 flags recorded',batch_size=1,num_workers=0,
        warmup=20,measured=100,sampling='120 uniformly spaced test indices; first20 warmup; remaining100 measured',
        indices=indices,params_m=sum(p.numel() for p in model.parameters())/1e6,
        model_allocated_gib=model_memory,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        timing_scope='forward includes camera, input preprocessing/H2D and CPU KNN; detector adds decoding/NMS/output CPU lists; pipeline adds synchronous dataset loading/collation including annotation I/O. Excludes startup, AP, output file writing, and batch cleanup.',
        gpu_snapshot_start=gpu_snapshot,
        source_sha256={str(p.relative_to(root)):digest(p) for folder in ['models','pipelines','datasets','utils'] for p in sorted((root/folder).rglob('*.py'))},
        stats=summary,rows=rows)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(summary),flush=True)

if __name__ == '__main__':
    main()
