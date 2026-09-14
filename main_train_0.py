import os
import argparse
from pathlib import Path

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

PATH_CONFIG = './configs/cfg_rl_3df_gate.yml' 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', dest='local_rank_dash', type=int, default=-1)
    parser.add_argument(
        '--config',
        default=PATH_CONFIG,
        help='Path to the experiment config file.',
    )
    parser.add_argument(
        '--run-path-file',
        default=None,
        help='Optional file that receives the created training log directory.',
    )
    args, _ = parser.parse_known_args()

    local_rank = args.local_rank if args.local_rank != -1 else args.local_rank_dash
    if local_rank != -1 and os.environ.get('LOCAL_RANK') is None:
        os.environ['LOCAL_RANK'] = str(local_rank)

    pline = PipelineDetection_v1_0(path_cfg=args.config, mode='train')
    if args.run_path_file is not None and getattr(pline, 'is_logging', False):
        run_path_file = Path(args.run_path_file).resolve()
        run_path_file.parent.mkdir(parents=True, exist_ok=True)
        run_path_file.write_text(pline.path_log + '\n', encoding='utf-8')

    ### Save this file for checking ###
    import shutil
    if getattr(pline, 'is_logging', False):
        shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
    ### Save this file for checking ###

    pline.train_network()

    run_conditional_at_end = bool(pline.cfg.VAL.get('RUN_CONDITIONAL_AT_END', True))
    if run_conditional_at_end and ((not pline.is_distributed) or (pline.local_rank == 0)):
        list_conf_thr = pline.cfg.VAL.get('LIST_VAL_CONF_THR', [0.3])
        pline.validate_kitti_conditional(
            list_conf_thr=list_conf_thr,
            is_subset=False,
            is_print_memory=False
        )
    
    if pline.is_distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    
