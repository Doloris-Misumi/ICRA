import os
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import yaml


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary_path, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the experiment config file.',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to the trained model checkpoint.',
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Epoch number used only for naming/logging in validation outputs.',
    )
    parser.add_argument(
        '--conf-thr',
        type=float,
        default=0.3,
        help='Confidence threshold for validation.',
    )
    parser.add_argument(
        '--subset',
        action='store_true',
        help='Run conditional validation on the subset split if supported.',
    )
    parser.add_argument(
        '--official-test-ledger',
        default=None,
        help='Exclusive ledger path for the one-shot clean official-test evaluation.',
    )
    parser.add_argument(
        '--refit-verification',
        default=None,
        help='Required Stage-2 refit completion-pass artifact for clean official testing.',
    )
    parser.add_argument(
        '--preflight-recovery-verification',
        default=None,
        help=(
            'Optional pass artifact proving an earlier process failed before any '
            'official-test sample reached inference. Required when using a ledger '
            'path different from VAL.OFFICIAL_TEST_LEDGER.'
        ),
    )
    parser.add_argument(
        '--run-path-file',
        default=None,
        help='Optional file that receives the created evaluation log directory.',
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    with config_path.open('r', encoding='utf-8') as file_obj:
        raw_config = yaml.safe_load(file_obj)
    clean_from_file = bool(raw_config.get('VAL', {}).get('USE_CLEAN_EVALUATOR', False))
    if clean_from_file and not bool(
        raw_config.get('VAL', {}).get('OFFICIAL_TEST_ONCE_GUARD', False)
    ):
        raise ValueError('formal clean official test requires OFFICIAL_TEST_ONCE_GUARD=True')
    refit_verification = None
    refit_verification_path = None
    if clean_from_file:
        if args.refit_verification is None or not str(args.refit_verification).strip():
            raise ValueError('formal clean official test requires --refit-verification')
        refit_verification_path = Path(args.refit_verification).resolve()
        if not refit_verification_path.is_file():
            raise FileNotFoundError(refit_verification_path)
        with refit_verification_path.open('r', encoding='utf-8') as file_obj:
            refit_verification = json.load(file_obj)
        if refit_verification.get('status') != 'pass':
            raise ValueError('Stage-2 refit completion verification did not pass')
        if Path(refit_verification['refit_config']).resolve() != config_path:
            raise ValueError('refit verification refers to a different evaluation config')
        if refit_verification['refit_config_sha256'] != sha256_file(config_path):
            raise ValueError('refit verification config hash mismatch')
        if Path(refit_verification['final_model']).resolve() != checkpoint_path:
            raise ValueError('official-test checkpoint is not the verified final refit model')
        if refit_verification['final_model_sha256'] != sha256_file(checkpoint_path):
            raise ValueError('verified final refit model hash mismatch')
        if int(refit_verification.get('official_test_evaluations_during_refit', -1)) != 0:
            raise ValueError('refit verification does not prove zero official-test evaluations')
        if not bool(refit_verification.get('official_test_ledger_absent', False)):
            raise ValueError('refit verification does not prove an unused official-test ledger')

    # Importing the evaluator stack can initialize Numba CUDA kernels.  Keep it
    # after all fail-closed artifact checks so an invalid formal-test request
    # cannot touch CUDA or create a logging directory.
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pline = PipelineDetection_v1_0(path_cfg=str(config_path), mode='test')
    
    # Only copy code if logging is enabled (Rank 0)
    import shutil
    if getattr(pline, 'is_logging', False):
        shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
        if args.run_path_file is not None:
            run_path_file = Path(args.run_path_file).resolve()
            run_path_file.parent.mkdir(parents=True, exist_ok=True)
            run_path_file.write_text(pline.path_log + '\n', encoding='utf-8')
        
    use_clean_evaluator = bool(pline.cfg.VAL.get('USE_CLEAN_EVALUATOR', False))
    pline.load_dict_model(str(checkpoint_path), is_strict=use_clean_evaluator)
    
    if use_clean_evaluator:
        if args.subset:
            raise ValueError('formal clean evaluation does not permit --subset')
        configured_confidence = [
            float(value) for value in pline.cfg.VAL.get('LIST_VAL_CONF_THR', [0.3])
        ]
        if configured_confidence != [0.3] or abs(float(args.conf_thr) - 0.3) > 1e-12:
            raise ValueError(
                'formal clean official test is fixed to confidence threshold 0.3; '
                f'config={configured_confidence}, argument={args.conf_thr}'
            )
        ledger_value = args.official_test_ledger
        if ledger_value is None:
            ledger_value = raw_config.get('VAL', {}).get('OFFICIAL_TEST_LEDGER', None)
        if ledger_value is None or not str(ledger_value).strip():
            raise ValueError('formal clean official test requires an exclusive ledger path')
        ledger_path = Path(str(ledger_value)).expanduser()
        if not ledger_path.is_absolute():
            ledger_path = Path(__file__).resolve().parent / ledger_path
        ledger_path = ledger_path.resolve()
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        configured_ledger_path = Path(
            str(raw_config.get('VAL', {}).get('OFFICIAL_TEST_LEDGER', ''))
        ).expanduser()
        if not configured_ledger_path.is_absolute():
            configured_ledger_path = Path(__file__).resolve().parent / configured_ledger_path
        configured_ledger_path = configured_ledger_path.resolve()
        preflight_recovery = None
        preflight_recovery_path = None
        if ledger_path != configured_ledger_path:
            if args.preflight_recovery_verification is None:
                raise ValueError(
                    'a non-configured official-test ledger requires '
                    '--preflight-recovery-verification'
                )
            preflight_recovery_path = Path(args.preflight_recovery_verification).resolve()
            if not preflight_recovery_path.is_file():
                raise FileNotFoundError(preflight_recovery_path)
            with preflight_recovery_path.open('r', encoding='utf-8') as file_obj:
                preflight_recovery = json.load(file_obj)
            if preflight_recovery.get('status') != 'pass':
                raise ValueError('preflight recovery verification did not pass')
            if Path(preflight_recovery['authorized_recovery_ledger']).resolve() != ledger_path:
                raise ValueError('preflight recovery artifact authorizes a different ledger')
            if Path(preflight_recovery['refit_verification']).resolve() != refit_verification_path:
                raise ValueError('preflight recovery artifact refers to a different refit verification')
            if preflight_recovery['refit_verification_sha256'] != sha256_file(refit_verification_path):
                raise ValueError('preflight recovery refit-verification hash mismatch')
            failed_ledger_path = Path(preflight_recovery['failed_ledger']).resolve()
            if not failed_ledger_path.is_file():
                raise FileNotFoundError(failed_ledger_path)
            if preflight_recovery['failed_ledger_sha256'] != sha256_file(failed_ledger_path):
                raise ValueError('preserved failed-ledger hash mismatch')
            if int(preflight_recovery.get('test_frames_reaching_inference', -1)) != 0:
                raise ValueError('preflight artifact does not prove zero evaluated test frames')
            if preflight_recovery.get('metrics_produced') is not False:
                raise ValueError('preflight artifact does not prove zero produced metrics')
        ledger = {
            'status': 'started',
            'started_at': datetime.datetime.now().astimezone().isoformat(),
            'config': str(config_path),
            'config_sha256': sha256_file(config_path),
            'detector_checkpoint': str(checkpoint_path),
            'detector_checkpoint_sha256': sha256_file(checkpoint_path),
            'stage1_checkpoint': str(pline.cfg.MODEL.IMG_CLS.MODEL_PATH),
            'stage1_checkpoint_sha256': sha256_file(pline.cfg.MODEL.IMG_CLS.MODEL_PATH),
            'confidence_threshold': float(args.conf_thr),
            'epoch_label': args.epoch,
            'eval_split': 'test',
            'conditional_breakdown': True,
            'log_directory': pline.path_log,
            'refit_verification': str(refit_verification_path),
            'refit_verification_sha256': sha256_file(refit_verification_path),
        }
        if preflight_recovery is not None:
            ledger.update({
                'preflight_recovery_verification': str(preflight_recovery_path),
                'preflight_recovery_verification_sha256': sha256_file(preflight_recovery_path),
                'preserved_failed_ledger': preflight_recovery['failed_ledger'],
                'preserved_failed_ledger_sha256': preflight_recovery['failed_ledger_sha256'],
                'prior_test_frames_reaching_inference': 0,
            })
        try:
            with ledger_path.open('x', encoding='utf-8') as file_obj:
                json.dump(ledger, file_obj, indent=2, sort_keys=True)
                file_obj.write('\n')
        except FileExistsError as exc:
            raise RuntimeError(
                f'official-test ledger already exists; refusing a second formal evaluation: {ledger_path}'
            ) from exc

        try:
            evaluation_results = pline.validate_kitti_clean(
                epoch=args.epoch,
                list_conf_thr=[args.conf_thr],
                eval_split='test',
                conditional=True,
            )
        except BaseException as exc:
            ledger['status'] = 'failed'
            ledger['finished_at'] = datetime.datetime.now().astimezone().isoformat()
            ledger['error_type'] = type(exc).__name__
            ledger['error'] = str(exc)
            atomic_write_json(ledger_path, ledger)
            raise
        ledger['status'] = 'complete'
        ledger['finished_at'] = datetime.datetime.now().astimezone().isoformat()
        ledger['num_samples'] = int(evaluation_results['num_samples'])
        ledger['failure_count'] = int(evaluation_results['failure_count'])
        epoch_name = 'none' if args.epoch is None else f'epoch_{args.epoch}'
        ledger['protocol_artifact'] = os.path.join(
            pline.path_log,
            'eval_kitti',
            'test',
            epoch_name,
            'evaluation_protocol_and_results.json',
        )
        atomic_write_json(ledger_path, ledger)
        print(f'* Official-test one-shot ledger complete: {ledger_path}')
    else:
        pline.validate_kitti_conditional(
            epoch=args.epoch,
            list_conf_thr=[args.conf_thr],
            is_subset=args.subset,
            is_print_memory=False,
        )
    
    if pline.is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()
