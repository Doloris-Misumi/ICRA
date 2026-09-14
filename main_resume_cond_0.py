"""Resume a verified interrupted clean official-test evaluation in place."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def atomic_write_json(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary_path, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--conf-thr', type=float, default=0.3)
    parser.add_argument('--failed-ledger', type=Path, required=True)
    parser.add_argument('--resume-verification', type=Path, required=True)
    parser.add_argument('--run-path-file', type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    ledger_path = args.failed_ledger.resolve()
    verification_path = args.resume_verification.resolve()
    for path in (config_path, checkpoint_path, ledger_path, verification_path):
        require(path.is_file(), f'missing required artifact: {path}')
    require(abs(args.conf_thr - 0.3) <= 1e-12, 'resume confidence threshold must be 0.3')

    verification = load_json(verification_path)
    require(verification.get('status') == 'pass', 'resume verification did not pass')
    require(Path(verification['authorized_failed_ledger']).resolve() == ledger_path,
            'resume verification authorizes a different ledger')
    require(verification['failed_ledger_sha256'] == sha256_file(ledger_path),
            'failed ledger changed after resume verification')
    require(Path(verification['config']).resolve() == config_path,
            'resume verification refers to a different config')
    require(verification['config_sha256'] == sha256_file(config_path),
            'resume config hash mismatch')
    require(Path(verification['checkpoint']).resolve() == checkpoint_path,
            'resume verification refers to a different checkpoint')
    require(verification['checkpoint_sha256'] == sha256_file(checkpoint_path),
            'resume checkpoint hash mismatch')
    require(int(verification['total_official_test_frames']) == 17536,
            'resume verification test size mismatch')
    resume_from = int(verification['resume_from_index'])
    require(0 < resume_from < 17536, f'invalid verified resume index: {resume_from}')
    for relative_path, expected_hash in verification['resume_source_sha256'].items():
        source_path = project_root / relative_path
        require(sha256_file(source_path) == expected_hash,
                f'resume source changed after verification: {relative_path}')

    # All artifact checks precede imports that initialize CUDA/Numba.
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pipeline = PipelineDetection_v1_0(path_cfg=str(config_path), mode='test')
    pipeline.load_dict_model(str(checkpoint_path), is_strict=True)
    runtime_log_dir = Path(pipeline.path_log).resolve()
    shutil.copy2(Path(__file__).resolve(), runtime_log_dir / 'executed_resume_code.txt')
    if args.run_path_file is not None:
        run_path_file = args.run_path_file.resolve()
        run_path_file.parent.mkdir(parents=True, exist_ok=True)
        run_path_file.write_text(str(runtime_log_dir) + '\n', encoding='utf-8')

    ledger = load_json(ledger_path)
    require(ledger.get('status') == 'failed', 'official-test ledger is not failed')
    prefix_eval_dir = Path(verification['evaluation_epoch_directory']).resolve()
    interrupted_event = {
        'status': ledger['status'],
        'started_at': ledger['started_at'],
        'failed_at': ledger['finished_at'],
        'error_type': ledger['error_type'],
        'error': ledger['error'],
        'completed_prefix_samples': resume_from,
        'failed_ledger_sha256_before_resume': verification['failed_ledger_sha256'],
    }
    ledger.update({
        'status': 'resumed',
        'resume_started_at': datetime.datetime.now().astimezone().isoformat(),
        'resume_from_index': resume_from,
        'resume_verification': str(verification_path),
        'resume_verification_sha256': sha256_file(verification_path),
        'resume_runtime_log_directory': str(runtime_log_dir),
        'interrupted_evaluation': interrupted_event,
        'completed_via_verified_resume': True,
    })
    atomic_write_json(ledger_path, ledger)

    try:
        evaluation_results = pipeline.validate_kitti_clean(
            epoch=args.epoch,
            list_conf_thr=[args.conf_thr],
            eval_split='test',
            conditional=True,
            resume_from=resume_from,
            resume_path_dir=prefix_eval_dir,
        )
    except BaseException as exc:
        ledger['status'] = 'failed'
        ledger['resume_failed_at'] = datetime.datetime.now().astimezone().isoformat()
        ledger['resume_error_type'] = type(exc).__name__
        ledger['resume_error'] = str(exc)
        atomic_write_json(ledger_path, ledger)
        raise

    ledger['status'] = 'complete'
    ledger['finished_at'] = datetime.datetime.now().astimezone().isoformat()
    ledger['resume_finished_at'] = ledger['finished_at']
    ledger['num_samples'] = int(evaluation_results['num_samples'])
    ledger['failure_count'] = int(evaluation_results['failure_count'])
    ledger['interrupted_inference_failures'] = 1
    ledger['successful_samples_before_resume'] = resume_from
    ledger['successful_samples_after_resume'] = 17536 - resume_from
    ledger['protocol_artifact'] = str(
        prefix_eval_dir / 'evaluation_protocol_and_results.json'
    )
    atomic_write_json(ledger_path, ledger)
    print(f'* Resumed official-test ledger complete: {ledger_path}')

    if pipeline.is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
