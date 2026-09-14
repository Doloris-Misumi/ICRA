"""Evaluate the paper's 10,065-frame legacy protocol; no training or refit."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/eval_full.yml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--labels', default='artifacts/test_gt.json')
    parser.add_argument('--epoch', type=int, default=16, help='Zero-based checkpoint epoch')
    args = parser.parse_args()
    sys.path.insert(0,str(ROOT))
    os.chdir(ROOT)
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
    labels = json.loads(Path(args.labels).read_text())
    expected = set((ROOT/'resources/split/test_10065.txt').read_text().splitlines())
    assert set(labels) == expected and len(labels) == 10065
    pline = PipelineDetection_v1_0(path_cfg=args.config, mode='test')
    dataset = pline.dataset_test
    assert len(dataset) == 17536 and dataset.type_item == 1
    def frame_id(path):
        p = Path(path)
        return p.parent.parent.name+','+p.name
    dataset.list_path_label = [p for p in dataset.list_path_label if frame_id(p) in expected]
    assert len(dataset) == 10065
    def get_labels(self, path_label, calib_info):
        return [(cls,self.dict_cls_id[cls],list(box),int(track[0]))
                for cls,box,track,avail in labels[frame_id(path_label)]]
    dataset.get_label_bboxes = types.MethodType(get_labels,dataset)
    pline.load_dict_model(args.checkpoint,is_strict=True)
    result=pline.validate_kitti_clean(epoch=args.epoch,list_conf_thr=[0.3],eval_split='test',conditional=True)
    assert result['num_samples']==10065 and result['failure_count']==0
    print('Results:',pline.path_log)

if __name__ == '__main__':
    main()
