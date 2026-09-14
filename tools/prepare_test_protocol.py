"""Rebuild the paper's v1 label adapter from a local licensed K-Radar copy."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def build(dataset_root):
    records = {}
    selected = (ROOT / 'resources/split/test_10065.txt').read_text().splitlines()
    for frame in selected:
        seq, filename = frame.split(',')
        base = dataset_root / seq
        calib = list(map(float, (base/'info_calib/calib_radar_lidar.txt').read_text().splitlines()[1].split(',')))
        rows = []
        for line in (base/'info_label'/filename).read_text().splitlines()[1:]:
            parts = line.split(',')
            if parts[0] != '*':
                continue
            offset = 1 if len(parts) == 11 else 0
            if parts[2+offset][1:] != 'Sedan':
                continue
            x,y,z,theta,l,w,h = map(float,parts[3+offset:10+offset])
            x,y,z = x+calib[1], y+calib[2], z+0.7
            if not (0 <= x <= 72 and -6.4 <= y <= 6.4 and -2 <= z <= 6):
                continue
            rows.append(['Sedan',[x,y,z,theta*(math.pi/180.),2*l,2*w,2*h],
                         [int(parts[1]), int(parts[2])] if offset else [int(parts[1]), int(parts[1])],'R'])
        if not rows:
            raise ValueError('No retained Sedan label: '+frame)
        records[frame] = rows
    assert len(records) == 10065
    return records

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/test_gt.json')
    args = parser.parse_args()
    records = build(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2)+'\n')
    print('Frames:',len(records),'objects:',sum(map(len,records.values())))
