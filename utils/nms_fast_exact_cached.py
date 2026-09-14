"""Independent postprocessing adapter: unchanged rotated IoU and greedy rule."""
import inspect
import textwrap
import types
import numpy as np
import torch
from utils.Rotated_IoU.oriented_iou_loss import cal_iou, box2corners_th, oriented_box_intersection_2d


def tensor_nms(boxes, scores, threshold):
    order = torch.argsort(scores.view(-1), descending=True)
    boxes = boxes[order].float()
    n = len(boxes)
    if n > 2048:  # bounded memory fallback, same greedy algorithm
        ids = order.clone(); keep = []
        while len(boxes):
            keep.append(ids[0].item())
            if len(boxes) == 1: break
            iou = cal_iou(boxes[:1].repeat(len(boxes)-1,1)[None], boxes[1:][None])[0][0]
            mask = iou <= threshold
            boxes, ids = boxes[1:][mask], ids[1:][mask]
        return keep
    # One batched pairwise calculation instead of launching IoU per retained box.
    # Same cal_iou, same argsort, same <= threshold (including NaN semantics).
    alive = np.ones(n, dtype=bool)
    ij = torch.triu_indices(n,n,offset=1,device=boxes.device)
    # Conservative AABB broad phase: geometrically disjoint rotated boxes
    # have zero IoU. Padding retains boundary cases for the original kernel.
    half = boxes[:,2:4].abs()/2
    c, sn = boxes[:,4].cos().abs(), boxes[:,4].sin().abs()
    ext = torch.stack((c*half[:,0]+sn*half[:,1],sn*half[:,0]+c*half[:,1]),dim=1)
    delta = (boxes[ij[0],:2]-boxes[ij[1],:2]).abs()
    bound = ext[ij[0]]+ext[ij[1]]
    valid = torch.isfinite(boxes).all(1) & (boxes[:,2:4]>0).all(1)
    possible = (delta <= bound+1e-3).all(1) | ~valid[ij[0]] | ~valid[ij[1]]
    ij = ij[:,possible]
    corners = box2corners_th(boxes[None])[0]
    areas = boxes[:,2]*boxes[:,3]
    chunks = []
    for start in range(0,ij.shape[1],4096):
        q = ij[:,start:start+4096]
        inter, _ = oriented_box_intersection_2d(corners[q[0]][None], corners[q[1]][None])
        iou = inter[0] / (areas[q[0]] + areas[q[1]] - inter[0])
        chunks.append(iou <= threshold)
    matrix = np.ones((n,n),dtype=bool)
    if chunks:
        cpu_ij=ij.cpu().numpy()
        matrix[cpu_ij[0],cpu_ij[1]]=torch.cat(chunks).cpu().numpy()
    ids=order.cpu().numpy();keep=[]
    for i in range(n):
        if alive[i]:
            keep.append(int(ids[i]));alive[i+1:] &= matrix[i,i+1:]
    return keep


def install(head):
    original=head.get_nms_pred_boxes_for_single_sample
    source=textwrap.dedent(inspect.getsource(original))
    start=source.index('            pred_reg_xy_xlyl_th =')
    end=source.index('            pred_reg_bbox_with_conf = pred_reg_bbox_with_conf[indices]', start)
    source=source[:start]+'''            boxes_for_nms = pred_reg_bbox_with_conf[:, [1,2,4,5,7]]
            indices = tensor_nms(boxes_for_nms, pred_reg_bbox_with_conf[:,0], self.nms_thr)
'''+source[end:]
    ns=dict(original.__func__.__globals__);ns['tensor_nms']=tensor_nms
    exec(compile(source,__file__,'exec'),ns)
    head.get_nms_pred_boxes_for_single_sample=types.MethodType(ns[original.__name__],head)
    return original
