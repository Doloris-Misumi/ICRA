import torch
import torch.nn as nn
import numpy as np
# import nms
import utils.nms_custom as nms
import time

from utils.Rotated_IoU.oriented_iou_loss import cal_iou

class RdrSpcubeHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.roi = self.cfg.DATASET.RDR_SP_CUBE.ROI
        self.grid_size = self.cfg.DATASET.RDR_SP_CUBE.GRID_SIZE
        try:
            self.nms_thr = self.cfg.MODEL.HEAD.NMS_OVERLAP_THRESHOLD
        except:
            print('* Exception error (Head): nms threshold is set as 0.3')
            self.nms_thr = 0.3

        ### Anchors ###
        self.anchor_per_grid = []
        num_anchor_temp = 0

        self.list_anchor_classes = []
        self.list_anchor_matched_thr = []
        self.list_anchor_unmatched_thr = []
        self.list_anchor_targ_idx = []
        self.list_anchor_idx = [] # to slice tensor

        # for inference
        self.list_anc_idx_to_cls_id = [] # except bg
        self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
        # self.dict_cls_id_to_name = dict()
        # for k, v in self.dict_cls_name_to_id.items():
        #     if v != -1:
        #         self.dict_cls_id_to_name[v] = k

        num_prior_anchor_idx = 0
        for info_anchor in self.cfg.MODEL.ANCHOR_GENERATOR_CONFIG: # per class
            now_cls_name = info_anchor['class_name']
            self.list_anchor_classes.append(now_cls_name)
            self.list_anchor_matched_thr.append(info_anchor['matched_threshold'])
            self.list_anchor_unmatched_thr.append(info_anchor['unmatched_threshold'])
            
            self.anchor_sizes = info_anchor['anchor_sizes']
            self.anchor_rotations = info_anchor['anchor_rotations']
            self.anchor_bottoms = info_anchor['anchor_bottom_heights']

            self.list_anchor_targ_idx.append(num_prior_anchor_idx)
            num_now_anchor = int(len(self.anchor_sizes)*len(self.anchor_rotations)*len(self.anchor_bottoms))
            num_now_anchor_idx = num_prior_anchor_idx+num_now_anchor
            self.list_anchor_idx.append(num_now_anchor_idx)
            num_prior_anchor_idx = num_now_anchor_idx

            for anchor_size in self.anchor_sizes: # per size
                for anchor_rot in self.anchor_rotations: # per rot
                    for anchor_bottom in self.anchor_bottoms: # per bottom: for predicting zc
                        temp_anchor = [anchor_bottom] + anchor_size + [np.cos(anchor_rot), np.sin(anchor_rot)]
                        num_anchor_temp += 1
                        self.anchor_per_grid.append(temp_anchor) # [bot, xl, yl, zl, cos, sin]
                        self.list_anc_idx_to_cls_id.append(self.dict_cls_name_to_id[now_cls_name])
        self.num_anchor_per_grid = num_anchor_temp
        self.num_class = self.cfg.DATASET.CLASS_INFO.NUM_CLS
        self.num_box_code = len(self.cfg.MODEL.HEAD.BOX_CODE)
        ### Anchors ###

        ### 1x1 conv ###
        input_channels = self.cfg.MODEL.HEAD.DIM
        self.conv_cls = nn.Conv2d(
            input_channels, 1 + self.num_anchor_per_grid, # plus one for background
            kernel_size=1
        )
        self.conv_reg = nn.Conv2d(
            input_channels, self.num_anchor_per_grid*self.num_box_code,
            kernel_size=1
        )
        ### 1x1 conv ###

        ### Loss & Logging ###
        self.bg_weight = cfg.MODEL.HEAD.BG_WEIGHT
        self.categorical_focal_loss = FocalLoss()
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING
        ### Loss & Logging ###

        ### Anchor map ###
        self.register_buffer('anchor_map_for_batch', self.create_anchors()) # no batch
        ### Anchor map ###
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, dict_item):
        spatial_features_2d = dict_item['bev_feat']

        cls_pred = self.conv_cls(spatial_features_2d) # B x num_anchor+1 x Y x X
        reg_pred = self.conv_reg(spatial_features_2d) # B x num_anchor*num_box_code x Y x X
        
        dict_item['pred'] = {
            'cls': cls_pred,
            'reg': reg_pred,
        }

        return dict_item

    # V2
    def create_anchors(self):
        '''
        * e.g., 2 anchors (a,b) per class for 3 classes (A,B,C),
        *       anchor order -> (Aa Ab Ba Bb Ca Cc)
        '''
        dtype = torch.float32
        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        grid_size = self.grid_size
        n_x = int((x_max-x_min)/grid_size)
        n_y = int((y_max-y_min)/grid_size)
        
        # anchor location = center
        half_grid_size = grid_size/2.
        anchor_y = torch.arange(y_min, y_max, grid_size, dtype=dtype) - half_grid_size # minus: checked with visualization
        anchor_x = torch.arange(x_min, x_max, grid_size, dtype=dtype) - half_grid_size
        # print(anchor_y) # 200
        # print(anchor_x) # 248

        anchor_y = anchor_y.repeat_interleave(n_x)
        anchor_x = anchor_x.repeat(n_y)
        # print(anchor_y.shape) # 49600
        # print(anchor_x.shape) # 49600

        flattened_anchor_map = torch.stack((anchor_x, anchor_y), dim=1).unsqueeze(0).repeat(self.num_anchor_per_grid, 1, 1)
        # print(flattened_anchor_map.shape) # 2 x 49600 x 2 (xc, yc)
        flattened_anchor_attr = torch.tensor(self.anchor_per_grid, dtype=dtype)
        # print(flattened_anchor_attr.shape) # 2 x 6 (bottom, xl, yl, zl, cos, sin)
        flattened_anchor_attr = flattened_anchor_attr.unsqueeze(1).repeat(1, flattened_anchor_map.shape[1], 1)
        # print(flattened_anchor_attr.shape) # 2 x 49600 x 6

        anchor_map = torch.cat((flattened_anchor_map, flattened_anchor_attr), \
            dim=-1).view(self.num_anchor_per_grid, n_y, n_x, 8).contiguous().permute(0,3,1,2)
        anchor_map = anchor_map.reshape(-1, n_y, n_x).contiguous() # 16, 200, 248
        # print(anchor_map.shape) # 2 * (2+6) x 200 x 248

        anchor_map_for_batch = anchor_map.unsqueeze(0) # 1 x 16 x 200 x 248

        return anchor_map_for_batch

    def loss(self, dict_item):
        """Match static anchors jointly; one categorical anchor/GT per cell.

        Positive collisions choose highest IoU, then canonical GT order, then
        lowest anchor index. Background requires ALL compatible anchors to be
        below the unmatched threshold for EVERY GT. Empty frames are negative.
        The per-GT best-anchor fallback remains, but a cell collision can still
        leave a GT unmatched because the existing head has one class per cell.
        """
        cls_pred = dict_item['pred']['cls']
        raw_reg = dict_item['pred']['reg']
        dtype, device = cls_pred.dtype, cls_pred.device
        B, _, n_y, n_x = cls_pred.shape
        n_cells = n_y * n_x
        anchor_maps = self.anchor_map_for_batch.to(device).expand(B, -1, -1, -1)
        static_anchor_maps = anchor_maps.reshape(B, self.num_anchor_per_grid, self.num_box_code, n_y, n_x)
        reg_pred = (anchor_maps + raw_reg).reshape(B, self.num_anchor_per_grid, self.num_box_code, n_cells)
        anc_idx_targets = torch.zeros((B, n_cells), dtype=torch.long, device=device)
        pos_reg_pred, pos_reg_targ = [], []

        for batch_idx, list_objs in enumerate(dict_item['label']):
            # Canonical geometry order makes ties independent of annotation order.
            objects = sorted(list_objs, key=lambda obj: (obj[0], tuple(float(v) for v in obj[2])))
            best_iou = torch.full((n_cells,), -float('inf'), dtype=dtype, device=device)
            best_anchor = torch.full((n_cells,), -1, dtype=torch.long, device=device)
            best_gt = torch.full((n_cells,), -1, dtype=torch.long, device=device)
            ignore = torch.zeros(n_cells, dtype=torch.bool, device=device)
            anchors_per_class = []
            prior_anc_idx = 0
            for idx_anc_cls, _ in enumerate(self.list_anchor_classes):
                now_anc_idx = self.list_anchor_idx[idx_anc_cls]
                fixed = torch.cat((
                    static_anchor_maps[batch_idx,prior_anc_idx:now_anc_idx,:2],
                    static_anchor_maps[batch_idx,prior_anc_idx:now_anc_idx,3:5],
                    torch.atan2(
                        static_anchor_maps[batch_idx,prior_anc_idx:now_anc_idx,7:8],
                        static_anchor_maps[batch_idx,prior_anc_idx:now_anc_idx,6:7])), dim=1)
                anchors_per_class.append(fixed.permute(0, 2, 3, 1).reshape(1, -1, 5))
                prior_anc_idx = now_anc_idx

            target_boxes = []
            with torch.no_grad():
                for gt_idx, label in enumerate(objects):
                    cls_name, _, (xc, yc, zc, rz, xl, yl, zl), _ = label
                    class_idx = self.list_anchor_classes.index(cls_name)
                    anchors = anchors_per_class[class_idx]
                    gt = torch.tensor([xc, yc, xl, yl, rz], dtype=dtype, device=device)
                    iou, _, _, _ = cal_iou(gt.reshape(1, 1, 5).expand_as(anchors), anchors)
                    iou = iou.reshape(-1, n_cells)
                    if not torch.isfinite(iou).all():
                        raise FloatingPointError('non-finite static-anchor IoU')
                    ignore |= (iou >= self.list_anchor_unmatched_thr[class_idx]).any(dim=0)
                    positive = iou > self.list_anchor_matched_thr[class_idx]
                    if not positive.any():
                        positive.reshape(-1)[iou.reshape(-1).argmax()] = True
                    candidate_iou, local_anchor = iou.masked_fill(~positive, -float('inf')).max(dim=0)
                    take = candidate_iou > best_iou
                    best_iou[take] = candidate_iou[take]
                    best_anchor[take] = local_anchor[take] + self.list_anchor_targ_idx[class_idx]
                    best_gt[take] = gt_idx
                    target_boxes.append([xc, yc, zc, xl, yl, zl, np.cos(rz), np.sin(rz)])

            anc_idx_targets[batch_idx, ignore] = -1
            positive_cells = torch.where(best_gt >= 0)[0]
            if positive_cells.numel():
                chosen_anchors = best_anchor[positive_cells]
                anc_idx_targets[batch_idx, positive_cells] = chosen_anchors + 1
                pos_reg_pred.append(reg_pred[batch_idx, chosen_anchors, :, positive_cells])
                boxes = torch.tensor(target_boxes, dtype=dtype, device=device)
                pos_reg_targ.append(boxes[best_gt[positive_cells]])

        valid = anc_idx_targets >= 0
        targets = anc_idx_targets[valid]
        logits = cls_pred.permute(0, 2, 3, 1).reshape(B, n_cells, -1)[valid]
        if targets.numel():
            counts = torch.bincount(targets, minlength=1+self.num_anchor_per_grid).to(dtype)
            weights = torch.where(counts > 0, counts.clamp_min(1).reciprocal(), torch.zeros_like(counts))
            weights[0] *= self.bg_weight
            self.categorical_focal_loss.weight = weights.clamp_max(1)
            loss_cls = self.categorical_focal_loss(logits, targets)
        else:
            loss_cls = cls_pred.sum() * 0.0
        loss_reg = (torch.nn.functional.smooth_l1_loss(torch.cat(pos_reg_pred), torch.cat(pos_reg_targ))
                    if pos_reg_pred else raw_reg.sum() * 0.0)
        total_loss = loss_cls + loss_reg
        if self.is_logging:
            dict_item.setdefault('logging', {}).update({
                'total_loss': total_loss.detach().item(),
                'loss_reg': loss_reg.detach().item(),
                'focal_loss_cls': loss_cls.detach().item(),
                'target_positive_cells': int((anc_idx_targets > 0).sum().item()),
                'target_background_cells': int((anc_idx_targets == 0).sum().item()),
                'target_ignored_cells': int((anc_idx_targets < 0).sum().item()),
            })
        return total_loss

    def logging_dict_loss(self, loss, name_key):
        try:
            log_loss = loss.cpu().detach().item()
        except:
            log_loss = loss # for 0. loss

        return {name_key: log_loss}

    ### Validation & Inference ###
    def get_nms_pred_boxes_for_single_sample(self, dict_item, conf_thr, is_nms=True):
        '''
        * This function is common function of head for validataion & inference
        * For convenience, we assume batch_size = 1
        '''
        cls_pred = dict_item['pred']['cls'][0] # (1+n_anc) x n_y x n_x
        reg_pred = dict_item['pred']['reg'][0] # n_anc x n_y x n_x
        anchor_map = self.anchor_map_for_batch[0]
        reg_pred = anchor_map + reg_pred
        
        device = cls_pred.device

        # n_y x n_x -> (n_y*n_x)
        cls_pred = cls_pred.view(cls_pred.shape[0], -1)
        reg_pred = reg_pred.view(reg_pred.shape[0], -1)

        # bg X & more than conf_thr
        cls_pred = torch.softmax(cls_pred, dim=0)
        idx_deal = torch.where(
            (torch.argmax(cls_pred, dim=0)!=0) & (torch.max(cls_pred, dim=0)[0]>conf_thr))

        # for finding cls (not anc idx)
        tensor_anc_idx_per_cls = torch.tensor(self.list_anc_idx_to_cls_id, dtype=torch.long, device=device)
        
        len_deal_anc = len(idx_deal[0])
        # print('* debug # of dealing anchor boxes == n_anc:', len_deal_anc)
        if len_deal_anc > 0: # for only dealing grids (not bg & more than conf)              
            grid_anc_cls_logit = cls_pred[:, idx_deal[0]] # logit x n_pred / slice 1 for bg
            grid_anc_cls_idx = torch.argmax(grid_anc_cls_logit, dim=0) # minus 1 for bg after get conf
            grid_reg = reg_pred[:, idx_deal[0]]

            # to arange anc
            idx_range_anc = torch.arange(0, len_deal_anc, dtype=torch.long, device=device)
            anc_conf_score = grid_anc_cls_logit[grid_anc_cls_idx,idx_range_anc].unsqueeze(0) # 1 x n_deal
            grid_anc_cls_idx = grid_anc_cls_idx -1 # minus 1 for bg
            # print(anc_conf_score) # check if it is larger than conf_thr

            list_sliced_reg_bbox = []
            idx_slice_start = (grid_anc_cls_idx*self.num_box_code).long()
            
            for idx_reg_value in range(self.num_box_code):
                list_sliced_reg_bbox.append(grid_reg[idx_slice_start+idx_reg_value,idx_range_anc]) # n_deal
            sliced_reg_bbox = torch.stack(list_sliced_reg_bbox)
            # print(sliced_reg_bbox.shape) # 8 x n_deal
            temp_angle = torch.atan2(sliced_reg_bbox[-1,:], sliced_reg_bbox[-2,:]).unsqueeze(0)
            # print(temp_angle.shape)
            pred_reg_bbox_with_conf = torch.cat((anc_conf_score, sliced_reg_bbox[:-2,:], temp_angle), dim=0) # conf, x, y, z, xl, yl, zl, theta
            pred_reg_bbox_with_conf = pred_reg_bbox_with_conf.transpose(0,1)
            # print(pred_reg_bbox_with_conf.shape) # n_anc x 8 (score, x, y, z, xl, yl, zl, theta)

            cls_id_per_anc = tensor_anc_idx_per_cls[grid_anc_cls_idx]
            # print(cls_id_per_anc.shape) # n_anc
            num_of_bbox = len_deal_anc
            # print('* debug before nms: ', num_of_bbox)

            ### nms ###
            # try:
            if is_nms:
                pred_reg_xy_xlyl_th = torch.cat((pred_reg_bbox_with_conf[:,1:3], \
                pred_reg_bbox_with_conf[:,4:6], pred_reg_bbox_with_conf[:,7:8]), dim=1).cpu().detach().numpy()
            
                c_list = list(map(tuple, pred_reg_xy_xlyl_th[:,:2]))

                ### Assert error (Height > 0) ###
                dim_list = list(map(np.abs, pred_reg_xy_xlyl_th[:,2:4]))
                ### Assert error (Height > 0) ###
                
                dim_list = list(map(tuple, pred_reg_xy_xlyl_th[:,2:4]))
                angle_list = list(map(float, pred_reg_xy_xlyl_th[:,4]))

                list_tuple_for_nms = [[a, b, c] for (a, b, c) in zip(c_list, dim_list, angle_list)]
                conf_score = pred_reg_bbox_with_conf[:, 0:1].cpu().detach().numpy()

                indices = nms.rboxes(list_tuple_for_nms, conf_score, nms_threshold=self.nms_thr) # nms_usage
                pred_reg_bbox_with_conf = pred_reg_bbox_with_conf[indices]
                cls_id_per_anc = cls_id_per_anc[indices]

                num_of_bbox = len(indices) # after nms
            # except:
            #     print('* Exception error (head.py): nms error, probably assert height > 0')

            pred_reg_bbox_with_conf = pred_reg_bbox_with_conf.cpu().detach().numpy().tolist()
            cls_id_per_anc = cls_id_per_anc.cpu().detach().numpy().tolist()
            # print('* debug after nms: ', num_of_bbox)
        else:
            # empty prediction
            pred_reg_bbox_with_conf = None
            cls_id_per_anc = None
            num_of_bbox = 0 # 0

        # pp: post-processing
        dict_item['pp_bbox'] = pred_reg_bbox_with_conf # score, x, y, z, xl, yl, zl, theta
        dict_item['pp_cls'] = cls_id_per_anc
        dict_item['pp_desc'] = dict_item['meta'][0]['desc']
        dict_item['pp_num_bbox'] = num_of_bbox
        # print(dict_item['label'][0])
        # print(dict_item['pp_bbox'])

        return dict_item
    ### Validation & Inference ###

class FocalLoss(nn.Module):
    def __init__(self, weight=None, 
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input_tensor, target_tensor):
        log_prob = nn.functional.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)

        return nn.functional.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )
