"""Opt-in inference backbone. Same checkpoint keys; old implementation untouched.
Exact Euclidean KNN via cKDTree. Boundary ties follow cKDTree; may differ from sklearn.
Within-neighborhood ordering and linear algebra order may differ numerically.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from .rl_3df import RL3DFBackbone_Branching, torch, spconv


def exact_knn_legacy_boundary(radar, lidar, k):
    if not 1 <= k <= len(radar):
        raise ValueError('invalid k')
    # Exact Euclidean neighbors, eps=0. Ties follow cKDTree, not sklearn.
    _, indices = cKDTree(radar).query(lidar, k=k, eps=0.0, workers=16)
    return np.asarray(indices).reshape(len(lidar), k)


class RL3DFBackbone_BranchingFastV2(RL3DFBackbone_Branching):
    def forward(self, dict_item):
        sparse_featuresR, sparse_indicesR = dict_item['sp_features'], dict_item['sp_indices']
        sparse_featuresL, sparse_indicesL = dict_item['sp_features_l'], dict_item['sp_indices_l']

        img_condition_token = self._image_condition(dict_item)
        prompt_condition_token = dict_item.get('prompt_weather_token', None)
        if self.condition_mode == 'sensor_only' and prompt_condition_token is not None:
            raise ValueError('sensor_only condition cannot include a prompt token')
        if prompt_condition_token is None:
            prompt_condition_token = img_condition_token
        condition_token = 0.5 * (img_condition_token + prompt_condition_token)

        radar_available = sparse_featuresR.shape[0] > 0
        lidar_available = sparse_featuresL.shape[0] > 0
        if not radar_available and not lidar_available:
            raise RuntimeError('both radar and LiDAR have zero in-ROI voxels')

        xR = None
        if radar_available:
            input_sp_tensorR = spconv.SparseConvTensor(
                features=sparse_featuresR,
                indices=sparse_indicesR.int(),
                spatial_shape=self.spatial_shape,
                batch_size=dict_item['batch_size']
            )
            xR = self.input_convR(input_sp_tensorR)

        xL = None
        if lidar_available:
            input_sp_tensorL = spconv.SparseConvTensor(
                features=sparse_featuresL,
                indices=sparse_indicesL.int(),
                spatial_shape=self.spatial_shape,
                batch_size=dict_item['batch_size']
            )
            xL = self.input_convL(input_sp_tensorL)

        empty_token = condition_token.new_zeros(
            (dict_item['batch_size'], self.sensor_token_proj.in_features)
        )
        radar_token = (
            self._pool_sparse_tokens(xR.features, xR.indices, dict_item['batch_size'])
            if xR is not None else empty_token
        )
        lidar_token = (
            self._pool_sparse_tokens(xL.features, xL.indices, dict_item['batch_size'])
            if xL is not None else empty_token
        )
        radar_token = self.sensor_token_proj(radar_token)
        lidar_token = self.sensor_token_proj(lidar_token)

        if self.condition_mode == 'camera_only':
            radar_token = torch.zeros_like(radar_token)
            lidar_token = torch.zeros_like(lidar_token)

        token_concat = torch.cat((condition_token, radar_token, lidar_token), dim=-1)
        condition_update = self.condition_mlp(token_concat)
        condition_token = self.condition_mlp_norm(condition_token + condition_update)
        dict_item['condition_token'] = condition_token
        if self.enable_weather_aux:
            dict_item['weather_logits_aux'] = self.weather_aux_head(condition_token)

        branch_weights = self._route(condition_token)
        if self.force_branch != 'none':
            forced = torch.zeros_like(branch_weights)
            branch_to_idx = {'lidar': 0, 'radar': 1, 'fusion': 2}
            forced[:, branch_to_idx[self.force_branch]] = 1.0
            branch_weights = forced
        dict_item['branch_weights'] = branch_weights

        xL_pure = xL
        xL_fused = xL 

        list_bev_L = []
        list_bev_R = []
        list_bev_F = []
        
        for idx_layer in range(self.num_layer):
            # 1. Process Radar Stream (xR).  A completely missing modality is
            # represented as a zero BEV below instead of a fake sparse point.
            if xR is not None:
                xR = getattr(self, f'spconv{idx_layer}R')(xR)
                xR = xR.replace_feature(getattr(self, f'bn{idx_layer}R')(xR.features))
                xR = xR.replace_feature(self.relu(xR.features))
                xR = getattr(self, f'subm{idx_layer}aR')(xR)
                xR = xR.replace_feature(getattr(self, f'bn{idx_layer}aR')(xR.features))
                xR = xR.replace_feature(self.relu(xR.features))
                xR = getattr(self, f'subm{idx_layer}bR')(xR)
                xR = xR.replace_feature(getattr(self, f'bn{idx_layer}bR')(xR.features))
                xR = xR.replace_feature(self.relu(xR.features))
            
            # 2. Process Pure Lidar Stream (xL_pure)
            if xL_pure is not None:
                xL_pure = getattr(self, f'spconv{idx_layer}L')(xL_pure)
                xL_pure = xL_pure.replace_feature(getattr(self, f'bn{idx_layer}L')(xL_pure.features))
                xL_pure = xL_pure.replace_feature(self.relu(xL_pure.features))
                xL_pure = getattr(self, f'subm{idx_layer}aL')(xL_pure)
                xL_pure = xL_pure.replace_feature(getattr(self, f'bn{idx_layer}aL')(xL_pure.features))
                xL_pure = xL_pure.replace_feature(self.relu(xL_pure.features))
                xL_pure = getattr(self, f'subm{idx_layer}bL')(xL_pure)
                xL_pure = xL_pure.replace_feature(getattr(self, f'bn{idx_layer}bL')(xL_pure.features))
                xL_pure = xL_pure.replace_feature(self.relu(xL_pure.features))
            
            # 3. Process Fusion Stream (xL_fused)
            # Use same weights as Lidar stream but on different tensor
            if xL_pure is None:
                xL_fused = None
            elif idx_layer == 0:
                 # Clone first layer output to branch off
                 xL_fused = spconv.SparseConvTensor(
                    features=xL_pure.features.clone(),
                    indices=xL_pure.indices,
                    spatial_shape=xL_pure.spatial_shape,
                    batch_size=xL_pure.batch_size
                )
            else:
                xLf = xL_fused
                xLf = getattr(self, f'spconv{idx_layer}L')(xLf)
                xLf = xLf.replace_feature(getattr(self, f'bn{idx_layer}L')(xLf.features))
                xLf = xLf.replace_feature(self.relu(xLf.features))
                xLf = getattr(self, f'subm{idx_layer}aL')(xLf)
                xLf = xLf.replace_feature(getattr(self, f'bn{idx_layer}aL')(xLf.features))
                xLf = xLf.replace_feature(self.relu(xLf.features))
                xLf = getattr(self, f'subm{idx_layer}bL')(xLf)
                xLf = xLf.replace_feature(getattr(self, f'bn{idx_layer}bL')(xLf.features))
                xLf = xLf.replace_feature(self.relu(xLf.features))
                xL_fused = xLf

            # 4. Apply Fusion Logic to xL_fused
            if xL_fused is not None and xR is not None:
                img_layer_feat = getattr(self, f'img_layer{idx_layer}')(condition_token)
                if len(img_layer_feat.shape) == 1:
                    img_layer_feat = img_layer_feat.unsqueeze(0)

                batch = dict_item['batch_size']
                xR_feat, xR_indices = xR.features, xR.indices
                xLf_feat, xLf_indices = xL_fused.features, xL_fused.indices
                new_xLf_feat_list = []
                for batch_idx in range(batch):
                    radar_indices = np.array(xR_indices[xR_indices[:, 0] == batch_idx][:, 1:].cpu())
                    lidar_indices = np.array(xLf_indices[xLf_indices[:, 0] == batch_idx][:, 1:].cpu())

                    if len(radar_indices) == 0 or len(lidar_indices) == 0:
                         new_xLf_feat_list.append(xLf_feat[xLf_indices[:, 0] == batch_idx])
                         continue

                    knn_indices = exact_knn_legacy_boundary(
                        radar_indices, lidar_indices,
                        min(len(radar_indices), int(64 / (2**idx_layer))))
                    knn_indices = torch.from_numpy(knn_indices).to(device=sparse_indicesR.device)

                    lidar_feat_b = xLf_feat[xLf_indices[:, 0] == batch_idx]
                    radar_feat_b = xR_feat[xR_indices[:, 0] == batch_idx]
                    query = lidar_feat_b
                    key = radar_feat_b[knn_indices]

                    attn = torch.bmm(query.unsqueeze(1), key.permute(0,2,1))
                    attn = torch.softmax(attn, dim=-1)
                    attn_value = getattr(self, f'value_layer{idx_layer}')(torch.bmm(attn, key))

                    gate = getattr(self, f'gate_layer{idx_layer}')
                    channels = key.shape[-1]
                    gate_mean = torch.nn.functional.linear(key.mean(dim=1), gate.weight[:, :channels], gate.bias)
                    gate_mean = gate_mean + torch.nn.functional.linear(img_layer_feat[batch_idx], gate.weight[:, channels:])
                    gate_gap_feat = gate_mean.unsqueeze(-1)
                    attn_value_gate = torch.einsum('abc, abc -> abc', attn_value, torch.sigmoid(gate_gap_feat.permute(0, 2, 1)))

                    new_xLf_feat_list.append(attn_value_gate.squeeze() + query)

                new_xLf_feat = torch.cat(new_xLf_feat_list, 0)
                xL_fused = spconv.SparseConvTensor(
                    features=new_xLf_feat,
                    indices=xLf_indices,
                    spatial_shape=xL_fused.spatial_shape,
                    batch_size=xL_fused.batch_size
                )

            # 5. BEV Conversion
            def to_bev(tensor, idx_layer, suffix):
                if self.is_z_embed:
                    bev_dense = getattr(self, f'chzcat{idx_layer}{suffix}')(tensor.dense())
                    bev_dense = getattr(self, f'convtrans2d{idx_layer}{suffix}')(bev_dense)
                else:
                    bev_sp = getattr(self, f'toBEV{idx_layer}{suffix}')(tensor)
                    bev_sp = bev_sp.replace_feature(getattr(self, f'bnBEV{idx_layer}{suffix}')(bev_sp.features))
                    bev_sp = bev_sp.replace_feature(self.relu(bev_sp.features))
                    bev_dense = getattr(self, f'convtrans2d{idx_layer}{suffix}')(bev_sp.dense().squeeze(2))
                
                bev_dense = getattr(self, f'bnt{idx_layer}{suffix}')(bev_dense)
                bev_dense = self.relu(bev_dense)
                return bev_dense

            bev_L = to_bev(xL_pure, idx_layer, 'L') if xL_pure is not None else None
            bev_R = to_bev(xR, idx_layer, 'R') if xR is not None else None
            bev_F = to_bev(xL_fused, idx_layer, 'L') if xL_fused is not None else None
            template = bev_R if bev_R is not None else bev_L
            if bev_L is None:
                bev_L = torch.zeros_like(template)
            if bev_R is None:
                bev_R = torch.zeros_like(template)
            if bev_F is None:
                bev_F = torch.zeros_like(template)
            list_bev_L.append(bev_L)
            list_bev_R.append(bev_R)
            list_bev_F.append(bev_F)

        bev_feat_L = torch.cat(list_bev_L, dim=1)
        bev_feat_R = torch.cat(list_bev_R, dim=1)
        bev_feat_F = torch.cat(list_bev_F, dim=1)
        
        # 6. Construct Final Output based on Soft Weighting
        # bev_feat_L: Pure Lidar Features
        # bev_feat_R: Radar Features
        # bev_feat_F: Fused Features
        # Target Output Shape: Concatenation of two feature maps (e.g., Radar-like and Lidar-like channels)
        # To maintain compatibility with the head which expects cat(R, L) shape, we can construct weighted features.
        
        # Structure of final_bev was:
        # Lidar Only: [Zeros, L_pure]
        # Radar Only: [R, Zeros]
        # Fusion:     [R, F]
        
        # Soft Weighting Logic:
        # We will create a weighted sum for the "Radar-side" channel and "Lidar-side" channel.
        # Let weights be w_L, w_R, w_F
        # Radar-side channel: w_R * R + w_F * R = (w_R + w_F) * R  (Since Lidar Only has Zeros here)
        # Lidar-side channel: w_L * L_pure + w_F * F (Since Radar Only has Zeros here)
        
        w_L = branch_weights[:, 0].view(-1, 1, 1, 1) # (B, 1, 1, 1)
        w_R = branch_weights[:, 1].view(-1, 1, 1, 1)
        w_F = branch_weights[:, 2].view(-1, 1, 1, 1)
        
        # 1. Radar-side Component (First half of channels)
        # In Hard Selection:
        # if Lidar Only (w_L=1): Zeros
        # if Radar Only (w_R=1): R
        # if Fusion (w_F=1):     R
        # Soft equivalent: (w_R + w_F) * bev_feat_R
        final_bev_R_side = (w_R + w_F) * bev_feat_R
        
        # 2. Lidar-side Component (Second half of channels)
        # In Hard Selection:
        # if Lidar Only (w_L=1): L_pure
        # if Radar Only (w_R=1): Zeros
        # if Fusion (w_F=1):     F
        # Soft equivalent: w_L * bev_feat_L + w_F * bev_feat_F
        final_bev_L_side = w_L * bev_feat_L + w_F * bev_feat_F
        
        final_bev = torch.cat((final_bev_R_side, final_bev_L_side), dim=1)
                
        dict_item['bev_feat'] = final_bev
        return dict_item
