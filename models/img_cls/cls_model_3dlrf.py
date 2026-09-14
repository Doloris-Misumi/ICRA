"""Original 3D-LRF Stage 1 architecture and state-dict interface.

No projection is added to the pretrained model. WCBR's 64->512 adapter is
trained in the detector backbone. Only device handling and batch-1 squeeze
are made explicit relative to the reference implementation.
"""
import torch.nn as nn


class ImageClsBackbone3DLRF(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=4)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=4)
        self.bn3 = nn.BatchNorm2d(64)
        self.adjust = nn.Linear(220, 256)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 7)

    def forward(self, data):
        image = data['cam_front_img'].to(self.conv1.weight.device, non_blocking=True)
        x1 = self.bn1(self.conv1(image))
        x2 = self.bn2(self.conv2(x1))
        x3 = self.bn3(self.conv3(x2))
        x4 = self.adjust(x3.reshape(image.shape[0], 64, -1))
        x5 = self.gap(x4)
        data.update(img_cls_output=self.fc(x5.squeeze(-1)), img_cls_gap=x5,
                    img_cls_feat=x4, x1=x1, x2=x2, x3=x3)
        return data
