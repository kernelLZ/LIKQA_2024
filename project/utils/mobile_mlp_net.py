import torch
import torch.nn as nn
import timm


class MobileNetMerged(nn.Module):
    def __init__(self, block_size=4):
        super(MobileNetMerged, self).__init__()

        self.backbone = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=False)

        self.conv2d_up = nn.Linear(1000, 3100)
        self.conv2d_dw = nn.Linear(3100, 512)

        self.head = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        x = self.conv2d_up(x)
        x = self.relu(x)
        x = self.conv2d_dw(x)
        x = self.relu(x)
        concat_pool = torch.cat([x], dim=1)
        output = self.head(concat_pool)

        return output
