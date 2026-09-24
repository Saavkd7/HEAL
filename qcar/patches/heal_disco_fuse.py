"""Plugin: restore opencood.models.fuse_modules.disco_fuse, which this HEAL
checkout is missing.

WHY
---
opencood/models/fuse_modules/fusion_in_one.py:DiscoFusion does
    from opencood.models.fuse_modules.disco_fuse import PixelWeightLayer
but disco_fuse.py is not in the repository, so fusion_method: disconet cannot
even be constructed (ModuleNotFoundError). HEAL's own released DiscoNet
checkpoint was trained with it: the authors' code snapshot saved next to that
checkpoint (checkpoints/heal_reference/opv2v_camera/
HeterBaseline_opv2v_camera_disco_2023_08_08_16_50_01/scripts/models/
fuse_modules/disco_fuse.py) contains PixelWeightLayer, copied here verbatim.
Its parameter names and shapes match the 23 fusion_net.pixel_weight_layer.*
tensors of that checkpoint exactly (strict load verified by qcar/zoo.py).

Registered under the original module name via sys.modules, so opencood/ is
not edited. Importing is the whole activation.
"""
import sys

import torch.nn as nn
import torch.nn.functional as F

MODULE_NAME = "opencood.models.fuse_modules.disco_fuse"


class PixelWeightLayer(nn.Module):
    def __init__(self, channel):
        super(PixelWeightLayer, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1


if MODULE_NAME not in sys.modules:
    sys.modules[MODULE_NAME] = sys.modules[__name__]
    print("[heal_disco_fuse] registered %s (PixelWeightLayer)" % MODULE_NAME)
