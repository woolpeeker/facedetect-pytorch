

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from model.module import SeparableConv2d

class DetectModule(nn.Module):
    def __init__(self, in_channel):
        super(DetectModule, self).__init__()
        self.log_module = nn.Sequential(SeparableConv2d(in_channel, 64, multiplier=3),
                                        SeparableConv2d(64, 2, multiplier=3, activation_fn=None))
        self.loc_module = nn.Sequential(SeparableConv2d(in_channel, 64, multiplier=3),
                                        SeparableConv2d(64, 4, multiplier=3, activation_fn=None))
    def forward(self, feature):
        log = self.log_module(feature)
        loc = self.loc_module(feature)
        return log, loc

class AlphaNet(nn.Module):
    def __init__(self, num_anchors):
        super(AlphaNet, self).__init__()
        self.strides = [8,8,8]
        #feature extraction module
        self.feature_module = nn.Sequential(
            SeparableConv2d(3, 64, multiplier=32, stride=2,),
            SeparableConv2d(64, 128, multiplier=3),
            SeparableConv2d(128, 128, multiplier=3, stride=2),
            SeparableConv2d(128, 128, multiplier=3),
            SeparableConv2d(128, 128, multiplier=3, stride=2)
        )
        for i in range(len(self.strides)):
            self.add_module('detect_module_%d'%i, DetectModule(in_channel=128))

    def forward(self, inputs):
        # the output tensor [Batch, Channel, H, W]
        cls_list = []
        log_list = []
        loc_list = []
        feature = self.feature_module(inputs)
        for i in range(len(self.strides)):
            detect_module = self._modules['detect_module_%d'%i]
            log, loc= detect_module(feature)
            cls = F.softmax(log, dim=1)[:, 1, ...]
            log_list.append(log)
            cls_list.append(cls)
            loc_list.append(loc)
        return {
            'log': log_list,
            'cls': cls_list,
            'loc': loc_list
        }