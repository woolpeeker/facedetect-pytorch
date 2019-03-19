
import torch as th
import torch.nn as nn
import utils as ut

import numpy as np

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, multiplier=1, stride=1, activation_fn=nn.functional.relu):
        super(SeparableConv2d, self).__init__()
        if stride in [1,2]:
            padding = 1
        else:
            raise Exception('stride must be 1 or 2')

        self.in_channels = in_channels
        self.multiplier = multiplier
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.activation_fn = activation_fn

        self.DSconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * multiplier, 3, stride=stride, padding=padding,
                      dilation=1, groups=in_channels, bias=True),
            nn.Conv2d(in_channels * multiplier, out_channels, 1, stride=1, padding=0,
                      dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, inputs):
        out = self.DSconv(inputs)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out


class MultiAnchor(nn.Module):
    def __init__(self, input_shape, strides, anchor_sizes):
        """Square anchor
        :param input_shape: input_image shapes (h,w)
        :param strides: anchor strides, a scalar
        :param anchor_sizes: a list of scalar
        """
        super(MultiAnchor, self).__init__()
        if len(strides) != len(anchor_sizes):
            raise Exception('input_shape must have the same number as anchor_sizes.')
        self.input_shape = input_shape
        self.strides = strides
        self.anchor_sizes = anchor_sizes
        self.num_anchor = len(anchor_sizes)

        # generate the anchor boxes
        for i in range(len(strides)):
            stride = strides[i]
            anchor_size = anchor_sizes[i]
            if input_shape[0]%stride != 0 or input_shape[1]%stride != 0:
                raise Exception('input_shape has to be divisible by stride, '
                                'while the input shape is {}, stride is {}' % {input_shape, stride})
            cy, cx = th.meshgrid(th.arange(0, input_shape[0], stride),
                                 th.arange(0, input_shape[1], stride))
            cy, cx = cy.float(), cx.float()
            h, w = th.full_like(cx, anchor_size), th.full_like(cx, anchor_size)
            yxhw = th.stack([cy,cx,h,w], dim=-1)
            yxyx = ut.yxhw_to_yxyx(yxhw)
            self.register_buffer('yxhw_%d'%i, yxhw)
            self.register_buffer('yxyx_%d'%i, yxyx)

    def encode(self, boxes_list):
        '''convert ground truth boxes to loc_offset and labels
        :param boxes: list of tensors. [[num_boxes,4],...], the list length if the batchSize
        :return: ious_list: the length of the list is same with the anchor nums,
                            each tensor is [batchsize, 1, h, w]
                 offset_list: each tensor is [batchsize, 4, h, w]
        '''
        ious_list = []
        offsets_list = []
        for anchor_idx in range(self.num_anchor):
            yxyx = self._buffers['yxyx_%d'%anchor_idx]
            yxhw = self._buffers['yxhw_%d'%anchor_idx]
            anchor_size = self.anchor_sizes[anchor_idx]
            ious = []
            offsets = []
            for boxes in boxes_list:
                flat_yxyx = th.reshape(yxyx, [-1, 4])
                iou = ut.jaccard(flat_yxyx, boxes)
                iou, idx = th.max(iou, -1)
                iou = th.reshape(iou, [yxyx.shape[0], yxyx.shape[1]])
                ious.append(iou)

                boxes_yxhw = ut.yxyx_to_yxhw(boxes)
                flat_yxhw = th.reshape(yxhw, [-1,4])
                expand_yxhw = flat_yxhw.unsqueeze(1).expand(flat_yxhw.size(0), boxes_yxhw.size(0), 4)
                offset_array = boxes_yxhw - expand_yxhw
                offset = [offset_array[i,idx[i]] for i in range(idx.size(0))]
                offset = th.stack(offset) / anchor_size
                offset = th.reshape(offset, yxyx.shape)
                offsets.append(offset)
            ious = th.stack(ious, dim=0)
            offsets = th.stack(offsets, dim=0).permute([0, 3, 1, 2])
            ious_list.append(ious)
            offsets_list.append(offsets)
        return ious_list, offsets_list

    def decode(self, labels_list, offsets_list):
        '''
        :param labels_list: len(list) is the anchor num
        :param offsets_list:
        :return: boxes_list: len(boxes_list) is the batch size
        '''
        batch_size = labels_list[0].size(0)
        boxes_list = [[] for _ in range(batch_size)]
        for i in range(self.num_anchor):
            labels = labels_list[i].permute([0, 2, 3, 1])
            offsets = offsets_list[i].permute([0, 2, 3, 1])
            yxhw = self._buffers['yxhw_%d'%i]
            boxes = yxhw + offsets * self.anchor_sizes[i]
            mask = labels > 0
            for i in range(batch_size):
                boxes_decode = boxes[i][mask[i]] # equal to boolean_mask in tf
                boxes_decode = ut.yxhw_to_yxyx(boxes_decode)
                boxes_list[i].append(boxes_decode)

        for i in range(batch_size):
            boxes_list[i] = th.cat(boxes_list[i], dim=0)
        return boxes_list