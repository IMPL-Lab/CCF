# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmcv.cnn import ConvModule, build_conv_layer
# from mmdet.models.utils import multi_apply
# from mmengine.model import BaseModule
# from mmengine.structures import InstanceData
from torch import Tensor, nn

# from mmdet3d.models.utils import (clip_sigmoid, draw_heatmap_gaussian,
#                                   gaussian_radius)
# from mmdet3d.registry import MODELS, TASK_UTILS
# from mmdet3d.core import Det3DDataSample, xywhr2xyxyr

#from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead

#@MODELS.register_module()
class MFSeparateHead(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(MFSeparateHead, self).__init__()
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]
            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)
            
            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv2d')
        
    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
        
    def forward(self, x_box, x_cls = None):
        ret_dict = dict()
        for head in self.heads:
            if head == 'heatmap':
                ret_dict[head] = self.__getattr__(head)(x_cls)
            else:
                ret_dict[head] = self.__getattr__(head)(x_box)

        return ret_dict