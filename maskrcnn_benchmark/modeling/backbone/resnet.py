# import torch.nn as nn
# #from .utils import load_state_dict_from_url
# import torch
# import collections
# from maskrcnn_benchmark.layers import Conv2d
# from maskrcnn_benchmark.layers import FrozenBatchNorm2d
#
# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
#
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
#
#
# # def conv3x3(in_channels, out_channels, stride=1):
# #     """3x3 convolution with padding"""
# #
# #     return Conv2d(in_channels, out_channels,kernel_size=3, stride=stride, bias=False)
# #
# #
# # def conv1x1(in_planes, out_planes, stride=1):
# #     """1x1 convolution"""
# #     return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn1 = FrozenBatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)
#         self.bn2 = FrozenBatchNorm2d(out_channels)
#         self.downsample = downsample
#         self.stride = stride
#
#         for l in [self.conv1, self.conv3,]:
#             nn.init.kaiming_uniform_(l.weight, a=1)
#
#
#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
#                  base_width=64, norm_layer=None, sact=True):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(out_channels * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#
#         self.conv1 = Conv2d(in_channels, width, kernel_size=1)
#         self.bn1 = FrozenBatchNorm2d(width)
#         self.conv2 = Conv2d(width, width, kernel_size=3)
#         self.bn2 = FrozenBatchNorm2d(width)
#         self.conv3 = Conv2d(width, out_channels * self.expansion, kernel_size=3)
#         self.bn3 = FrozenBatchNorm2d(out_channels * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#         self.sact = sact if sact else None
#
#         if sact:
#             self.conv4 = Conv2d(out_channels * self.expansion, 1, kernel_size=3) # used for calculating halting values
#
#     def forward(self, x):
#         identity = x
#
#         #print('[*unit*] input shape {}'.format(x.shape))
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         if not self.sact:
#             return out
#
#         return out, self.conv4(out) # output + halting values from each unit
#
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, cfg):
#         super(ResNet, self).__init__()
#
#         block = Bottleneck
#         layers = [3, 4, 6, 3]
#
#         zero_init_residual = False,
#         groups = 1
#         width_per_group = 64
#         replace_stride_with_dilation = None
#         norm_layer = None
#         sact = False
#
#
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#
#         # input size (224,224)
#         self.sact = sact
#         self.groups = groups
#         self.base_width = width_per_group
#
#         #torch.save(self.conv1.weight, 'kernel.pt')
#
#         for p in self.parameters():
#             p.requires_grad = False
#
#
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                    bias=False)
#
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.block1 = Block(1, layers[0], block, 64)
#
#         self.block2 = Block(2, layers[1], block, 128, stride=2)
#
#         self.block3 = Block(3, layers[2], block, 256, stride=2)
#
#         self.block4 = Block(4, layers[3], block, 512, stride=2)
#
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#
#
#     def forward(self, x):
#
#         outputs = []
#         ponder_cost = 0
#         units = []
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x, ponder, unit = self.block1(x)
#         outputs.append(x)
#         ponder_cost+= ponder
#         units.append(unit)
#
#         x, ponder, unit = self.block2(x)
#         outputs.append(x)
#         ponder_cost+= ponder
#         units.append(unit)
#
#         x, ponder, unit = self.block3(x)
#         outputs.append(x)
#         ponder_cost += ponder
#         units.append(unit)
#
#         x, ponder, unit = self.block4(x)
#         outputs.append(x)
#         ponder_cost += ponder
#         units.append(unit)
#
#         losses = {'ponder_cost': ponder_cost}
#         return outputs, losses, units

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple, defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DFConv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry


# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index -1]

            # module = _make_stage(
            #     transformation_module,
            #     in_channels,
            #     bottleneck_channels,
            #     out_channels,
            #     stage_spec.block_count,
            #     num_groups,
            #     cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            #     first_stride=int(stage_spec.index > 1) + 1,
            #     dcn_config={
            #         "stage_with_dcn": stage_with_dcn,
            #         "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
            #         "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
            #     }
            # )

            module = Block(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                }
            )

            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        ponder_cost = 0
        units = []
        for stage_name in self.stages:
            x, cost, used_unit= getattr(self, stage_name)(x)
            total = 0
            for i in range(len(x.shape)):
                total *= x.shape[i]
            ponder_cost += torch.sum(cost) / total
            units.append(used_unit)
            if self.return_features[stage_name]:
                outputs.append(x)
        losses = {'ponder_cost': ponder_cost.cuda()}

        return outputs, losses, units


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1,
        dcn_config={}
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1


            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config
            )

            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1,
    dcn_config={}
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)

class Block(nn.Module):

    def __init__(self, transformation_module,
                 in_channels,
                 bottleneck_channels,
                 out_channels,
                 block_count,
                 num_groups,
                 stride_in_1x1,
                 first_stride,
                 dilatition=1,
                 dcn_config={}):

        super(Block, self).__init__()

        stride = first_stride
        self.block_name = []
        self.eps = 1e-2

        # important parameters.
        self.ponder_weight = 0.005
        
        for i in range(block_count):

            name = "block" + str(i+1)

            module = transformation_module(
                    in_channels,
                    bottleneck_channels,
                    out_channels,
                    num_groups,
                    stride_in_1x1,
                    stride,
                    dilation=dilatition,
                    dcn_config=dcn_config
            )

            self.add_module(name, module)
            self.block_name.append(name)

            stride = 1
            in_channels = out_channels


    def forward(self, x):


        for index, name in enumerate(self.block_name):

            x, halting_proba = getattr(self, name)(x)

            # todo: check whether it stops at position i, j? if true, skip.
            # should pass a mask to self.unit
            if not index:
                shape = [x.shape[0], 1, x.shape[2], x.shape[3]]
                self.sact_params = defaultdict()
                self.sact_params['halting_cumsum'] = torch.zeros(shape)
                self.sact_params['element_finished'] = torch.zeros(shape)
                self.sact_params['remainder'] = torch.ones(shape)
                self.sact_params['ponder_cost'] = torch.ones(shape)
                self.sact_params['num_unit'] = torch.zeros(shape, dtype=torch.int32)

            # always halting at the last unit
            if index == len(self.block_name) - 1:
                halting_proba = torch.ones(shape)
            else:
                halting_proba = halting_proba.cpu()

            self.sact_params['halting_cumsum'] += halting_proba
            cur_elements_finished = (self.sact_params['halting_cumsum'] >= 1 - self.eps)
            #halting_proba = torch.where(cur_elements_finished, torch.zeros(x.shape), halting_proba)

            # find positions which have halted at the current unit
            just_finished = cur_elements_finished & (self.sact_params['element_finished'] == False)

            # see paper equation 9
            # for such positions, the halting distribution value is the remainder,
            # for others not finished yet, it is the halting probability
            cur_halting_dist = torch.where(just_finished, self.sact_params['remainder'], halting_proba)

            # see equation 11 in the paper
            # Since R (remainder) is a fixed value, we add it to ponder cost at the end.
            # If it has been finished before the current step, we add 0.
            # If it is not done yet, we add 1 to to N.
            self.sact_params['ponder_cost'] += torch.where(cur_elements_finished,
                                                           torch.where(just_finished, self.sact_params['remainder'], torch.zeros(shape)),
                                                           torch.ones(shape)) * self.ponder_weight

            self.sact_params['num_unit'] += (self.sact_params['element_finished'] == False).type(torch.int32) #need to consider shape!

            if index == 0:
                output = x * cur_halting_dist.cuda()
            else:
                output = output + x * cur_halting_dist.cuda()

            self.sact_params['remainder'] -= halting_proba

            self.sact_params['element_finished'] = cur_elements_finished

        return output, self.sact_params['ponder_cost'], self.sact_params['num_unit']


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config
    ):
        super(Bottleneck, self).__init__()

        self.downsample = None


        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(
                bottleneck_channels,
                bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn,
                kernel_size=3,
                stride=stride_3x3,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

        self.conv4 = Conv2d(
            out_channels, 1, kernel_size=3, padding=dilation, bias=False
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        halting_cost = self.conv4(out)

        return out, halting_cost


class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})