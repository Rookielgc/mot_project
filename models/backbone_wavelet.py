# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import pywt


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 使用 PyWavelets 进行二维小波变换，返回低频和高频分量
        coeffs = pywt.dwt2(x.cpu().numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs  # 低频和高频分量
        cA = torch.tensor(cA).to(x.device)
        cH = torch.tensor(cH).to(x.device)
        return cA, cH  # 返回低频和水平高频部分


class WaveletCBAM(nn.Module):
    def __init__(self, channels, reduction=11, kernel_size=5, wavelet='db2'):
        super(WaveletCBAM, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)

        self.spatial_attention = SpatialAttention(kernel_size)

        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # # 空间注意力模块
        # self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 小波分解
        cA, cH = self.wavelet_transform(x)

        # 通道注意力
        avg_pool = torch.mean(cA, dim=(2, 3), keepdim=True)
        max_pool = torch.max(cH, dim=(2, 3), keepdim=True)[0]
        channel_att = self.channel_fc(avg_pool + max_pool).permute(0, 2, 3, 1)
        channel_att = self.sigmoid(channel_att)

        # 空间注意力
        spatial_input = torch.cat([cA, cH], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # 注意力增强
        out = x * channel_att * spatial_att
        return out




class FrozenBatchNorm2d(torch.nn.Module):#是Joiner类的一部分，自定义的批量归一化层，冻结了批量统计和仿射参数，这意味着在训练过程中
    #这些参数不会更新，这通常用于迁移学习或者当你希望使用预训练参数时。
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))#注册名为weight的缓冲区，其值为形状为n的全0张量
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典加载模型状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'#构造num_batch_tracked的完整键名
        if num_batches_tracked_key in state_dict:#如果状态字典中包括num_batches_tracked键
            del state_dict[num_batches_tracked_key]#删除状态字典中的

        #传递状态字典和其他参数给父类方法
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning，将重塑操作移到前面，以便于融合器友好处理
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5#定义一个很小的正数，用于数值稳定性
        scale = w * (rv + eps).rsqrt()#计算缩放因子
        bias = b - rm * scale#计算bias偏移
        return x * scale + bias#返回应用了批量归一化的输入张量


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)#通过调用模型的body属性来处理tensor_list.tensor中的张量。xs是一个包含特征图的字典，其中键是层的名字，值是对应的特征图
        out: Dict[str, NestedTensor] = {}#初始化一个名为out的字典，用于存储输出的NestedTensor对象，字典的键是字符串，值是NestedTensor类型
        for name, x in xs.items():#遍历xs字典中的每一项，name是层的名称，x是对应的特征图
            m = tensor_list.mask#获取输入的tensor_list掩码
            assert m is not None#确保已经提供了掩码
            #使用插值函数F.interpolate将掩码m调整到与特征图x相同的空间尺寸，m[None]是增加一个新的批次维度，
            # .float()将掩码转换为浮点类型，size=x.shape[-2:]指定新的大小，.to(torch.bool)将结果转换为布尔类型的张量，[0]取出批次维度
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #创建一个新的NestedTensor对象，包含特征图x和对应的掩码mask，然后将这个对象存储到输出字典out中，键是层的名称
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, wavelet="db2"):
        norm_layer = FrozenBatchNorm2d  #设置批量归一化层为冻结状态
        #动态获取指定名称的ResNet模型，并根据是否使用空洞卷积对模型进行配置
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        #调用父类初始化方法，传入配置好的ResNet模型，是否训练back_bone，是否返回中间层特征
        super().__init__(backbone, train_backbone, return_interm_layers)
        #如果使用空洞卷积，则调整步长以适应空洞卷积
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
            # 假设 self.num_channels 包含了每个卷积层输出的通道数

        self.cbam_blocks = nn.ModuleList([WaveletCBAM(channels, wavelet=wavelet) for channels in self.num_channels])

class Joiner(nn.Sequential):#将骨干网络backbone和位置嵌入position embedding结合
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides#从backbone中获取步长并将其保存为Joiner类的一个属性
        self.num_channels = backbone.num_channels#从backbone中获取步长并保存为一个属性

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)#通过调用序列中的第一个模块（其backbone）来处理输入的tensor_list，得到输出xs
        out: List[NestedTensor] = []#初始化一个空列表out用于存储输出的NestedTensor对象
        pos = []#初始化一个空列表pos,用于存储位置编码
        for name, x in sorted(xs.items()):#遍历字典中xs中的项（已排序），将对应的NestedTensor添加到out
            out.append(x)

        # position encoding
        for x in out:#遍历out中的每个NestedTensor
            pos.append(self[1](x).to(x.tensors.dtype))#调用序列中的第二个模块（position_embedding）来处理x，将结果中的数据类型转换为
            #与x中的张量相同的类型，并将其添加到位置编码列表pos中

        return out, pos#返回输出的NestedTensor列表out和位置编码列表pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    wavelet = 'db2'
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, wavelet=wavelet)
    model = Joiner(backbone, position_embedding)
    return model# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import pywt


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 使用 PyWavelets 进行二维小波变换，返回低频和高频分量
        coeffs = pywt.dwt2(x.cpu().numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs  # 低频和高频分量
        cA = torch.tensor(cA).to(x.device)
        cH = torch.tensor(cH).to(x.device)
        return cA, cH  # 返回低频和水平高频部分


class WaveletCBAM(nn.Module):
    def __init__(self, channels, reduction=11, kernel_size=5, wavelet='db2'):
        super(WaveletCBAM, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)

        self.spatial_attention = SpatialAttention(kernel_size)

        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # # 空间注意力模块
        # self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 小波分解
        cA, cH = self.wavelet_transform(x)

        # 通道注意力
        avg_pool = torch.mean(cA, dim=(2, 3), keepdim=True)
        max_pool = torch.max(cH, dim=(2, 3), keepdim=True)[0]
        channel_att = self.channel_fc(avg_pool + max_pool).permute(0, 2, 3, 1)
        channel_att = self.sigmoid(channel_att)

        # 空间注意力
        spatial_input = torch.cat([cA, cH], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # 注意力增强
        out = x * channel_att * spatial_att
        return out




class FrozenBatchNorm2d(torch.nn.Module):#是Joiner类的一部分，自定义的批量归一化层，冻结了批量统计和仿射参数，这意味着在训练过程中
    #这些参数不会更新，这通常用于迁移学习或者当你希望使用预训练参数时。
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))#注册名为weight的缓冲区，其值为形状为n的全0张量
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典加载模型状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'#构造num_batch_tracked的完整键名
        if num_batches_tracked_key in state_dict:#如果状态字典中包括num_batches_tracked键
            del state_dict[num_batches_tracked_key]#删除状态字典中的

        #传递状态字典和其他参数给父类方法
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning，将重塑操作移到前面，以便于融合器友好处理
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5#定义一个很小的正数，用于数值稳定性
        scale = w * (rv + eps).rsqrt()#计算缩放因子
        bias = b - rm * scale#计算bias偏移
        return x * scale + bias#返回应用了批量归一化的输入张量


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)#通过调用模型的body属性来处理tensor_list.tensor中的张量。xs是一个包含特征图的字典，其中键是层的名字，值是对应的特征图
        out: Dict[str, NestedTensor] = {}#初始化一个名为out的字典，用于存储输出的NestedTensor对象，字典的键是字符串，值是NestedTensor类型
        for name, x in xs.items():#遍历xs字典中的每一项，name是层的名称，x是对应的特征图
            m = tensor_list.mask#获取输入的tensor_list掩码
            assert m is not None#确保已经提供了掩码
            #使用插值函数F.interpolate将掩码m调整到与特征图x相同的空间尺寸，m[None]是增加一个新的批次维度，
            # .float()将掩码转换为浮点类型，size=x.shape[-2:]指定新的大小，.to(torch.bool)将结果转换为布尔类型的张量，[0]取出批次维度
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #创建一个新的NestedTensor对象，包含特征图x和对应的掩码mask，然后将这个对象存储到输出字典out中，键是层的名称
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, wavelet="db2"):
        norm_layer = FrozenBatchNorm2d  #设置批量归一化层为冻结状态
        #动态获取指定名称的ResNet模型，并根据是否使用空洞卷积对模型进行配置
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        #调用父类初始化方法，传入配置好的ResNet模型，是否训练back_bone，是否返回中间层特征
        super().__init__(backbone, train_backbone, return_interm_layers)
        #如果使用空洞卷积，则调整步长以适应空洞卷积
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
            # 假设 self.num_channels 包含了每个卷积层输出的通道数

        self.cbam_blocks = nn.ModuleList([WaveletCBAM(channels, wavelet=wavelet) for channels in self.num_channels])

class Joiner(nn.Sequential):#将骨干网络backbone和位置嵌入position embedding结合
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides#从backbone中获取步长并将其保存为Joiner类的一个属性
        self.num_channels = backbone.num_channels#从backbone中获取步长并保存为一个属性

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)#通过调用序列中的第一个模块（其backbone）来处理输入的tensor_list，得到输出xs
        out: List[NestedTensor] = []#初始化一个空列表out用于存储输出的NestedTensor对象
        pos = []#初始化一个空列表pos,用于存储位置编码
        for name, x in sorted(xs.items()):#遍历字典中xs中的项（已排序），将对应的NestedTensor添加到out
            out.append(x)

        # position encoding
        for x in out:#遍历out中的每个NestedTensor
            pos.append(self[1](x).to(x.tensors.dtype))#调用序列中的第二个模块（position_embedding）来处理x，将结果中的数据类型转换为
            #与x中的张量相同的类型，并将其添加到位置编码列表pos中

        return out, pos#返回输出的NestedTensor列表out和位置编码列表pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    wavelet = 'db2'
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, wavelet=wavelet)
    model = Joiner(backbone, position_embedding)
    return model# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import pywt


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 使用 PyWavelets 进行二维小波变换，返回低频和高频分量
        coeffs = pywt.dwt2(x.cpu().numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs  # 低频和高频分量
        cA = torch.tensor(cA).to(x.device)
        cH = torch.tensor(cH).to(x.device)
        return cA, cH  # 返回低频和水平高频部分


class WaveletCBAM(nn.Module):
    def __init__(self, channels, reduction=11, kernel_size=5, wavelet='db2'):
        super(WaveletCBAM, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)

        self.spatial_attention = SpatialAttention(kernel_size)

        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # # 空间注意力模块
        # self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 小波分解
        cA, cH = self.wavelet_transform(x)

        # 通道注意力
        avg_pool = torch.mean(cA, dim=(2, 3), keepdim=True)
        max_pool = torch.max(cH, dim=(2, 3), keepdim=True)[0]
        channel_att = self.channel_fc(avg_pool + max_pool).permute(0, 2, 3, 1)
        channel_att = self.sigmoid(channel_att)

        # 空间注意力
        spatial_input = torch.cat([cA, cH], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # 注意力增强
        out = x * channel_att * spatial_att
        return out




class FrozenBatchNorm2d(torch.nn.Module):#是Joiner类的一部分，自定义的批量归一化层，冻结了批量统计和仿射参数，这意味着在训练过程中
    #这些参数不会更新，这通常用于迁移学习或者当你希望使用预训练参数时。
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))#注册名为weight的缓冲区，其值为形状为n的全0张量
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典加载模型状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'#构造num_batch_tracked的完整键名
        if num_batches_tracked_key in state_dict:#如果状态字典中包括num_batches_tracked键
            del state_dict[num_batches_tracked_key]#删除状态字典中的

        #传递状态字典和其他参数给父类方法
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning，将重塑操作移到前面，以便于融合器友好处理
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5#定义一个很小的正数，用于数值稳定性
        scale = w * (rv + eps).rsqrt()#计算缩放因子
        bias = b - rm * scale#计算bias偏移
        return x * scale + bias#返回应用了批量归一化的输入张量


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)#通过调用模型的body属性来处理tensor_list.tensor中的张量。xs是一个包含特征图的字典，其中键是层的名字，值是对应的特征图
        out: Dict[str, NestedTensor] = {}#初始化一个名为out的字典，用于存储输出的NestedTensor对象，字典的键是字符串，值是NestedTensor类型
        for name, x in xs.items():#遍历xs字典中的每一项，name是层的名称，x是对应的特征图
            m = tensor_list.mask#获取输入的tensor_list掩码
            assert m is not None#确保已经提供了掩码
            #使用插值函数F.interpolate将掩码m调整到与特征图x相同的空间尺寸，m[None]是增加一个新的批次维度，
            # .float()将掩码转换为浮点类型，size=x.shape[-2:]指定新的大小，.to(torch.bool)将结果转换为布尔类型的张量，[0]取出批次维度
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #创建一个新的NestedTensor对象，包含特征图x和对应的掩码mask，然后将这个对象存储到输出字典out中，键是层的名称
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, wavelet="db2"):
        norm_layer = FrozenBatchNorm2d  #设置批量归一化层为冻结状态
        #动态获取指定名称的ResNet模型，并根据是否使用空洞卷积对模型进行配置
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        #调用父类初始化方法，传入配置好的ResNet模型，是否训练back_bone，是否返回中间层特征
        super().__init__(backbone, train_backbone, return_interm_layers)
        #如果使用空洞卷积，则调整步长以适应空洞卷积
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
            # 假设 self.num_channels 包含了每个卷积层输出的通道数

        self.cbam_blocks = nn.ModuleList([WaveletCBAM(channels, wavelet=wavelet) for channels in self.num_channels])

class Joiner(nn.Sequential):#将骨干网络backbone和位置嵌入position embedding结合
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides#从backbone中获取步长并将其保存为Joiner类的一个属性
        self.num_channels = backbone.num_channels#从backbone中获取步长并保存为一个属性

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)#通过调用序列中的第一个模块（其backbone）来处理输入的tensor_list，得到输出xs
        out: List[NestedTensor] = []#初始化一个空列表out用于存储输出的NestedTensor对象
        pos = []#初始化一个空列表pos,用于存储位置编码
        for name, x in sorted(xs.items()):#遍历字典中xs中的项（已排序），将对应的NestedTensor添加到out
            out.append(x)

        # position encoding
        for x in out:#遍历out中的每个NestedTensor
            pos.append(self[1](x).to(x.tensors.dtype))#调用序列中的第二个模块（position_embedding）来处理x，将结果中的数据类型转换为
            #与x中的张量相同的类型，并将其添加到位置编码列表pos中

        return out, pos#返回输出的NestedTensor列表out和位置编码列表pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    wavelet = 'db2'
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, wavelet=wavelet)
    model = Joiner(backbone, position_embedding)
    return model# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import pywt


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 使用 PyWavelets 进行二维小波变换，返回低频和高频分量
        coeffs = pywt.dwt2(x.cpu().numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs  # 低频和高频分量
        cA = torch.tensor(cA).to(x.device)
        cH = torch.tensor(cH).to(x.device)
        return cA, cH  # 返回低频和水平高频部分


class WaveletCBAM(nn.Module):
    def __init__(self, channels, reduction=11, kernel_size=5, wavelet='db2'):
        super(WaveletCBAM, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)

        self.spatial_attention = SpatialAttention(kernel_size)

        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # # 空间注意力模块
        # self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 小波分解
        cA, cH = self.wavelet_transform(x)

        # 通道注意力
        avg_pool = torch.mean(cA, dim=(2, 3), keepdim=True)
        max_pool = torch.max(cH, dim=(2, 3), keepdim=True)[0]
        channel_att = self.channel_fc(avg_pool + max_pool).permute(0, 2, 3, 1)
        channel_att = self.sigmoid(channel_att)

        # 空间注意力
        spatial_input = torch.cat([cA, cH], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # 注意力增强
        out = x * channel_att * spatial_att
        return out




class FrozenBatchNorm2d(torch.nn.Module):#是Joiner类的一部分，自定义的批量归一化层，冻结了批量统计和仿射参数，这意味着在训练过程中
    #这些参数不会更新，这通常用于迁移学习或者当你希望使用预训练参数时。
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))#注册名为weight的缓冲区，其值为形状为n的全0张量
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典加载模型状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'#构造num_batch_tracked的完整键名
        if num_batches_tracked_key in state_dict:#如果状态字典中包括num_batches_tracked键
            del state_dict[num_batches_tracked_key]#删除状态字典中的

        #传递状态字典和其他参数给父类方法
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning，将重塑操作移到前面，以便于融合器友好处理
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5#定义一个很小的正数，用于数值稳定性
        scale = w * (rv + eps).rsqrt()#计算缩放因子
        bias = b - rm * scale#计算bias偏移
        return x * scale + bias#返回应用了批量归一化的输入张量


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)#通过调用模型的body属性来处理tensor_list.tensor中的张量。xs是一个包含特征图的字典，其中键是层的名字，值是对应的特征图
        out: Dict[str, NestedTensor] = {}#初始化一个名为out的字典，用于存储输出的NestedTensor对象，字典的键是字符串，值是NestedTensor类型
        for name, x in xs.items():#遍历xs字典中的每一项，name是层的名称，x是对应的特征图
            m = tensor_list.mask#获取输入的tensor_list掩码
            assert m is not None#确保已经提供了掩码
            #使用插值函数F.interpolate将掩码m调整到与特征图x相同的空间尺寸，m[None]是增加一个新的批次维度，
            # .float()将掩码转换为浮点类型，size=x.shape[-2:]指定新的大小，.to(torch.bool)将结果转换为布尔类型的张量，[0]取出批次维度
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #创建一个新的NestedTensor对象，包含特征图x和对应的掩码mask，然后将这个对象存储到输出字典out中，键是层的名称
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, wavelet="db2"):
        norm_layer = FrozenBatchNorm2d  #设置批量归一化层为冻结状态
        #动态获取指定名称的ResNet模型，并根据是否使用空洞卷积对模型进行配置
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        #调用父类初始化方法，传入配置好的ResNet模型，是否训练back_bone，是否返回中间层特征
        super().__init__(backbone, train_backbone, return_interm_layers)
        #如果使用空洞卷积，则调整步长以适应空洞卷积
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
            # 假设 self.num_channels 包含了每个卷积层输出的通道数

        self.cbam_blocks = nn.ModuleList([WaveletCBAM(channels, wavelet=wavelet) for channels in self.num_channels])

class Joiner(nn.Sequential):#将骨干网络backbone和位置嵌入position embedding结合
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides#从backbone中获取步长并将其保存为Joiner类的一个属性
        self.num_channels = backbone.num_channels#从backbone中获取步长并保存为一个属性

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)#通过调用序列中的第一个模块（其backbone）来处理输入的tensor_list，得到输出xs
        out: List[NestedTensor] = []#初始化一个空列表out用于存储输出的NestedTensor对象
        pos = []#初始化一个空列表pos,用于存储位置编码
        for name, x in sorted(xs.items()):#遍历字典中xs中的项（已排序），将对应的NestedTensor添加到out
            out.append(x)

        # position encoding
        for x in out:#遍历out中的每个NestedTensor
            pos.append(self[1](x).to(x.tensors.dtype))#调用序列中的第二个模块（position_embedding）来处理x，将结果中的数据类型转换为
            #与x中的张量相同的类型，并将其添加到位置编码列表pos中

        return out, pos#返回输出的NestedTensor列表out和位置编码列表pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    wavelet = 'db2'
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, wavelet=wavelet)
    model = Joiner(backbone, position_embedding)
    return model# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import pywt


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 使用 PyWavelets 进行二维小波变换，返回低频和高频分量
        coeffs = pywt.dwt2(x.cpu().numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs  # 低频和高频分量
        cA = torch.tensor(cA).to(x.device)
        cH = torch.tensor(cH).to(x.device)
        return cA, cH  # 返回低频和水平高频部分


class WaveletCBAM(nn.Module):
    def __init__(self, channels, reduction=11, kernel_size=5, wavelet='db2'):
        super(WaveletCBAM, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)

        self.spatial_attention = SpatialAttention(kernel_size)

        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # # 空间注意力模块
        # self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 小波分解
        cA, cH = self.wavelet_transform(x)

        # 通道注意力
        avg_pool = torch.mean(cA, dim=(2, 3), keepdim=True)
        max_pool = torch.max(cH, dim=(2, 3), keepdim=True)[0]
        channel_att = self.channel_fc(avg_pool + max_pool).permute(0, 2, 3, 1)
        channel_att = self.sigmoid(channel_att)

        # 空间注意力
        spatial_input = torch.cat([cA, cH], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # 注意力增强
        out = x * channel_att * spatial_att
        return out




class FrozenBatchNorm2d(torch.nn.Module):#是Joiner类的一部分，自定义的批量归一化层，冻结了批量统计和仿射参数，这意味着在训练过程中
    #这些参数不会更新，这通常用于迁移学习或者当你希望使用预训练参数时。
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))#注册名为weight的缓冲区，其值为形状为n的全0张量
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典加载模型状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'#构造num_batch_tracked的完整键名
        if num_batches_tracked_key in state_dict:#如果状态字典中包括num_batches_tracked键
            del state_dict[num_batches_tracked_key]#删除状态字典中的

        #传递状态字典和其他参数给父类方法
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning，将重塑操作移到前面，以便于融合器友好处理
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5#定义一个很小的正数，用于数值稳定性
        scale = w * (rv + eps).rsqrt()#计算缩放因子
        bias = b - rm * scale#计算bias偏移
        return x * scale + bias#返回应用了批量归一化的输入张量


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)#通过调用模型的body属性来处理tensor_list.tensor中的张量。xs是一个包含特征图的字典，其中键是层的名字，值是对应的特征图
        out: Dict[str, NestedTensor] = {}#初始化一个名为out的字典，用于存储输出的NestedTensor对象，字典的键是字符串，值是NestedTensor类型
        for name, x in xs.items():#遍历xs字典中的每一项，name是层的名称，x是对应的特征图
            m = tensor_list.mask#获取输入的tensor_list掩码
            assert m is not None#确保已经提供了掩码
            #使用插值函数F.interpolate将掩码m调整到与特征图x相同的空间尺寸，m[None]是增加一个新的批次维度，
            # .float()将掩码转换为浮点类型，size=x.shape[-2:]指定新的大小，.to(torch.bool)将结果转换为布尔类型的张量，[0]取出批次维度
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #创建一个新的NestedTensor对象，包含特征图x和对应的掩码mask，然后将这个对象存储到输出字典out中，键是层的名称
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, wavelet="db2"):
        norm_layer = FrozenBatchNorm2d  #设置批量归一化层为冻结状态
        #动态获取指定名称的ResNet模型，并根据是否使用空洞卷积对模型进行配置
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        #调用父类初始化方法，传入配置好的ResNet模型，是否训练back_bone，是否返回中间层特征
        super().__init__(backbone, train_backbone, return_interm_layers)
        #如果使用空洞卷积，则调整步长以适应空洞卷积
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
            # 假设 self.num_channels 包含了每个卷积层输出的通道数

        self.cbam_blocks = nn.ModuleList([WaveletCBAM(channels, wavelet=wavelet) for channels in self.num_channels])

class Joiner(nn.Sequential):#将骨干网络backbone和位置嵌入position embedding结合
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides#从backbone中获取步长并将其保存为Joiner类的一个属性
        self.num_channels = backbone.num_channels#从backbone中获取步长并保存为一个属性

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)#通过调用序列中的第一个模块（其backbone）来处理输入的tensor_list，得到输出xs
        out: List[NestedTensor] = []#初始化一个空列表out用于存储输出的NestedTensor对象
        pos = []#初始化一个空列表pos,用于存储位置编码
        for name, x in sorted(xs.items()):#遍历字典中xs中的项（已排序），将对应的NestedTensor添加到out
            out.append(x)

        # position encoding
        for x in out:#遍历out中的每个NestedTensor
            pos.append(self[1](x).to(x.tensors.dtype))#调用序列中的第二个模块（position_embedding）来处理x，将结果中的数据类型转换为
            #与x中的张量相同的类型，并将其添加到位置编码列表pos中

        return out, pos#返回输出的NestedTensor列表out和位置编码列表pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    wavelet = 'db2'
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, wavelet=wavelet)
    model = Joiner(backbone, position_embedding)
    return model# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import pywt


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 使用 PyWavelets 进行二维小波变换，返回低频和高频分量
        coeffs = pywt.dwt2(x.cpu().numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs  # 低频和高频分量
        cA = torch.tensor(cA).to(x.device)
        cH = torch.tensor(cH).to(x.device)
        return cA, cH  # 返回低频和水平高频部分


class WaveletCBAM(nn.Module):
    def __init__(self, channels, reduction=11, kernel_size=5, wavelet='db2'):
        super(WaveletCBAM, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)

        self.spatial_attention = SpatialAttention(kernel_size)

        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # # 空间注意力模块
        # self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 小波分解
        cA, cH = self.wavelet_transform(x)

        # 通道注意力
        avg_pool = torch.mean(cA, dim=(2, 3), keepdim=True)
        max_pool = torch.max(cH, dim=(2, 3), keepdim=True)[0]
        channel_att = self.channel_fc(avg_pool + max_pool).permute(0, 2, 3, 1)
        channel_att = self.sigmoid(channel_att)

        # 空间注意力
        spatial_input = torch.cat([cA, cH], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # 注意力增强
        out = x * channel_att * spatial_att
        return out




class FrozenBatchNorm2d(torch.nn.Module):#是Joiner类的一部分，自定义的批量归一化层，冻结了批量统计和仿射参数，这意味着在训练过程中
    #这些参数不会更新，这通常用于迁移学习或者当你希望使用预训练参数时。
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))#注册名为weight的缓冲区，其值为形状为n的全0张量
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典加载模型状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'#构造num_batch_tracked的完整键名
        if num_batches_tracked_key in state_dict:#如果状态字典中包括num_batches_tracked键
            del state_dict[num_batches_tracked_key]#删除状态字典中的

        #传递状态字典和其他参数给父类方法
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning，将重塑操作移到前面，以便于融合器友好处理
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5#定义一个很小的正数，用于数值稳定性
        scale = w * (rv + eps).rsqrt()#计算缩放因子
        bias = b - rm * scale#计算bias偏移
        return x * scale + bias#返回应用了批量归一化的输入张量


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)#通过调用模型的body属性来处理tensor_list.tensor中的张量。xs是一个包含特征图的字典，其中键是层的名字，值是对应的特征图
        out: Dict[str, NestedTensor] = {}#初始化一个名为out的字典，用于存储输出的NestedTensor对象，字典的键是字符串，值是NestedTensor类型
        for name, x in xs.items():#遍历xs字典中的每一项，name是层的名称，x是对应的特征图
            m = tensor_list.mask#获取输入的tensor_list掩码
            assert m is not None#确保已经提供了掩码
            #使用插值函数F.interpolate将掩码m调整到与特征图x相同的空间尺寸，m[None]是增加一个新的批次维度，
            # .float()将掩码转换为浮点类型，size=x.shape[-2:]指定新的大小，.to(torch.bool)将结果转换为布尔类型的张量，[0]取出批次维度
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #创建一个新的NestedTensor对象，包含特征图x和对应的掩码mask，然后将这个对象存储到输出字典out中，键是层的名称
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, wavelet="db2"):
        norm_layer = FrozenBatchNorm2d  #设置批量归一化层为冻结状态
        #动态获取指定名称的ResNet模型，并根据是否使用空洞卷积对模型进行配置
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        #调用父类初始化方法，传入配置好的ResNet模型，是否训练back_bone，是否返回中间层特征
        super().__init__(backbone, train_backbone, return_interm_layers)
        #如果使用空洞卷积，则调整步长以适应空洞卷积
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
            # 假设 self.num_channels 包含了每个卷积层输出的通道数

        self.cbam_blocks = nn.ModuleList([WaveletCBAM(channels, wavelet=wavelet) for channels in self.num_channels])

class Joiner(nn.Sequential):#将骨干网络backbone和位置嵌入position embedding结合
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides#从backbone中获取步长并将其保存为Joiner类的一个属性
        self.num_channels = backbone.num_channels#从backbone中获取步长并保存为一个属性

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)#通过调用序列中的第一个模块（其backbone）来处理输入的tensor_list，得到输出xs
        out: List[NestedTensor] = []#初始化一个空列表out用于存储输出的NestedTensor对象
        pos = []#初始化一个空列表pos,用于存储位置编码
        for name, x in sorted(xs.items()):#遍历字典中xs中的项（已排序），将对应的NestedTensor添加到out
            out.append(x)

        # position encoding
        for x in out:#遍历out中的每个NestedTensor
            pos.append(self[1](x).to(x.tensors.dtype))#调用序列中的第二个模块（position_embedding）来处理x，将结果中的数据类型转换为
            #与x中的张量相同的类型，并将其添加到位置编码列表pos中

        return out, pos#返回输出的NestedTensor列表out和位置编码列表pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    wavelet = 'db2'
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, wavelet=wavelet)
    model = Joiner(backbone, position_embedding)
    return model# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import pywt


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 使用 PyWavelets 进行二维小波变换，返回低频和高频分量
        coeffs = pywt.dwt2(x.cpu().numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs  # 低频和高频分量
        cA = torch.tensor(cA).to(x.device)
        cH = torch.tensor(cH).to(x.device)
        return cA, cH  # 返回低频和水平高频部分


class WaveletCBAM(nn.Module):
    def __init__(self, channels, reduction=11, kernel_size=5, wavelet='db2'):
        super(WaveletCBAM, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)

        self.spatial_attention = SpatialAttention(kernel_size)

        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # # 空间注意力模块
        # self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 小波分解
        cA, cH = self.wavelet_transform(x)

        # 通道注意力
        avg_pool = torch.mean(cA, dim=(2, 3), keepdim=True)
        max_pool = torch.max(cH, dim=(2, 3), keepdim=True)[0]
        channel_att = self.channel_fc(avg_pool + max_pool).permute(0, 2, 3, 1)
        channel_att = self.sigmoid(channel_att)

        # 空间注意力
        spatial_input = torch.cat([cA, cH], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # 注意力增强
        out = x * channel_att * spatial_att
        return out




class FrozenBatchNorm2d(torch.nn.Module):#是Joiner类的一部分，自定义的批量归一化层，冻结了批量统计和仿射参数，这意味着在训练过程中
    #这些参数不会更新，这通常用于迁移学习或者当你希望使用预训练参数时。
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))#注册名为weight的缓冲区，其值为形状为n的全0张量
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典加载模型状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'#构造num_batch_tracked的完整键名
        if num_batches_tracked_key in state_dict:#如果状态字典中包括num_batches_tracked键
            del state_dict[num_batches_tracked_key]#删除状态字典中的

        #传递状态字典和其他参数给父类方法
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning，将重塑操作移到前面，以便于融合器友好处理
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5#定义一个很小的正数，用于数值稳定性
        scale = w * (rv + eps).rsqrt()#计算缩放因子
        bias = b - rm * scale#计算bias偏移
        return x * scale + bias#返回应用了批量归一化的输入张量


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)#通过调用模型的body属性来处理tensor_list.tensor中的张量。xs是一个包含特征图的字典，其中键是层的名字，值是对应的特征图
        out: Dict[str, NestedTensor] = {}#初始化一个名为out的字典，用于存储输出的NestedTensor对象，字典的键是字符串，值是NestedTensor类型
        for name, x in xs.items():#遍历xs字典中的每一项，name是层的名称，x是对应的特征图
            m = tensor_list.mask#获取输入的tensor_list掩码
            assert m is not None#确保已经提供了掩码
            #使用插值函数F.interpolate将掩码m调整到与特征图x相同的空间尺寸，m[None]是增加一个新的批次维度，
            # .float()将掩码转换为浮点类型，size=x.shape[-2:]指定新的大小，.to(torch.bool)将结果转换为布尔类型的张量，[0]取出批次维度
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #创建一个新的NestedTensor对象，包含特征图x和对应的掩码mask，然后将这个对象存储到输出字典out中，键是层的名称
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, wavelet="db2"):
        norm_layer = FrozenBatchNorm2d  #设置批量归一化层为冻结状态
        #动态获取指定名称的ResNet模型，并根据是否使用空洞卷积对模型进行配置
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        #调用父类初始化方法，传入配置好的ResNet模型，是否训练back_bone，是否返回中间层特征
        super().__init__(backbone, train_backbone, return_interm_layers)
        #如果使用空洞卷积，则调整步长以适应空洞卷积
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
            # 假设 self.num_channels 包含了每个卷积层输出的通道数

        self.cbam_blocks = nn.ModuleList([WaveletCBAM(channels, wavelet=wavelet) for channels in self.num_channels])

class Joiner(nn.Sequential):#将骨干网络backbone和位置嵌入position embedding结合
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides#从backbone中获取步长并将其保存为Joiner类的一个属性
        self.num_channels = backbone.num_channels#从backbone中获取步长并保存为一个属性

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)#通过调用序列中的第一个模块（其backbone）来处理输入的tensor_list，得到输出xs
        out: List[NestedTensor] = []#初始化一个空列表out用于存储输出的NestedTensor对象
        pos = []#初始化一个空列表pos,用于存储位置编码
        for name, x in sorted(xs.items()):#遍历字典中xs中的项（已排序），将对应的NestedTensor添加到out
            out.append(x)

        # position encoding
        for x in out:#遍历out中的每个NestedTensor
            pos.append(self[1](x).to(x.tensors.dtype))#调用序列中的第二个模块（position_embedding）来处理x，将结果中的数据类型转换为
            #与x中的张量相同的类型，并将其添加到位置编码列表pos中

        return out, pos#返回输出的NestedTensor列表out和位置编码列表pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    wavelet = 'db2'
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, wavelet=wavelet)
    model = Joiner(backbone, position_embedding)
    return model# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import pywt


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 使用 PyWavelets 进行二维小波变换，返回低频和高频分量
        coeffs = pywt.dwt2(x.cpu().numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs  # 低频和高频分量
        cA = torch.tensor(cA).to(x.device)
        cH = torch.tensor(cH).to(x.device)
        return cA, cH  # 返回低频和水平高频部分


class WaveletCBAM(nn.Module):
    def __init__(self, channels, reduction=11, kernel_size=5, wavelet='db2'):
        super(WaveletCBAM, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)

        self.spatial_attention = SpatialAttention(kernel_size)

        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # # 空间注意力模块
        # self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 小波分解
        cA, cH = self.wavelet_transform(x)

        # 通道注意力
        avg_pool = torch.mean(cA, dim=(2, 3), keepdim=True)
        max_pool = torch.max(cH, dim=(2, 3), keepdim=True)[0]
        channel_att = self.channel_fc(avg_pool + max_pool).permute(0, 2, 3, 1)
        channel_att = self.sigmoid(channel_att)

        # 空间注意力
        spatial_input = torch.cat([cA, cH], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # 注意力增强
        out = x * channel_att * spatial_att
        return out




class FrozenBatchNorm2d(torch.nn.Module):#是Joiner类的一部分，自定义的批量归一化层，冻结了批量统计和仿射参数，这意味着在训练过程中
    #这些参数不会更新，这通常用于迁移学习或者当你希望使用预训练参数时。
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))#注册名为weight的缓冲区，其值为形状为n的全0张量
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典加载模型状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'#构造num_batch_tracked的完整键名
        if num_batches_tracked_key in state_dict:#如果状态字典中包括num_batches_tracked键
            del state_dict[num_batches_tracked_key]#删除状态字典中的

        #传递状态字典和其他参数给父类方法
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning，将重塑操作移到前面，以便于融合器友好处理
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5#定义一个很小的正数，用于数值稳定性
        scale = w * (rv + eps).rsqrt()#计算缩放因子
        bias = b - rm * scale#计算bias偏移
        return x * scale + bias#返回应用了批量归一化的输入张量


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)#通过调用模型的body属性来处理tensor_list.tensor中的张量。xs是一个包含特征图的字典，其中键是层的名字，值是对应的特征图
        out: Dict[str, NestedTensor] = {}#初始化一个名为out的字典，用于存储输出的NestedTensor对象，字典的键是字符串，值是NestedTensor类型
        for name, x in xs.items():#遍历xs字典中的每一项，name是层的名称，x是对应的特征图
            m = tensor_list.mask#获取输入的tensor_list掩码
            assert m is not None#确保已经提供了掩码
            #使用插值函数F.interpolate将掩码m调整到与特征图x相同的空间尺寸，m[None]是增加一个新的批次维度，
            # .float()将掩码转换为浮点类型，size=x.shape[-2:]指定新的大小，.to(torch.bool)将结果转换为布尔类型的张量，[0]取出批次维度
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #创建一个新的NestedTensor对象，包含特征图x和对应的掩码mask，然后将这个对象存储到输出字典out中，键是层的名称
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, wavelet="db2"):
        norm_layer = FrozenBatchNorm2d  #设置批量归一化层为冻结状态
        #动态获取指定名称的ResNet模型，并根据是否使用空洞卷积对模型进行配置
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        #调用父类初始化方法，传入配置好的ResNet模型，是否训练back_bone，是否返回中间层特征
        super().__init__(backbone, train_backbone, return_interm_layers)
        #如果使用空洞卷积，则调整步长以适应空洞卷积
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
            # 假设 self.num_channels 包含了每个卷积层输出的通道数

        self.cbam_blocks = nn.ModuleList([WaveletCBAM(channels, wavelet=wavelet) for channels in self.num_channels])

class Joiner(nn.Sequential):#将骨干网络backbone和位置嵌入position embedding结合
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides#从backbone中获取步长并将其保存为Joiner类的一个属性
        self.num_channels = backbone.num_channels#从backbone中获取步长并保存为一个属性

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)#通过调用序列中的第一个模块（其backbone）来处理输入的tensor_list，得到输出xs
        out: List[NestedTensor] = []#初始化一个空列表out用于存储输出的NestedTensor对象
        pos = []#初始化一个空列表pos,用于存储位置编码
        for name, x in sorted(xs.items()):#遍历字典中xs中的项（已排序），将对应的NestedTensor添加到out
            out.append(x)

        # position encoding
        for x in out:#遍历out中的每个NestedTensor
            pos.append(self[1](x).to(x.tensors.dtype))#调用序列中的第二个模块（position_embedding）来处理x，将结果中的数据类型转换为
            #与x中的张量相同的类型，并将其添加到位置编码列表pos中

        return out, pos#返回输出的NestedTensor列表out和位置编码列表pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    wavelet = 'db2'
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, wavelet=wavelet)
    model = Joiner(backbone, position_embedding)
    return model# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import pywt


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 使用 PyWavelets 进行二维小波变换，返回低频和高频分量
        coeffs = pywt.dwt2(x.cpu().numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs  # 低频和高频分量
        cA = torch.tensor(cA).to(x.device)
        cH = torch.tensor(cH).to(x.device)
        return cA, cH  # 返回低频和水平高频部分


class WaveletCBAM(nn.Module):
    def __init__(self, channels, reduction=11, kernel_size=5, wavelet='db2'):
        super(WaveletCBAM, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)

        self.spatial_attention = SpatialAttention(kernel_size)

        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # # 空间注意力模块
        # self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 小波分解
        cA, cH = self.wavelet_transform(x)

        # 通道注意力
        avg_pool = torch.mean(cA, dim=(2, 3), keepdim=True)
        max_pool = torch.max(cH, dim=(2, 3), keepdim=True)[0]
        channel_att = self.channel_fc(avg_pool + max_pool).permute(0, 2, 3, 1)
        channel_att = self.sigmoid(channel_att)

        # 空间注意力
        spatial_input = torch.cat([cA, cH], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # 注意力增强
        out = x * channel_att * spatial_att
        return out




class FrozenBatchNorm2d(torch.nn.Module):#是Joiner类的一部分，自定义的批量归一化层，冻结了批量统计和仿射参数，这意味着在训练过程中
    #这些参数不会更新，这通常用于迁移学习或者当你希望使用预训练参数时。
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))#注册名为weight的缓冲区，其值为形状为n的全0张量
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典加载模型状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'#构造num_batch_tracked的完整键名
        if num_batches_tracked_key in state_dict:#如果状态字典中包括num_batches_tracked键
            del state_dict[num_batches_tracked_key]#删除状态字典中的

        #传递状态字典和其他参数给父类方法
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning，将重塑操作移到前面，以便于融合器友好处理
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5#定义一个很小的正数，用于数值稳定性
        scale = w * (rv + eps).rsqrt()#计算缩放因子
        bias = b - rm * scale#计算bias偏移
        return x * scale + bias#返回应用了批量归一化的输入张量


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)#通过调用模型的body属性来处理tensor_list.tensor中的张量。xs是一个包含特征图的字典，其中键是层的名字，值是对应的特征图
        out: Dict[str, NestedTensor] = {}#初始化一个名为out的字典，用于存储输出的NestedTensor对象，字典的键是字符串，值是NestedTensor类型
        for name, x in xs.items():#遍历xs字典中的每一项，name是层的名称，x是对应的特征图
            m = tensor_list.mask#获取输入的tensor_list掩码
            assert m is not None#确保已经提供了掩码
            #使用插值函数F.interpolate将掩码m调整到与特征图x相同的空间尺寸，m[None]是增加一个新的批次维度，
            # .float()将掩码转换为浮点类型，size=x.shape[-2:]指定新的大小，.to(torch.bool)将结果转换为布尔类型的张量，[0]取出批次维度
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #创建一个新的NestedTensor对象，包含特征图x和对应的掩码mask，然后将这个对象存储到输出字典out中，键是层的名称
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, wavelet="db2"):
        norm_layer = FrozenBatchNorm2d  #设置批量归一化层为冻结状态
        #动态获取指定名称的ResNet模型，并根据是否使用空洞卷积对模型进行配置
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        #调用父类初始化方法，传入配置好的ResNet模型，是否训练back_bone，是否返回中间层特征
        super().__init__(backbone, train_backbone, return_interm_layers)
        #如果使用空洞卷积，则调整步长以适应空洞卷积
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
            # 假设 self.num_channels 包含了每个卷积层输出的通道数

        self.cbam_blocks = nn.ModuleList([WaveletCBAM(channels, wavelet=wavelet) for channels in self.num_channels])

class Joiner(nn.Sequential):#将骨干网络backbone和位置嵌入position embedding结合
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides#从backbone中获取步长并将其保存为Joiner类的一个属性
        self.num_channels = backbone.num_channels#从backbone中获取步长并保存为一个属性

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)#通过调用序列中的第一个模块（其backbone）来处理输入的tensor_list，得到输出xs
        out: List[NestedTensor] = []#初始化一个空列表out用于存储输出的NestedTensor对象
        pos = []#初始化一个空列表pos,用于存储位置编码
        for name, x in sorted(xs.items()):#遍历字典中xs中的项（已排序），将对应的NestedTensor添加到out
            out.append(x)

        # position encoding
        for x in out:#遍历out中的每个NestedTensor
            pos.append(self[1](x).to(x.tensors.dtype))#调用序列中的第二个模块（position_embedding）来处理x，将结果中的数据类型转换为
            #与x中的张量相同的类型，并将其添加到位置编码列表pos中

        return out, pos#返回输出的NestedTensor列表out和位置编码列表pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    wavelet = 'db2'
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, wavelet=wavelet)
    model = Joiner(backbone, position_embedding)
    return model