import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

# ==========================================
# Layers (from texo/model/layer.py)
# ==========================================

class PaddingSameAsPaddleMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = torch.nn.MaxPool2d(kernel_size, stride, padding=0, ceil_mode=True)

    def forward(self, x):
        _, _, h, w = x.shape
        pad_h_total = max(0, (math.ceil(h / self.stride) - 1) * self.stride + self.kernel_size - h)
        pad_w_total = max(0, (math.ceil(w / self.stride) - 1) * self.stride + self.kernel_size - w)
        pad_h = pad_h_total // 2
        pad_w = pad_w_total // 2
        x = torch.nn.functional.pad(x, [pad_w, pad_w_total - pad_w, pad_h, pad_h_total - pad_h])
        return self.pool(x)

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return "{num_features}, eps={eps}".format(**self.__dict__)

def freeze_batch_norm2d(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.BatchNorm2d):
        module = FrozenBatchNorm2d(module.num_features)
    else:
        for name, child in module.named_children():
            _child = freeze_batch_norm2d(child)
            if _child is not child:
                setattr(module, name, _child)
    return module

class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        use_act=True,
    ):
        super().__init__()
        self.use_act = use_act

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding= (kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        if self.use_act:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x

class LightConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            use_act=False,
        )
        self.conv2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# ==========================================
# HGNetV2 (from texo/model/hgnet2.py)
# ==========================================

__all__ = ["HGNetv2", "HGNetv2Config"]

class StemBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, mid_channels, kernel_size=3, stride=2)
        self.stem2a = ConvBNAct(mid_channels, mid_channels // 2, kernel_size=2, stride=1)
        self.stem2b = ConvBNAct(mid_channels // 2, mid_channels, kernel_size=2, stride=1)
        self.stem3 = ConvBNAct(mid_channels * 2, mid_channels, kernel_size=3, stride=2)
        self.stem4 = ConvBNAct(mid_channels, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

class HG_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        layer_num,
        kernel_size=3,
        residual=False,
        light_block=True,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_channels if i == 0 else mid_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_channels if i == 0 else mid_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                        stride=1,
                    )
                )

        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            total_channels,
            out_channels // 2,
            kernel_size=1,
            stride=1,
        )
        self.aggregation_excitation_conv = ConvBNAct(
            out_channels // 2,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.residual:
            x = x + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
        self,
        in_channels:int,
        mid_channels:int,
        out_channels:int,
        num_blocks:int,
        num_layers:int,
        kernel_size:int=3,
        downsample:bool=True,
        light_block:bool=True,
    ):
        super().__init__()
        self.use_downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
            )

        blocks_list = []
        for i in range(num_blocks):
            blocks_list.append(
                HG_Block(
                    in_channels if i == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    num_layers,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.use_downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x

class HGNetv2Config(PretrainedConfig):
    model_type = "my_hgnetv2"

    def __init__(
        self,
        stem_channels: List[int]=[3, 32, 48],
        stage_config: Dict[str, Tuple[int,int,int,int,int,int,bool,bool]]={
            "stage1": (48, 48, 128, 1, 6, 3, False, False),
            "stage2": (128, 96, 512, 1, 6, 3, True, False),
            "stage3": (512, 192, 1024, 3, 6, 5, True, True),
            "stage4": (1024, 384, 2048, 1, 6, 5, True, True),
        },
        hidden_size:int=384,
        pretrained:str|Path="",
        freeze:bool=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stem_channels = stem_channels
        self.stage_config = stage_config
        self.hidden_size = hidden_size
        self.pretrained = pretrained
        self.freeze = freeze

class HGNetv2(PreTrainedModel):
    config_class = HGNetv2Config
    base_model_prefix = "my_hgnetv2"
    main_input_name = "pixel_values"

    def __init__(self, config:HGNetv2Config):
        super().__init__(config)
        self.stem = StemBlock(*config.stem_channels)
        self.stages = nn.ModuleList(HG_Stage(*config.stage_config[k]) for k in config.stage_config)

        if config.pretrained:
            logging.log(logging.INFO, f"load pretrained model from {config.pretrained}")
            state_dict = torch.load(config.pretrained)
            self.load_state_dict(state_dict)

        if config.freeze:
            logging.log(logging.INFO, "freeze model weight")
            self._freeze_norm(self)
            self._freeze_parameters(self)


    def forward(self, pixel_values, **kwargs):
        x = self.stem(pixel_values)
        for stage in self.stages:
            x = stage(x)
        out = x.flatten(2).transpose(1,2)
        return BaseModelOutput(last_hidden_state=out)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False
