from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from lib.models.DCNv2.dcn_v2 import DCN
from .conv_gru import ConvGRU
from .deconv_gru import DeConvGRU

import matplotlib.pyplot as plt

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x

class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        # self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
        #                    level_root=True, root_residual=residual_root)

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, gru):
        y = []
        x = self.base_layer(x)
        x = x + gru
        for i in range(5):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)

def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            #layers[i] = project(layers[i])
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            # print(i, channels[j], in_channels[j:])
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):  # 0, 1, 2, 3
            # print(i, layers[0].shape, layers[1].shape, layers[2].shape, layers[3].shape, layers[4].shape)
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])  # 4, 3, 2, 1
        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, temp):
        B, C, H, W = x1.shape
        N = H * W
        x1 = x1.view(B, C, -1).transpose(1, 2)    # nW * B, N, C
        temp = temp.view(B, C, -1).transpose(1, 2)    # nW * B, N, C

        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(temp).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x.transpose(1, 2).view(B, C, H, W)

class MaskProp(nn.Module):
    def __init__(self, dim, win_size):
        super(MaskProp, self).__init__()
        self.ws = win_size
        self.low_thr = 0.2
        self.attn_cross = Attention(dim, 4)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, pre_feat, feat, mask, name=None):
        B, C, H, W = feat.shape
        _, _, H_org, W_org = mask.shape
        # downsample mask
        mask = F.interpolate(mask, [H, W], mode='bilinear', align_corners=False)

        # ----------- partition -----------
        feat_patch = window_partition(feat, self.ws)           # Wn, C, Ws, Ws
        pre_feat_patch = window_partition(pre_feat, self.ws)   # Wn, C, Ws, Ws
        mask_patch = window_partition(mask, self.ws)           # Wn, 1, Ws, Ws
        Wn, _, _, _ = mask_patch.shape

        # ----------- choose -----------
        mask_max = mask_patch.view(-1, 1, self.ws * self.ws).max(dim=-1)[0].squeeze(1)  # Wn
        keep_index = torch.where(mask_max >= self.low_thr)[0].cpu().numpy().tolist()   # <=Wn, list
        if len(keep_index) == 0:
            keep_index = [0]

        # ----------- * -----------
        pre_patch_kept = pre_feat_patch[keep_index, :]                                      # Ind, C, Ws, Ws
        feat_patch_kept = feat_patch[keep_index, :]                                         # Ind, C, Ws, Ws
        mask_patch_kept = mask_patch[keep_index, :]                                         # Ind, 1, Ws, Ws
        heat_attn = self.attn_cross(feat_patch_kept, pre_patch_kept * mask_patch_kept)      # Ind, C, Ws, Ws
        heat_compe = torch.zeros(Wn, C, self.ws, self.ws).cuda()
        heat_compe[keep_index, :, :, :] = heat_attn

        # ----------- reverse -----------
        heat_enhance = window_reverse(heat_compe, self.ws, H, W)       # B, C, H, W
        heat = self.bn(heat_enhance + feat)
        return heat

class ConvGRUNet(nn.Module):
    def __init__(self):
        super(ConvGRUNet, self).__init__()
        self.conv_gru_layer1 = DeConvGRU(3, 16, (7, 7), 1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

        self.conv_gru_layer2 = DeConvGRU(16, 16, (3, 3), 1)
    
    def forward(self, c0_sta):
        # layer 1
        c0_seq, _ = self.conv_gru_layer1(c0_sta)                    # B, N, 32, 512, 512
        c0_seq_down = self.maxpool1(c0_seq.transpose(1, 2)).transpose(1, 2)     # B, N, 32, 256, 256

        # layer 2
        c1_seq, _ = self.conv_gru_layer2(c0_seq_down)               # B, N, 64, 256, 256

        return c1_seq


class DeConvGRUNet(nn.Module):
    def __init__(self):
        super(DeConvGRUNet, self).__init__()
        self.proj0 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.proj_hm = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.deconv_gru_layer = DeConvGRU(16, 16, (3, 3), 1)

    def forward(self, p0_sta):
        B, N, _, H, W = p0_sta.shape
        p0_seq_gru, _ = self.deconv_gru_layer(p0_sta)          # B, N, 16, 512, 512
        final_hm = p0_seq_gru[:, -1] + self.proj_hm(p0_sta[:, -1])
        for i in range(N - 1):
            if i == 0:
                final_gru = self.proj0(torch.cat((p0_seq_gru[:, i], p0_seq_gru[:, i + 1]), dim=1)).unsqueeze(1)
            else:
                final_temp = self.proj0(torch.cat((p0_seq_gru[:, i], p0_seq_gru[:, i + 1]), dim=1)).unsqueeze(1)
                final_gru = torch.cat((final_gru, final_temp), dim=1)       # B, N - 1, 32, 512, 512

        return final_gru, final_hm


class DLASeg(nn.Module):
    def __init__(self, heads, final_kernel,
                  head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        self.prev_state = False
        self.first_level = 0  # int(np.log2(down_ratio))
        self.last_level = 3  # last_level
        # ---------down---------
        # static down
        self.backbone = dla34(pretrained=True)

        # sequence down
        self.conv_gru = ConvGRUNet()
        # ---------down---------

        # ---------mask prop---------
        self.mask_prop1 = MaskProp(32, 8)
        self.mask_prop2 = MaskProp(64, 8)
        # ---------mask prop---------

        # ---------up---------
        channels = [16, 32, 64, 128, 256]  # self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]  # 1, 2, 4, 8, 16
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        # sequence up
        self.deconv_gru = DeConvGRUNet()

        # generate layer mask
        self.mask_layer = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.mask_layer.bias.data.fill_(-4.6)
        # ---------up---------

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-4.6)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)


    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, in_filters, planes, num_kernels, with_dcn=True):
        layers = []

        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels)

        up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
        fill_up_weights(up)

        if with_dcn:
            fc = DCN(in_filters, planes,
                kernel_size=(3,3), stride=1,
                padding=1, dilation=1, deformable_groups=1)
            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

        layers.append(up)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def train_feat(self, x, gru, i, p1_pre=None, p2_pre=None, mask=None):
        B, C, H, W = x.shape

        if mask == None:
            mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=x.device)
            p1_pre = torch.zeros((B, 32, H // 2, W // 2), dtype=torch.float32, device=x.device)
            p2_pre = torch.zeros((B, 64, H // 4, W // 4), dtype=torch.float32, device=x.device)
        # static down
        c0, c1, c2, c3, c4 = self.backbone(x, gru[0][:, i])   # 16, 32, 64, 128
        c1 = self.mask_prop1(p1_pre, c1, mask)
        c2 = self.mask_prop2(p2_pre, c2, mask)
        p0, p1, p2, p3, _ = self.dla_up([c0, c1, c2, c3, c4])
        # ida up
        y = [p0, p1, p2]
        self.ida_up(y, 0, len(y))
        p0 = y[-1]
        return p0, p1, p2, p3

    def track_feat(self, x, gru, i, N, p1_pre=None, p2_pre=None, mask=None):
        B, _, H, W = x.shape

        if self.prev_state != False and i < N - 1:
            return self.prev_feat[i + 1][0], self.prev_feat[i + 1][1], self.prev_feat[i + 1][2], self.prev_feat[i + 1][3]
        else:
            if mask == None:
                mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=x.device)
                p1_pre = torch.zeros((B, 32, H // 2, W // 2), dtype=torch.float32, device=x.device)
                p2_pre = torch.zeros((B, 64, H // 4, W // 4), dtype=torch.float32, device=x.device)
            # static down
            c0, c1, c2, c3, c4 = self.backbone(x, gru[0][:, i])   # 16, 32, 64, 128
            c1 = self.mask_prop1(p1_pre, c1, mask)
            c2 = self.mask_prop2(p2_pre, c2, mask)
            p0, p1, p2, p3, _ = self.dla_up([c0, c1, c2, c3, c4])
            # ida up
            y = [p0, p1, p2]
            self.ida_up(y, 0, len(y))
            p0 = y[-1]
            return p0, p1, p2, p3

    def forward(self, img_input, training=True, vid=None):
        # x : B, N, C, H, W
        # ..., -3, -2, -1, 0
        B, N, _, H, W = img_input.shape
        mask = None
        p1_pre = None
        p2_pre = None
        gru_set = []
        temp_feat = []
        ret_temp = {}

        # ------------------down begin------------------
        # sequence down
        c0_gru = self.conv_gru(img_input)
        gru_set.extend([c0_gru])

        # static down and up
        for i in range(N):
            x = img_input[:, i, :]
            if training:
                p0, p1, p2, p3 = self.train_feat(x, gru_set, i, p1_pre, p2_pre, mask)
                temp_feat.append([p0, p1, p2, p3])    # N, 4
            else:
                p0, p1, p2, p3 = self.track_feat(x, gru_set, i, N, p1_pre, p2_pre, mask)
                temp_feat.append([p0, p1, p2, p3])    # N, 4
            
            p1_pre = p1
            p2_pre = p2

            # mask prop
            if i == 0:
                mask_out = self.mask_layer(p0).unsqueeze(1)                                 # B, N, 1, 512, 512
            else:
                mask_out = torch.cat((mask_out, self.mask_layer(p0).unsqueeze(1)), dim=1)

            mask = F.sigmoid(mask_out[:, i])

            # construct sequence
            if i == 0:
                p0_seq = p0.unsqueeze(1)
            else:
                p0_seq = torch.cat((p0_seq, p0.unsqueeze(1)), dim=1)    # B, N, 32, 512, 512
        # ------------------down end------------------

        # ------------------up begin------------------
        # sequence up
        final_seq, final_hm = self.deconv_gru(p0_seq)       # B, N - 1, 32, 512, 512
        # ------------------up end------------------

        ret_temp['hm_seq'] = mask_out       # B, N, 1, 512, 512s

        # only for track
        if not training:
            self.prev_feat = temp_feat
            self.prev_state = True

        ret = {}
        for head in self.heads:
            if 'dis' in head:
                trk_out = self.__getattr__(head)(final_seq[:, 0]).unsqueeze(1)
                for i in range(1, N - 1):
                    x = self.__getattr__(head)(final_seq[:, i]).unsqueeze(1)
                    trk_out = torch.cat((trk_out, x), dim=1)
                ret_temp[head] = trk_out
            else:
                ret_temp[head] = self.__getattr__(head)(final_hm)
        ret[1] = ret_temp
        return [temp_feat, ret]


def dla_dcn_net(heads, head_conv=128):
    model = DLASeg(heads,final_kernel=1,
                 head_conv=head_conv)
    return model