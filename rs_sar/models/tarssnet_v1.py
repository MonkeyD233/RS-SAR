"""
TARSSNet_V1: Temporal-Aware Radar Semantic Segmentation Network (v1).

Single-view (RA-only) adaptation for the RS-SAR training framework.
Core innovation: Spatial squeeze TAM with ATTPool (channel-wise attentive
temporal pooling with sinusoidal positional encoding).

Original: TARSS-Net/mvrss/models/tarssnet_v1.py (multi-view RD+RA+AD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Basic Convolutional Blocks
# ==============================================================================

class DoubleConvBlock(nn.Module):
    """(2D conv => BN => LeakyReLU) * 2"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Double3DConvBlock(nn.Module):
    """(3D conv => BN => LeakyReLU) * 2  (temporally shared 2D conv)"""

    def __init__(self, in_ch, out_ch, k_size, strd, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=strd,
                      padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, stride=strd,
                      padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    """2D conv => BN => LeakyReLU"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TREBlock(nn.Module):
    """Temporal Relation Embeddings: 3D conv => BN => LeakyReLU"""

    def __init__(self, in_ch, out_ch, k_size, strd, pad):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ==============================================================================
# ASPP Block  (72x72 for RSSAR 288 input)
# ==============================================================================

class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling with 5 parallel branches."""

    def __init__(self, in_ch, out_ch=256, feat_size=72):
        super().__init__()
        self.feat_size = feat_size
        self.global_avg_pool = nn.AvgPool2d((feat_size, feat_size))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0,
                                   dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1,
                                                pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3,
                                                pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3,
                                                pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3,
                                                pad=18, dil=18)

    def forward(self, x):
        sz = (self.feat_size, self.feat_size)
        x1 = F.interpolate(self.global_avg_pool(x), size=sz,
                           align_corners=False, mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        return torch.cat((x2, x3, x4, x5, x1), 1)


# ==============================================================================
# Encoding Branch  (RA-only, preserves temporal dim for TAM)
# ==============================================================================

class EncodingBranch(nn.Module):
    """
    Single-view (range-angle) encoding branch.

    Unlike svnet_freqpkc, the temporal dimension is preserved throughout so
    that the downstream TAM can perform attentive temporal aggregation.
    """

    def __init__(self, n_frames):
        super().__init__()
        self.n_frames = n_frames

        self.double_3dconv_block1 = Double3DConvBlock(
            in_ch=1, out_ch=128, k_size=(1, 3, 3), strd=(1, 1, 1),
            pad=(0, 1, 1), dil=1)

        self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                       stride=(1, 2, 2))

        self.double_3dconv_block2 = Double3DConvBlock(
            in_ch=128, out_ch=128, k_size=(1, 3, 3), strd=(1, 1, 1),
            pad=(0, 1, 1), dil=1)

        self.temp_avg_pool = nn.AvgPool3d(kernel_size=(self.n_frames, 1, 1),
                                          stride=(self.n_frames, 1, 1))

        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128,
                                                k_size=1, pad=0, dil=1)

    def forward(self, x):
        # x: (B, 1, T, H, W)
        x1 = self.double_3dconv_block1(x)              # (B, 128, T, H, W)
        x1_down = self.max_pool3d(x1)                  # (B, 128, T, H/2, W/2)

        x2 = self.double_3dconv_block2(x1_down)        # (B, 128, T, H/2, W/2)
        x2_down = self.max_pool3d(x2)                  # (B, 128, T, H/4, W/4)

        # Temporal average pool → spatial latent
        x2_tmp = self.temp_avg_pool(x2_down)           # (B, 128, 1, H/4, W/4)
        x2_tmp_sq = torch.squeeze(x2_tmp, 2)           # (B, 128, H/4, W/4)

        x3 = self.single_conv_block1_1x1(x2_tmp_sq)    # (B, 128, H/4, W/4)

        # Return temporal features (for TAM) and spatial latent (for ASPP)
        return x2_down, x3


# ==============================================================================
# Positional Encoding  ("Attention Is All You Need" style)
# ==============================================================================

class PosEnc(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, coef=0.1, const=8):
        super().__init__()
        self.Coef = coef
        self.Const = const

    def forward(self, batch_size, nfrms, chns):
        pos_vec = torch.zeros(nfrms, chns)
        weight = torch.ones(1)
        for i in range(int(chns / 2) - 1):
            for t in range(nfrms):
                weight = weight * t / pow(10, self.Const * i / chns)
                pos_vec[t][2 * i] = self.Coef * torch.sin(weight)
                pos_vec[t][2 * i + 1] = self.Coef * torch.cos(weight)
        pos_tensor = torch.unsqueeze(pos_vec, 0)
        pos_tensor = torch.repeat_interleave(pos_tensor, repeats=batch_size,
                                             dim=0)
        return pos_tensor


# ==============================================================================
# Attentive Temporal Pooling  (channel-wise attention over time)
# ==============================================================================

class ATTPool(nn.Module):
    """
    Channel-wise attentive pooling with positional encoding.

    Takes temporal relation embeddings x_tre (B, T, C) and a feature tensor
    x (B, C, T, H, W).  Learns per-channel attention scores via a linear
    projection + softmax, then computes a weighted sum over time.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pos_enc = PosEnc(0.1, 8)
        self.fc_layer = nn.Linear(in_ch, out_ch)
        self.att_scoring = nn.Softmax(dim=1)

    def forward(self, x, x_tre):
        # x:     (B, C, T, H, W)
        # x_tre: (B, T, C)
        B, C, T = x.shape[:3]

        x_pos = self.pos_enc(B, T, C).to(x_tre.device)
        x_tre = torch.add(x_tre, x_pos)

        x_att = self.fc_layer(x_tre)                   # (B, T, C)
        att_sc = self.att_scoring(x_att)               # softmax over T
        att_sc = att_sc.permute(0, 2, 1)               # (B, C, T)
        att_sc = att_sc.unsqueeze(3).unsqueeze(4)      # (B, C, T, 1, 1)

        x_att = att_sc * x
        x_att = torch.sum(x_att, dim=2)                # (B, C, H, W)
        x_att = x_att + x[:, :, -1, :, :]              # residual
        return x_att


# ==============================================================================
# Temporal-Aware Module  (spatial squeeze variant)
# ==============================================================================

class TAM(nn.Module):
    """
    TAM with spatial squeeze.

    1. Differential encoding: concat(target_frame, full_seq) → 2 TREBlocks
    2. Spatial squeeze: AvgPool3d + MaxPool3d over (H, W) → motion cues
    3. Linear projection → ATTPool for attentive temporal aggregation
    """

    def __init__(self, in_ch, out_ch, signal_type=None):
        super().__init__()
        self.tr_embedding_1 = TREBlock(in_ch * 2, out_ch,
                                       k_size=(1, 3, 3), strd=(1, 2, 2),
                                       pad=(0, 1, 1))
        self.tr_embedding_2 = TREBlock(out_ch, out_ch,
                                       k_size=(1, 3, 3), strd=(1, 2, 2),
                                       pad=(0, 1, 1))

        self.spatial_squeeze_avg = nn.AvgPool3d(kernel_size=(1, 16, 16),
                                                stride=(1, 16, 16))
        self.spatial_squeeze_max = nn.MaxPool3d(kernel_size=(1, 16, 16),
                                                stride=(1, 16, 16))
        self.fc_layer_avg_max = nn.Linear(out_ch * 2, out_ch)
        self.att_pool = ATTPool(out_ch, out_ch)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x_tgt = x[:, :, -1, :, :].unsqueeze(2).expand_as(x)
        x0 = torch.cat((x_tgt, x), 1)                  # (B, 2C, T, H, W)

        x1 = self.tr_embedding_1(x0)                   # (B, C, T, H/2, W/2)
        x2 = self.tr_embedding_2(x1)                   # (B, C, T, H/4, W/4)

        x2_spsq_avg = self.spatial_squeeze_avg(x2)     # (B, C, T, 1, 1)
        x2_spsq_max = self.spatial_squeeze_max(x2)
        x2_spsq = torch.cat((x2_spsq_avg, x2_spsq_max), 1)  # (B, 2C, T, 1, 1)
        x_tre = torch.squeeze(x2_spsq, 3)
        x_tre = torch.squeeze(x_tre, 3)                # (B, 2C, T)
        x_tre = x_tre.permute(0, 2, 1)                 # (B, T, 2C)
        x_tre = self.fc_layer_avg_max(x_tre)           # (B, T, C)

        x_tam = self.att_pool(x, x_tre)                # (B, C, H, W)
        return x_tam


# ==============================================================================
# TARSSNet_V1  (single-view RA-only)
# ==============================================================================

class TARSSNet_V1(nn.Module):
    """
    TARSS-Net V1 — single-view adaptation for rs_sar.

    Constructor signature matches the project convention:
        (n_classes, n_frames, dataset_type='RS-SAR')
    """

    def __init__(self, n_classes, n_frames, dataset_type='RS-SAR'):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone
        self.ra_encoding_branch = EncodingBranch(n_frames)

        # Temporal-Aware Module
        self.ra_tam_branch = TAM(in_ch=128, out_ch=128)

        # ASPP
        self.ra_aspp_block = ASPPBlock(in_ch=128, out_ch=128, feat_size=72)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128,
                                                   k_size=1, pad=0, dil=1)

        # TAM-ASPP Fusion
        self.ra_single_conv_block1_1x1_tsf = ConvBlock(in_ch=256, out_ch=128,
                                                       k_size=1, pad=0, dil=1)

        # Decoder
        self.ra_upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128,
                                                     k_size=3, pad=1, dil=1)
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128,
                                                     k_size=3, pad=1, dil=1)

        # Final 1x1 conv
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes,
                                  kernel_size=1)

    def forward(self, x_ra):
        # x_ra: (B, 1, T, H, W)

        # Backbone
        ra_features, ra_latent = self.ra_encoding_branch(x_ra)

        # TAM
        ra_tmp_aug = self.ra_tam_branch(ra_features)

        # ASPP
        x1_ra = self.ra_aspp_block(ra_latent)
        x2_ra = self.ra_single_conv_block1_1x1(x1_ra)

        # Temporal-Spatial Fusion
        ra_tam_aspp = torch.cat((ra_tmp_aug, x2_ra), 1)
        ra_tsf = self.ra_single_conv_block1_1x1_tsf(ra_tam_aspp)

        # Decoder input: TSF + latent
        x4_ra = torch.cat((ra_tsf, ra_latent), 1)

        # Decoding
        x5_ra = self.ra_upconv1(x4_ra)
        x6_ra = self.ra_double_conv_block1(x5_ra)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_ra = self.ra_double_conv_block2(x7_ra)

        # Final classification
        x9_ra = self.ra_final(x8_ra)

        return x9_ra
