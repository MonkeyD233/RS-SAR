"""
TARSSNet_V2: Temporal-Aware Radar Semantic Segmentation Network (v2).

Single-view (RA-only) adaptation for the RS-SAR training framework.
Core innovation: Channel squeeze TAM with ATTPool2D (spatial attention over
time via learned 2D softmax mask).

Original: TARSS-Net/mvrss/models/tarssnet_v2.py (multi-view RD+RA+AD)
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
    Preserves temporal dimension throughout for downstream TAM.
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
        x1 = self.double_3dconv_block1(x)
        x1_down = self.max_pool3d(x1)

        x2 = self.double_3dconv_block2(x1_down)
        x2_down = self.max_pool3d(x2)

        x2_tmp = self.temp_avg_pool(x2_down)
        x2_tmp_sq = torch.squeeze(x2_tmp, 2)

        x3 = self.single_conv_block1_1x1(x2_tmp_sq)
        return x2_down, x3


# ==============================================================================
# 2D Attentive Temporal Pooling  (spatial attention over time)
# ==============================================================================

class ATTPool2D(nn.Module):
    """
    Spatio-attentive pooling via a learned 2D softmax mask.

    Takes motion cues x_tre (B, 2, T, 16, 16) and generates a per-pixel
    spatial attention mask, then computes a weighted sum over time.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.att_embedding_1 = TREBlock(2, 16, k_size=(1, 3, 3),
                                        strd=(1, 1, 1), pad=(0, 1, 1))
        self.att_embedding_2 = TREBlock(16, 1, k_size=(1, 1, 1),
                                        strd=(1, 1, 1), pad=(0, 1, 1))
        self.att_scoring = nn.Softmax2d()

    def forward(self, x, x_tre):
        # x:     (B, C, T, H, W)
        # x_tre: (B, 2, T, 16, 16)
        B, C, T, H, W = x.shape

        att_tensor = self.att_embedding_1(x_tre)          # (B, 16, T, 16, 16)
        att_tensor = self.att_embedding_2(att_tensor)     # (B, 1, T, 16, 16)
        att_tensor = torch.squeeze(att_tensor, 1)         # (B, T, 16, 16)
        att_mask = self.att_scoring(att_tensor)           # (B, T, 16, 16)

        att_mask = F.interpolate(att_mask, [H, W])        # (B, T, H, W)
        att_mask = att_mask.unsqueeze(1)                  # (B, 1, T, H, W)
        att_mask = att_mask.expand(-1, C, -1, -1, -1)     # (B, C, T, H, W)
        x_att = att_mask * x

        x_att = torch.sum(x_att, dim=2)                   # (B, C, H, W)
        x_att = x_att + x[:, :, -1, :, :]                # residual
        return x_att


# ==============================================================================
# Temporal-Aware Module  (channel squeeze variant)
# ==============================================================================

class TAM(nn.Module):
    """
    TAM with channel squeeze.

    1. Differential encoding: concat(target_frame, full_seq) → 2 TREBlocks
    2. Channel squeeze: AvgPool3d + MaxPool3d over C → motion cues
    3. ATTPool2D for spatio-attentive temporal aggregation
    """

    def __init__(self, in_ch, out_ch, signal_type=None):
        super().__init__()
        self.tr_embedding_1 = TREBlock(in_ch * 2, out_ch,
                                       k_size=(1, 3, 3), strd=(1, 2, 2),
                                       pad=(0, 1, 1))
        self.tr_embedding_2 = TREBlock(out_ch, out_ch,
                                       k_size=(1, 3, 3), strd=(1, 2, 2),
                                       pad=(0, 1, 1))

        self.channel_squeeze_avg = nn.AvgPool3d(kernel_size=(out_ch, 1, 1),
                                                stride=(out_ch, 1, 1))
        self.channel_squeeze_max = nn.MaxPool3d(kernel_size=(out_ch, 1, 1),
                                                stride=(out_ch, 1, 1))

        self.att_pool_2d = ATTPool2D(out_ch, out_ch)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x_tgt = x[:, :, -1, :, :].unsqueeze(2).expand_as(x)
        x0 = torch.cat((x_tgt, x), 1)                      # (B, 2C, T, H, W)

        x1 = self.tr_embedding_1(x0)                       # (B, C, T, H/2, W/2)
        x2 = self.tr_embedding_2(x1)                       # (B, C, T, H/4, W/4)

        x2 = x2.permute(0, 2, 1, 3, 4)                    # (B, T, C, H/4, W/4)
        x2_chsq_avg = self.channel_squeeze_avg(x2)         # (B, T, 1, H/4, W/4)
        x2_chsq_max = self.channel_squeeze_max(x2)
        x2_chsq = torch.cat((x2_chsq_avg, x2_chsq_max), 2) # (B, T, 2, H/4, W/4)
        x_tre = x2_chsq.permute(0, 2, 1, 3, 4)            # (B, 2, T, H/4, W/4)

        x_tam = self.att_pool_2d(x, x_tre)                 # (B, C, H, W)
        return x_tam


# ==============================================================================
# TARSSNet_V2  (single-view RA-only)
# ==============================================================================

class TARSSNet_V2(nn.Module):
    """
    TARSS-Net V2 — single-view adaptation for rs_sar.

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
