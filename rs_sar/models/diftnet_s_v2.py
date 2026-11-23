"""
DiFTNet_S_v2: Dive into Frequency Temporal — Spatial DiFT variant.

Single-view (RA-only) adaptation for the RS-SAR training framework.
Core innovation: Spatio-DiFT with dual-core frequency-adaptive mechanism —
local LSI (spatial attention inversion) + GDSE (1D-FFT spectral equalization),
with learnable weighted temporal compression (v2.1).

Original: TARSS-Net/mvrss/models/diftnet_s_v2.py (multi-view RD+RA+AD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
# Activation Helpers
# ==============================================================================

class StarReLU(nn.Module):
    """StarReLU activation: s * ReLU(x)^2 + b"""

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_value))
        self.bias = nn.Parameter(torch.tensor(bias_value))

    def forward(self, x):
        return self.scale * F.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    """MLP with StarReLU for global dynamic routing in FFT spectral equalization."""

    def __init__(self, dim, mlp_ratio=4, out_features=None,
                 act_layer=StarReLU, drop=0., bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ==============================================================================
# Global Dynamic Spectral Equalization (GDSE)
# ==============================================================================

class GlobalDynamicSpectralEqualization(nn.Module):
    """
    1D-FFT spectral equalization with learned channel-grouped complex weights.

    Applies per-position frequency gain curves via spatially-aware routing,
    preserving phase information.  canonical_freq_len=6 is fixed so the
    module is agnostic to the number of input frames.
    """

    def __init__(self, dim, expansion_ratio=0.125, group=16, num_filters=4,
                 init_scale=1e-5):
        super().__init__()
        self.dim = dim
        self.num_filters = num_filters
        self.group = group
        canonical_freq_len = 6

        self.reweight = nn.Conv2d(dim, group * num_filters, kernel_size=1,
                                  bias=False)

        self.complex_weights = nn.Parameter(
            torch.randn(num_filters, dim // group, canonical_freq_len,
                        dtype=torch.float32) * init_scale
        )
        nn.init.trunc_normal_(self.complex_weights, std=init_scale)

    def forward(self, x):
        B, C, T, H, W = x.shape

        x_rfft = torch.fft.rfft(x.to(torch.float32), dim=2, norm='ortho')
        freq_len = x_rfft.shape[2]

        x_pooled = x.mean(dim=2)  # [B, C, H, W]
        routeing = self.reweight(x_pooled).view(B, self.num_filters, self.group, H, W).tanh_()

        weight = self.complex_weights
        if weight.shape[-1] != freq_len:
            nf, cpg, _ = weight.shape
            weight = weight.reshape(1, nf * cpg, -1)
            weight = F.interpolate(weight, size=freq_len, mode='linear',
                                   align_corners=True)
            weight = weight.reshape(nf, cpg, freq_len)

        weight = torch.einsum('bfghw,fcq->bgcqhw', routeing, weight)
        weight = weight.reshape(B, C, freq_len, H, W)

        x_rfft = torch.view_as_complex(
            torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1)
        )

        x_out = torch.fft.irfft(x_rfft, n=T, dim=2, norm='ortho')
        return x_out


# ==============================================================================
# Spatial-level Dynamic Frequency Weights  (channel-shared)
# ==============================================================================

class TemporalDynamicFreqWeight(nn.Module):
    """Spatial-level, channel-shared dynamic frequency weights."""

    def __init__(self, in_ch):
        super().__init__()
        self.dy_freq_starrelu = StarReLU()
        self.dy_freq_lf_conv = nn.Conv3d(in_ch, 1, kernel_size=1)
        self.dy_freq_hf_conv = nn.Conv3d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        dy_freq_feat = self.dy_freq_starrelu(x)
        dy_freq_lf = self.dy_freq_lf_conv(dy_freq_feat).tanh()
        dy_freq_hf = F.softplus(self.dy_freq_hf_conv(dy_freq_feat))
        dy_freq2 = dy_freq_hf ** 2
        dy_freq_hf = 2 * dy_freq2 / (dy_freq2 + 0.3678)
        return dy_freq_lf, dy_freq_hf


# ==============================================================================
# DiFT Module  (spatial LSI + GDSE)
# ==============================================================================

class DiFT(nn.Module):
    """
    Dual-core frequency-adaptive module (spatial variant).

    Core 1 — Local LSI: generates a spatial attention mask from motion cues,
        separates low/high frequency components via impulse subtraction,
        and applies dynamic spatial weights.
    Core 2 — Global GDSE: 1D-FFT spectral equalization with learned complex
        weights for phase-preserving frequency modulation.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.att_embedding_1 = TREBlock(2, 16, k_size=(1, 3, 3),
                                        strd=(1, 1, 1), pad=(0, 1, 1))
        self.att_embedding_2 = TREBlock(16, 1, k_size=(1, 1, 1),
                                        strd=(1, 1, 1), pad=(0, 1, 1))
        self.att_scoring = nn.Softmax2d()

        self.dyn_weight_gen = TemporalDynamicFreqWeight(in_ch)

        self.lf_gamma = nn.Parameter(1e-5 * torch.ones(1, in_ch, 1, 1, 1))
        self.hf_gamma = nn.Parameter(1e-5 * torch.ones(1, in_ch, 1, 1, 1))

        self.gdse = GlobalDynamicSpectralEqualization(dim=in_ch)
        self.gdse_gamma = nn.Parameter(1e-5 * torch.ones(1, in_ch, 1, 1, 1))

        # Learnable temporal compression logits: stored at canonical size
        # (zero-init => uniform window, i.e., plain summation), resampled to the
        # actual window length at forward — n_frames-agnostic, mirroring the
        # canonical frequency grid of GDSE.
        canonical_window_len = 5  # fixed canonical size, independent of n_frames
        self.comp_theta = nn.Parameter(torch.zeros(canonical_window_len))

    def forward(self, x, x_tre):
        """
        Args:
            x: Raw feature tensor [B, C, T, H, W]
            x_tre: Motion cue tensor from channel squeeze [B, 2, T, 16, 16]
        Returns:
            x_out: Frequency-enhanced temporal-aggregated features [B, C, H, W]
        """
        B, C, T, H, W = x.shape

        # ---- Core 1: Local LSI ----
        att_tensor = self.att_embedding_1(x_tre)
        att_tensor = self.att_embedding_2(att_tensor)
        att_tensor = torch.squeeze(att_tensor, 1)
        att_mask = self.att_scoring(att_tensor)
        att_mask = F.interpolate(att_mask, [H, W])

        impulse_t = torch.zeros(T, device=att_mask.device, dtype=att_mask.dtype)
        impulse_t[0] = 1.0
        impulse = impulse_t.view(1, T, 1, 1)
        att_hf = impulse - att_mask

        dy_freq_lf, dy_freq_hf = self.dyn_weight_gen(x)

        x_lf = att_mask.unsqueeze(1) * x
        x_hf = att_hf.unsqueeze(1) * x
        x_lsi = (x_lf * dy_freq_lf * self.lf_gamma +
                 x_hf * dy_freq_hf * self.hf_gamma)

        # ---- Core 2: Global GDSE ----
        x_gdse = self.gdse(x_lsi)

        # ---- Fusion ----
        x_fused = x_lsi + (x_gdse * self.gdse_gamma)

        # Temporal aggregation with learnable compression weights + current-frame residual
        theta = self.comp_theta
        if theta.numel() != T:  # resample canonical window to the actual length
            theta = F.interpolate(theta.view(1, 1, -1), size=T,
                                  mode='linear', align_corners=True).view(-1)
        w = torch.softmax(theta, dim=0) * T
        x_out = (x_fused * w.view(1, 1, T, 1, 1)).sum(dim=2) + x[:, :, -1, :, :]
        return x_out


# ==============================================================================
# Temporal-Aware Module  (channel squeeze + DiFT)
# ==============================================================================

class TAM(nn.Module):
    """
    TAM with channel squeeze and DiFT frequency-adaptive pooling.

    1. Differential encoding → 2 TREBlocks with spatial downsampling
    2. Channel squeeze: AvgPool3d + MaxPool3d over C → motion cues
    3. DiFT dual-core frequency-adaptive temporal aggregation
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

        self.dift_block = DiFT(in_ch=out_ch, out_ch=out_ch)

    def forward(self, x):
        x_tgt = x[:, :, -1, :, :].unsqueeze(2).expand_as(x)
        x0 = torch.cat((x_tgt, x), 1)

        x1 = self.tr_embedding_1(x0)
        x2 = self.tr_embedding_2(x1).permute(0, 2, 1, 3, 4)

        x_tre = torch.cat((
            self.channel_squeeze_avg(x2),
            self.channel_squeeze_max(x2)
        ), 2).permute(0, 2, 1, 3, 4)

        return self.dift_block(x, x_tre)


# ==============================================================================
# DiFTNet_S_v2  (single-view RA-only)
# ==============================================================================

class DiFTNet_S_v2(nn.Module):
    """
    DiFT-Net S v2 — single-view adaptation for rs_sar.

    Constructor signature matches the project convention:
        (n_classes, n_frames, dataset_type='RS-SAR')
    """

    def __init__(self, n_classes, n_frames, dataset_type='RS-SAR'):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone
        self.ra_encoding_branch = EncodingBranch(n_frames)

        # Temporal-Aware Module with DiFT
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

        # TAM with DiFT
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
