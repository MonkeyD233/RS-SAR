# SVNet-TMVANet: Single-View version of TMVA-Net w/ 3D Conv temporal backbone
# Trimmed from tmvanet.py following the same logic as svnet_adapkctheta.py vs adapkc_theta.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=128, dataset_type='RS-SAR'):
        super().__init__()
        self.dataset_type = dataset_type
        # For RSSAR dataset
        if self.dataset_type == 'RS-SAR':
            self.global_avg_pool = nn.AvgPool2d((72, 72))
        else:
            raise KeyError("we only support RSSAR dataset for now, please set dataset_type='RS-SAR'")
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        # For RSSAR dataset
        if self.dataset_type == 'RS-SAR':
            x1 = F.interpolate(self.global_avg_pool(x), size=(72, 72), align_corners=False,
                               mode='bilinear')
        else:
            raise KeyError("we only support RSSAR dataset for now, please set dataset_type='RS-SAR'")
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        # output: 5 * out_ch = 640 channels
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view (range_doppler only)

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    n_frames: int
        Total number of frames used as a sequence
    dataset_type: str
        Dataset type. Currently only 'RS-SAR' is supported.
    """

    def __init__(self, signal_type, n_frames, dataset_type='RS-SAR'):
        super().__init__()
        self.signal_type = signal_type
        self.n_frames = n_frames
        self.dataset_type = dataset_type
        # 3D conv: out_ch reduced from 128 to 32; temporal kernel size is n_frames-aware
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=32,
                                                      k_size=(1 + n_frames // 2, 3, 3),
                                                      pad=(0, 1, 1), dil=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        # Second block: 32 -> 64 (2D conv after temporal dim is squeezed)
        self.double_conv_block2 = DoubleConvBlock(in_ch=32, out_ch=64, k_size=3,
                                                  pad=1, dil=1)
        # 1x1 conv: 64 -> 128 (latent)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=64, out_ch=128, k_size=1,
                                                 pad=0, dil=1)

    def forward(self, x):
        # x: (B, 1, n_frames, H, W)
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension -> (B, 32, H, W)

        # For RSSAR: uniform max pool
        if self.dataset_type == 'RS-SAR':
            x1_pad = F.pad(x1, (0, 0, 0, 0), "constant", 0)
            x1_down = self.max_pool(x1_pad)
        else:
            raise KeyError("we only support RSSAR dataset for now, please set dataset_type='RS-SAR'")

        x2 = self.double_conv_block2(x1_down)
        x2_pad = F.pad(x2, (0, 0, 0, 1), "constant", 0)
        x2_down = self.max_pool(x2_pad)

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block (x2_down, 64ch) + latent features (x3, 128ch)
        return x2_down, x3


class SVNet_TMVA(nn.Module):
    """
    Single-View version of TMVA-Net w/ 3D Conv temporal backbone (SVNet-TMVA)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total number of frames used as a sequence
    dataset_type: str
        Dataset type. Currently only 'RS-SAR' is supported.
    """

    def __init__(self, n_classes, n_frames, dataset_type='RS-SAR'):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone (encoding) — single RD branch only
        self.rd_encoding_branch = EncodingBranch('range_doppler', n_frames, dataset_type)

        # ASPP Block: in_ch=64 (from encoder x2_down), out_ch=128 → 5*128=640 output
        self.rd_aspp_block = ASPPBlock(in_ch=64, out_ch=128, dataset_type=dataset_type)
        # Compress ASPP output 640 -> 128
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoding: cat(ASPP 128, latent 128) = 256 channels
        self.rd_upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # Final 1x1 conv
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x_rd):
        # x_rd: (B, 1, n_frames, H, W)
        # Backbone
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)

        # ASPP block
        x1_rd = self.rd_aspp_block(rd_features)
        x2_rd = self.rd_single_conv_block1_1x1(x1_rd)

        # Latent Space + ASPP features: cat(128, 128) = 256
        x4_rd = torch.cat((x2_rd, rd_latent), 1)

        # Decoding branch with upconvs
        x5_rd = self.rd_upconv1(x4_rd)
        x6_rd = self.rd_double_conv_block1(x5_rd)

        x7_rd = self.rd_upconv2(x6_rd)
        x8_rd = self.rd_double_conv_block2(x7_rd)

        # Final 1x1 convolution
        x9_rd = self.rd_final(x8_rd)

        return x9_rd