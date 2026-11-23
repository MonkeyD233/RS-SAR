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




class FrequencySelection(nn.Module):
    def __init__(self,
                 in_channels,
                 gb_max,
                 k_list=[2],
                 # freq_list=[2, 3, 5, 7, 9, 11],
                 lowfreq_att=False,
                 fs_feat='feat',
                 lp_type='freq',
                 act='sigmoid',
                 spatial='conv',
                 spatial_group=1,
                 spatial_kernel=3,
                 init='zero',
                 global_selection=False,
                 ):
        super().__init__()
        # k_list.sort()
        # print()
        self.k_list = k_list
        # self.freq_list = freq_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.gb_max = gb_max
        self.in_channels = in_channels

        self.norm = nn.Conv2d(self.in_channels, 4, kernel_size=(gb_max * 2 + 1, gb_max * 2 + 1), padding=(gb_max, gb_max),
                                         dilation=1)
        self.bn = nn.BatchNorm2d(4)
        nn.init.constant_(self.norm.weight, 0)

        # self.residual = residual
        if spatial_group > 64: spatial_group = in_channels
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:  _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(in_channels=in_channels,
                                             out_channels=self.spatial_group,
                                             stride=1,
                                             kernel_size=spatial_kernel,
                                             groups=self.spatial_group,
                                             padding=spatial_kernel // 2,
                                             bias=True)
                if init == 'zero':
                    freq_weight_conv.weight.data.zero_()
                    freq_weight_conv.bias.data.zero_()
                else:
                    # raise NotImplementedError
                    pass
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError

        self.act = act
        # self.freq_weight_conv_list.append(nn.Conv2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 1, kernel_size=1, padding=0, bias=True))
        self.global_selection = global_selection
        if self.global_selection:
            self.global_selection_conv_real = nn.Conv2d(in_channels=in_channels,
                                                        out_channels=self.spatial_group,
                                                        stride=1,
                                                        kernel_size=1,
                                                        groups=self.spatial_group,
                                                        padding=0,
                                                        bias=True)
            self.global_selection_conv_imag = nn.Conv2d(in_channels=in_channels,
                                                        out_channels=self.spatial_group,
                                                        stride=1,
                                                        kernel_size=1,
                                                        groups=self.spatial_group,
                                                        padding=0,
                                                        bias=True)
            if init == 'zero':
                self.global_selection_conv_real.weight.data.zero_()
                self.global_selection_conv_real.bias.data.zero_()
                self.global_selection_conv_imag.weight.data.zero_()
                self.global_selection_conv_imag.bias.data.zero_()

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        return freq_weight

    def forward(self, x, att_feat=None):
        """
        att_feat:feat for gen att
        """
        # freq_weight = self.freq_weight_conv(x)
        # self.sp_act(freq_weight)
        # if self.residual: x_residual = x.clone()
        if att_feat is None: att_feat = x
        x_list = []

        if self.lp_type == 'freq':
            pre_x = x.clone()
            b, _, h, w = x.shape
            # b, _c, h, w = freq_weight.shape
            # freq_weight = freq_weight.reshape(b, self.spatial_group, -1, h, w)
            x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))
            if self.global_selection:
                # global_att_real = self.global_selection_conv_real(x_fft.real)
                # global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, h, w)
                # global_att_imag = self.global_selection_conv_imag(x_fft.imag)
                # global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, h, w)
                # x_fft = x_fft.reshape(b, self.spatial_group, -1, h, w)
                # x_fft.real *= global_att_real
                # x_fft.imag *= global_att_imag
                # x_fft = x_fft.reshape(b, -1, h, w)
                # 将x_fft复数拆分成实部和虚部
                x_real = x_fft.real
                x_imag = x_fft.imag
                # 计算实部的全局注意力
                global_att_real = self.global_selection_conv_real(x_real)
                global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, h, w)
                # 计算虚部的全局注意力
                global_att_imag = self.global_selection_conv_imag(x_imag)
                global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, h, w)
                # 重塑x_fft为形状为(b, self.spatial_group, -1, h, w)的张量
                x_real = x_real.reshape(b, self.spatial_group, -1, h, w)
                x_imag = x_imag.reshape(b, self.spatial_group, -1, h, w)
                # 分别应用实部和虚部的全局注意力
                x_fft_real_updated = x_real * global_att_real
                x_fft_imag_updated = x_imag * global_att_imag
                # 合并为复数
                x_fft_updated = torch.complex(x_fft_real_updated, x_fft_imag_updated)
                # 重塑x_fft为形状为(b, -1, h, w)的张量
                x_fft = x_fft_updated.reshape(b, -1, h, w)

            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                mask[:, :, round(h / 2 - h / (2 * freq)):round(h / 2 + h / (2 * freq)),
                round(w / 2 - w / (2 * freq)):round(w / 2 + w / (2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask), norm='ortho').real
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group,
                                                                                               -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h,
                                                                                           w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pre_x)
        x = sum(x_list)
        x = self.bn(self.norm(x))
        x = torch.sigmoid(x) * (self.gb_max - 1)

        return x



# FreqPKC: Adaptive PeakConv (frequency-selective, learning-based guard bandwidth)
class FreqPeakConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, bias, refer_band, signal_type, gb_max=3,
                 fs_cfg={
                     'k_list': [2, 4, 8],
                     'fs_feat': 'feat',
                     'lowfreq_att': True,
                     'lp_type': 'freq',
                     # 'lp_type':'laplacian',
                     'act': 'sigmoid',
                     'spatial': 'conv',
                     'spatial_group': 1,
                     'global_selection': False,
                 },):
        """
        Args:
            in_ch: the input channel.
            out_ch: the output channel.
            bias: the bias for peak convolution.
            refer_band: the reference bandwidth.
            signal_type: str in ('range_doppler', 'angle_doppler', 'range_angle').
            gb_max: the upper bound of guard bandwidth in four directions.
        """
        super(FreqPeakConv2D, self).__init__()
        self.signal_type = signal_type
        self.in_ch = in_ch
        self.out_ch = out_ch
        # the initialized guard bandwidth, also the lower bound of guard bandwidth
        self.init_guard_band = (1, 1)
        # padding order: l-r-t-b
        self.padding = (self.init_guard_band[1] + refer_band,
                        self.init_guard_band[1] + refer_band,
                        self.init_guard_band[0] + refer_band,
                        self.init_guard_band[0] + refer_band)
        self.refer_band = refer_band
        # stride of peak conv only supports 1 now
        self.pc_strd = 1
        self.rep_padding = nn.ReplicationPad2d(self.padding)
        # the estimation network of guard bandwidth
        self.gb_estimator = FrequencySelection(self.in_ch, gb_max, **fs_cfg)
        
        # conv block for peak receptive field (PRF) in x
        self.kernel_size = (self.init_guard_band[0]+self.init_guard_band[1]+self.refer_band*2), 4
        self.stride = self.kernel_size
        self.bias = bias
        self.peak_conv = nn.Conv2d(self.in_ch, self.out_ch,
                                   kernel_size=self.kernel_size,
                                   padding=0,
                                   stride=self.stride,
                                   bias=self.bias)

    # update p_r with initialized p_r and estimated guard bandwidth offsets
    def get_pr_learned(self, pr, gb_mask):
        b, N, h, w = pr.size()
        gb_x_t = - gb_mask[:, 0, :, :].view(b, 1, h, w)
        gb_x_b = gb_mask[:, 1, :, :].view(b, 1, h, w)
        gb_y_l = - gb_mask[:, 2, :, :].view(b, 1, h, w)
        gb_y_r = gb_mask[:, 3, :, :].view(b, 1, h, w)
        zero = torch.zeros([int(b), int(N/8), int(h), int(w)], device=pr.device)
        move = torch.cat([gb_x_t.repeat(1, int(N/8), 1, 1), zero,
                          gb_x_b.repeat(1, int(N/8), 1, 1), zero,
                          zero, gb_y_r.repeat(1, int(N/8), 1, 1),
                          zero, gb_y_l.repeat(1, int(N/8), 1, 1)], 1)
        pr = pr + move
        return pr

    def forward(self, x):
        b, _, h, w = x.size()
        # kernel_size for pkc
        k_pkc = self.kernel_size
        # prf size
        N = k_pkc[0] * k_pkc[1]
        dtype = x.data.type()

        x_pad = self.rep_padding(x)
        # p_c denotes the center point positions
        p_c = self._get_p_c(b, h, w, N, dtype)
        # p_r denotes the initialized reference point positions
        # p_r shape: Bx2NxHxW
        p_r = self._get_p_r(N, p_c, dtype)
        # sample x_c (b,c,h,w,N) using p_c
        # (b, 2N, h, w) -> (b, h, w, 2N)
        p_c = p_c.contiguous().permute(0, 2, 3, 1).floor().long()
        x_c = self._sample_x(x_pad, p_c, N)

        # move initialized p_r with estimated gb_mask
        # estimate guard bandwidth offsets.
        # shape pf (b, 4, h, w)
        gb_mask = self.gb_estimator(x)
        # update p_r with offsets and initialized p_r
        # shape of (b, 2N, h, w)
        p = self.get_pr_learned(p_r, gb_mask)

        # put channel dimension at the last
        # shape of (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # let linear interpolation operate on q = [q_lt, q_rb] for each p
        # q in left/top position is the nearest integer number of p
        q_lt = p.detach().floor()
        # q in right/below position = 1 + q in left/top position
        q_rb = q_lt + 1

        # restrict q in the range of [(0, 0), (H-1, W-1)], the H and W is the shape of padded x.
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x_pad.size(2)-1), torch.clamp(q_lt[..., N:], 0, x_pad.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x_pad.size(2)-1), torch.clamp(q_rb[..., N:], 0, x_pad.size(3)-1)], dim=-1).long()

        # restrict p in the range of [(0, 0), (H-1, W-1)]
        p = torch.cat([torch.clamp(p[..., :N], 0, x_pad.size(2)-1), torch.clamp(p[..., N:], 0, x_pad.size(3)-1)], dim=-1)

        # using q to get linear kernel weights for p
        # shape of (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))

        # getting x(q) from input x
        # shape of (b, c, h, w, N)
        x_q_lt = self._sample_x(x_pad, q_lt, N)
        x_q_rb = self._sample_x(x_pad, q_rb, N)

        # weighted summation of linear kernel interpolation
        # (b, c, h, w, N)
        x_r = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb

        # getting x_prf for peak convolution
        x_prf = x_c - x_r
        x_prf = self._reshape_x_prf(k_pkc[0], k_pkc[1], x_prf)

        # peak convolution
        out = self.peak_conv(x_prf)
        return out

    # getting p_c (the center point coords) from the padded grid of input x
    def _get_p_c(self, b, h, w, N, dtype):
        # generating pc_grid
        p_c_x, p_c_y = torch.meshgrid(
            torch.arange(self.padding[2], h * self.pc_strd + self.padding[2], self.pc_strd),
            torch.arange(self.padding[0], w * self.pc_strd + self.padding[0], self.pc_strd))
        p_c_x = torch.flatten(p_c_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_c_y = torch.flatten(p_c_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_c: 1x2NxHxW
        p_c = torch.cat([p_c_x, p_c_y], 1).type(dtype)
        # (b,2N,h,w)
        p_c = p_c.repeat(b, 1, 1, 1)
        return p_c

    # generating peak receptive field grid
    def _gen_prf_grid(self, rb, gb, N, dtype):
        # h for row (x); w for col (y)
        h_t = -(rb + gb[0])
        h_d = rb + gb[0]
        w_l = -(rb + gb[1])
        w_r = rb + gb[1]
        # width and height of receptive field
        w_prf = (rb + gb[1]) * 2 + 1
        h_prf = (rb + gb[0]) * 2 + 1

        prf_x_idx, prf_y_idx = torch.meshgrid(
            torch.arange(h_t, h_d + 1),
            torch.arange(w_l, w_r + 1))

        # taking positions clockwise 
        prf_xt = prf_x_idx[0:rb, 0:(w_prf - rb)]
        prf_xr = prf_x_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        prf_xd = prf_x_idx[(h_prf - rb):h_prf, rb:w_prf]
        prf_xl = prf_x_idx[rb:h_prf, 0:rb]

        prf_x = torch.cat([torch.flatten(prf_xt),
                           torch.flatten(prf_xr),
                           torch.flatten(prf_xd),
                           torch.flatten(prf_xl)], 0)

        prf_yt = prf_y_idx[0:rb, 0:(w_prf - rb)]
        prf_yr = prf_y_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        prf_yd = prf_y_idx[(h_prf - rb):h_prf, rb:w_prf]
        prf_yl = prf_y_idx[rb:h_prf, 0:rb]

        prf_y = torch.cat([torch.flatten(prf_yt),
                           torch.flatten(prf_yr),
                           torch.flatten(prf_yd),
                           torch.flatten(prf_yl)], 0)

        prf = torch.cat([prf_x, prf_y], 0)
        prf = prf.view(1, 2 * N, 1, 1).type(dtype)
        return prf

    # getting p_r positions from each p_c
    def _get_p_r(self, N, p_c, dtype):
        # (1, 2N, 1, 1)
        prf = self._gen_prf_grid(self.refer_band, self.init_guard_band, N, dtype)
        # (B, 2N, h, w)
        p_r = p_c + prf
        return p_r

    # sampling x using p_r or p_c
    def _sample_x(self, x_pad, p, N):
        b, h, w, _ = p.size()
        # x_pad: shape of (b, c, h_pad, w_pad)
        h_pad = x_pad.size(2)
        w_pad = x_pad.size(3)
        c = x_pad.size(1)
        # strech each spatial channel of x_pad as 1-D vector
        x_pad = x_pad.contiguous().view(b, c, -1)
        # transform spatial coord of p into the 1-D index
        index = p[..., :N] * w_pad + p[..., N:]
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_r = x_pad.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_r

    @staticmethod
    # reshape the x_prf
    def _reshape_x_prf(k_h, k_w, x_prf):
        b, c, h, w, N = x_prf.size()
        x_prf = torch.cat([x_prf[..., s:s + k_w].contiguous().view(b, c, h, w * k_w) for s in range(0, N, k_w)], dim=-1)
        x_prf = x_prf.contiguous().view(b, c, h * k_h, w * k_w)
        return x_prf

# Double Adaptive PeakConv 2D Block
class DoubleAdaPKC2D(nn.Module):
    """ (2D AdaPKC => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, bias, rb, sig, gb_max=3):
        super(DoubleAdaPKC2D, self).__init__()
        self.bias = bias
        self.refer_band = rb
        self.gb_max = gb_max
        self.signal_type = sig
        self.pk_conv1 = FreqPeakConv2D(in_ch, out_ch, bias=self.bias,
                                      refer_band=self.refer_band,
                                      signal_type=self.signal_type,
                                      gb_max=self.gb_max)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.pk_conv2 = FreqPeakConv2D(out_ch, out_ch, bias=self.bias,
                                      refer_band=self.refer_band,
                                      signal_type=self.signal_type,
                                      gb_max=self.gb_max)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.pk_conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pk_conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256, dataset_type='RS-SAR'):
        super().__init__()
        self.dataset_type = dataset_type
        # For RSSAR dataset
        if self.dataset_type == 'RS-SAR':
            self.global_avg_pool = nn.AvgPool2d((72, 72))
        # For KuRALS_PD dataset
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
        # For KuRALS_PD dataset
        else:
            raise KeyError("we only support RSSAR dataset for now, please set dataset_type='RS-SAR'")
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type, n_frames, dataset_type='RS-SAR'):
        super().__init__()
        self.signal_type = signal_type
        self.dataset_type = dataset_type
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=32, k_size=(1+n_frames//2, 3, 3),
                                                      pad=(0, 1, 1), dil=1)
        # self.doppler_max_pool = nn.MaxPool2d(2, stride=(1, 2))
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.double_pkc_block = DoubleAdaPKC2D(in_ch=32, out_ch=64,
                                                 bias=True,
                                                 rb=1,
                                                 sig=self.signal_type,
                                                 gb_max=3)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=64, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        # For RSSAR dataset
        if self.dataset_type == 'RS-SAR':
            x1_pad = F.pad(x1, (0, 0, 0, 0), "constant", 0)
            x1_down = self.max_pool(x1_pad)
        # For KuRALS_PD dataset
        else:
            raise KeyError("we only support RSSAR dataset for now, please set dataset_type='RS-SAR'")
        

        x2 = self.double_pkc_block(x1_down)
        x2_pad = F.pad(x2, (0, 0, 0, 1), "constant", 0)
        x2_down = self.max_pool(x2_pad)

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x2_down, x3


class SVNet_FreqPKC(nn.Module):
    """
    RSSAR with FreqPKC module

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames, dataset_type='RS-SAR'):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', n_frames, dataset_type)

        # ASPP Blocks
        self.rd_aspp_block = ASPPBlock(in_ch=64, out_ch=128, dataset_type=dataset_type)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Range-Doppler (RD) decoding branch
        # For RSSAR dataset
        if dataset_type == 'RS-SAR':
            self.rd_upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # For KuRALS_PD dataset
        else:
            raise KeyError("we only support RSSAR dataset for now, please set dataset_type='RS-SAR'")
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        
        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x_rd):
        # Backbone
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)

        # ASPP blocks
        x2_rd = self.rd_aspp_block(rd_features)
        x2_rd = self.rd_single_conv_block1_1x1(x2_rd)

        # Latent Space + ASPP features
        x4_rd = torch.cat((x2_rd, rd_latent), 1)

        # Decoding branch with upconvs
        x4_rd = self.rd_upconv1(x4_rd)
        x4_rd = self.rd_double_conv_block1(x4_rd)

        x4_rd = self.rd_upconv2(x4_rd)
        x4_rd = self.rd_double_conv_block2(x4_rd)
        
        # Final 1D convolutions
        x4_rd = self.rd_final(x4_rd)

        return x4_rd