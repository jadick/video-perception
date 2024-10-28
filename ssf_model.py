import math

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

from torch.cuda import amp

#from compressai.entropy_models import EntropyBottleneck, GaussianConditional
#from compressai.layers import QReLU
#from compressai.ops import quantize_ste
#from compressai.registry import register_model

def conv(in_channels, out_channels, kernel_size=5, stride=2, padding=True):
    if padding:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
    else:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride
        )


def deconv(in_channels, out_channels, kernel_size=5, stride=2, padding=True):
    if padding:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1,
            padding=kernel_size // 2,
        )
    else:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1
        )
    
def deconv_v2(in_channels, out_channels, kernel_size=5, stride=2, padding=True):

    if padding:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1,
            padding=kernel_size // 2,
        )
    else:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1
        )

def gaussian_kernel1d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()

def gaussian_kernel2d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])

def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError("Missing kernel_size or sigma parameters")
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)

    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    x = torch.nn.functional.conv2d(
        x,
        kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)),
        groups=x.size(1),
    )
    return x

def meshgrid2d(N: int, C: int, H: int, W: int, device: torch.device):
    """Create a 2D meshgrid for interpolation."""
    theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(N, 2, 3)
    return F.affine_grid(theta, (N, C, H, W), align_corners=False)

    
class ScaleSpaceFlow(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        self.img_encoder = Encoder(3)
        self.img_decoder = Decoder(3)
        #self.img_hyperprior = Hyperprior()

        self.res_encoder = Encoder(6)
        self.res_decoder = Decoder(3, in_planes=384)
        #self.res_hyperprior = Hyperprior()
        
        self.P_encoder = Encoder(3)

        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1)
        #self.motion_hyperprior = Hyperprior()

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift
        self.tanh = nn.Tanh()
    
    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []
        frames_likelihoods = []

        #x_hat, likelihoods = self.forward_keyframe(frames[0])
        x_hat = frames[0]
        reconstructions.append(x_hat)
        
        #frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        for i in range(1, 2):
            x = frames[i]
            x_ref = self.forward_inter(x, x_ref)

        return x_ref

    def forward_inter(self, x_cur, x_ref):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion_hat = self.motion_encoder(x)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = torch.cat((x_cur, x_pred), dim=1)
        y_res_hat = self.res_encoder(x_res)
        #y_res_hat, res_likelihoods = self.res_hyperprior(y_res)
        
        # Condition
        y_cond = self.P_encoder(x_ref)
        y_combine = torch.cat((y_res_hat, y_cond), dim=1)
        x_rec = self.res_decoder(y_combine)
        x_rec = self.tanh(x_rec)
        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    @amp.autocast(enabled=False)
    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

    def compress(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        frame_strings = []
        shape_infos = []

        x_ref, out_keyframe = self.encode_keyframe(frames[0])

        frame_strings.append(out_keyframe["strings"])
        shape_infos.append(out_keyframe["shape"])

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref)

            frame_strings.append(out_interframe["strings"])
            shape_infos.append(out_interframe["shape"])

        return frame_strings, shape_infos

    def decompress(self, strings, shapes):
        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f"Invalid number of frames: {len(strings)}.")

        assert len(strings) == len(
            shapes
        ), f"Number of information should match {len(strings)} != {len(shapes)}."

        dec_frames = []

        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)

        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape)
            dec_frames.append(x_ref)

        return dec_frames

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net
    
class ScaleSpaceFlow_gridless(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    #deconv_v2(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.Upsample(scale_factor = 2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    #deconv_v2(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.Upsample(scale_factor = 2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    #deconv_v2(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.Upsample(scale_factor = 2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    #deconv_v2(mid_planes, out_planes, kernel_size=5, stride=2),
                    nn.Upsample(scale_factor = 2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=0),
                )

        self.img_encoder = Encoder(3)
        self.img_decoder = Decoder(3)
        #self.img_hyperprior = Hyperprior()

        self.res_encoder = Encoder(6)
        self.res_decoder = Decoder(3, in_planes=384)
        #self.res_hyperprior = Hyperprior()
        
        self.P_encoder = Encoder(3)

        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1)
        #self.motion_hyperprior = Hyperprior()

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift
        self.tanh = nn.Tanh()
    
    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []
        frames_likelihoods = []

        #x_hat, likelihoods = self.forward_keyframe(frames[0])
        x_hat = frames[0]
        reconstructions.append(x_hat)
        
        #frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        for i in range(1, 2):
            x = frames[i]
            x_ref = self.forward_inter(x, x_ref)

        return x_ref

    def forward_inter(self, x_cur, x_ref):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion_hat = self.motion_encoder(x)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = torch.cat((x_cur, x_pred), dim=1)
        y_res_hat = self.res_encoder(x_res)
        #y_res_hat, res_likelihoods = self.res_hyperprior(y_res)
        
        # Condition
        y_cond = self.P_encoder(x_ref)
        y_combine = torch.cat((y_res_hat, y_cond), dim=1)
        x_rec = self.res_decoder(y_combine)
        x_rec = self.tanh(x_rec)
        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    @amp.autocast(enabled=False)
    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

    def compress(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        frame_strings = []
        shape_infos = []

        x_ref, out_keyframe = self.encode_keyframe(frames[0])

        frame_strings.append(out_keyframe["strings"])
        shape_infos.append(out_keyframe["shape"])

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref)

            frame_strings.append(out_interframe["strings"])
            shape_infos.append(out_interframe["shape"])

        return frame_strings, shape_infos

    def decompress(self, strings, shapes):
        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f"Invalid number of frames: {len(strings)}.")

        assert len(strings) == len(
            shapes
        ), f"Number of information should match {len(strings)} != {len(shapes)}."

        dec_frames = []

        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)

        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape)
            dec_frames.append(x_ref)

        return dec_frames

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net
    
    
class ScaleSpaceFlow_quantized(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        stochastic = True,
        quantize_latents = True,
        L=4, q_limits=(-4.0, 4.0),freeze_enc=False,
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 24
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 24, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )
        
        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits
        self.freeze_enc= freeze_enc
        
        self.img_encoder = Encoder(3)
        self.img_decoder = Decoder(3)
        #self.img_hyperprior = Hyperprior()

        self.res_encoder = Encoder(6)
        self.res_decoder = Decoder(3, in_planes=48)
        #self.res_hyperprior = Hyperprior()
        
        self.P_encoder = Encoder(3)

        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1)
        #self.motion_hyperprior = Hyperprior()

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift
        self.tanh = nn.Tanh()
        
        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)
        
    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y
    
    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []
        frames_likelihoods = []

        #x_hat, likelihoods = self.forward_keyframe(frames[0])
        x_hat = frames[0]
        reconstructions.append(x_hat)
        
        #frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        for i in range(1, 2):
            x = frames[i]
            x_rec = self.forward_inter(x, x_ref)

        return x_rec

    def forward_inter(self, x_cur, x_ref):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        if self.freeze_enc:
            with torch.no_grad():
                y_motion_hat = self.motion_encoder(x)
        else:
            y_motion_hat = self.motion_encoder(x)
        y_motion_hat = self.quantize_noise(y_motion_hat)
        #print ("Motion dim: ", y_motion_hat.shape)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = torch.cat((x_cur, x_pred), dim=1)
        if self.freeze_enc:
            with torch.no_grad():
                y_res_hat = self.res_encoder(x_res)
                y_res_hat = self.quantize_noise(y_res_hat)
        else:
            y_res_hat = self.res_encoder(x_res)
            y_res_hat = self.quantize_noise(y_res_hat)
        #y_res_hat, res_likelihoods = self.res_hyperprior(y_res)
        #print ("Motion dim: ", y_res_hat.shape)
        
        # Condition
        y_cond = self.P_encoder(x_ref)
        y_combine = torch.cat((y_res_hat, y_cond), dim=1)
        x_rec = self.res_decoder(y_combine)
        x_rec = self.tanh(x_rec)
        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    @amp.autocast(enabled=False)
    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

    def compress(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        frame_strings = []
        shape_infos = []

        x_ref, out_keyframe = self.encode_keyframe(frames[0])

        frame_strings.append(out_keyframe["strings"])
        shape_infos.append(out_keyframe["shape"])

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref)

            frame_strings.append(out_interframe["strings"])
            shape_infos.append(out_interframe["shape"])

        return frame_strings, shape_infos

    def decompress(self, strings, shapes):
        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f"Invalid number of frames: {len(strings)}.")

        assert len(strings) == len(
            shapes
        ), f"Number of information should match {len(strings)} != {len(shapes)}."

        dec_frames = []

        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)

        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape)
            dec_frames.append(x_ref)

        return dec_frames

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net
    
class ScaleSpaceFlow_gridless_quantized(nn.Module): #not done yet
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    #deconv_v2(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.Upsample(scale_factor = 2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    #deconv_v2(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.Upsample(scale_factor = 2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    #deconv_v2(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.Upsample(scale_factor = 2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    #deconv_v2(mid_planes, out_planes, kernel_size=5, stride=2),
                    nn.Upsample(scale_factor = 2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=0),
                )

        self.img_encoder = Encoder(3)
        self.img_decoder = Decoder(3)
        #self.img_hyperprior = Hyperprior()

        self.res_encoder = Encoder(6)
        self.res_decoder = Decoder(3, in_planes=384)
        #self.res_hyperprior = Hyperprior()
        
        self.P_encoder = Encoder(3)

        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1)
        #self.motion_hyperprior = Hyperprior()

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift
        self.tanh = nn.Tanh()
    
    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []
        frames_likelihoods = []

        #x_hat, likelihoods = self.forward_keyframe(frames[0])
        x_hat = frames[0]
        reconstructions.append(x_hat)
        
        #frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        for i in range(1, 2):
            x = frames[i]
            x_ref = self.forward_inter(x, x_ref)

        return x_ref

    def forward_inter(self, x_cur, x_ref):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion_hat = self.motion_encoder(x)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = torch.cat((x_cur, x_pred), dim=1)
        y_res_hat = self.res_encoder(x_res)
        #y_res_hat, res_likelihoods = self.res_hyperprior(y_res)
        
        # Condition
        y_cond = self.P_encoder(x_ref)
        y_combine = torch.cat((y_res_hat, y_cond), dim=1)
        x_rec = self.res_decoder(y_combine)
        x_rec = self.tanh(x_rec)
        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    @amp.autocast(enabled=False)
    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

    def compress(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        frame_strings = []
        shape_infos = []

        x_ref, out_keyframe = self.encode_keyframe(frames[0])

        frame_strings.append(out_keyframe["strings"])
        shape_infos.append(out_keyframe["shape"])

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref)

            frame_strings.append(out_interframe["strings"])
            shape_infos.append(out_interframe["shape"])

        return frame_strings, shape_infos

    def decompress(self, strings, shapes):
        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f"Invalid number of frames: {len(strings)}.")

        assert len(strings) == len(
            shapes
        ), f"Number of information should match {len(strings)} != {len(shapes)}."

        dec_frames = []

        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)

        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape)
            dec_frames.append(x_ref)

        return dec_frames

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net