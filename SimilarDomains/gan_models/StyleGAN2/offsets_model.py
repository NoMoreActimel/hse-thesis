import math
import random
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .op import FusedLeakyReLU

from gan_models.StyleGAN2.model import (
    PixelNorm, EqualLinear, Blur, ModulatedConv2d,
    NoiseInjection, ConstantInput, ToRGB
)
from gan_models.StyleGAN2.featureshift_conv import FeatureiResnet

from core.stylegan_patches import modulation_patches, decomposition_patches, style_patches


class ModModulatedConv2d(nn.Module):
    def __init__(
        self,
        patch_key,
        conv_to_patch
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = conv_to_patch.kernel_size
        self.in_channel = conv_to_patch.in_channel
        self.out_channel = conv_to_patch.out_channel
        self.upsample = conv_to_patch.upsample
        self.downsample = conv_to_patch.downsample

        if self.upsample or self.downsample:
            self.blur = conv_to_patch.blur

        self.scale = conv_to_patch.scale
        self.padding = conv_to_patch.padding
        self.modulation = conv_to_patch.modulation
        self.demodulate = conv_to_patch.demodulate
        self.weight = conv_to_patch.weight

        self.offsets_modulation = modulation_patches[patch_key](
            conv_to_patch.weight
        )
        # self.matrix_parametrizator.matrix_decomposition(conv_to_patch.weight)

    def forward(self, input, style, offsets=None, is_s_code=False, offset_power=1.):
        batch, in_channel, height, width = input.shape

        if not is_s_code:
            style = self.modulation(style)
        style = style.view(batch, 1, in_channel, 1, 1)
        
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        if offsets is not None:
            weight = self.offsets_modulation(weight, offsets, offset_power)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class DecModulatedConv2d(nn.Module):
    def __init__(
        self,
        patch_key,
        conv_to_patch,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = conv_to_patch.kernel_size
        self.in_channel = conv_to_patch.in_channel
        self.out_channel = conv_to_patch.out_channel
        self.upsample = conv_to_patch.upsample
        self.downsample = conv_to_patch.downsample

        if self.upsample or self.downsample:
            self.blur = conv_to_patch.blur

        self.scale = conv_to_patch.scale
        self.padding = conv_to_patch.padding
        self.modulation = conv_to_patch.modulation
        self.demodulate = conv_to_patch.demodulate
        self.matrix_parametrizator = decomposition_patches[patch_key](
            conv_to_patch.weight
        )

    def forward(self, input, style, offsets=None, is_s_code=False, offset_power=1.):
        batch, in_channel, height, width = input.shape
        weight = self.matrix_parametrizator(offsets)
        
        if not is_s_code:
            style = self.modulation(style)

        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class StyleModulatedConv2d(nn.Module):
    def __init__(
            self,
            patch_key,
            conv_to_patch
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = conv_to_patch.kernel_size
        self.in_channel = conv_to_patch.in_channel
        self.out_channel = conv_to_patch.out_channel
        self.upsample = conv_to_patch.upsample
        self.downsample = conv_to_patch.downsample

        if self.upsample or self.downsample:
            self.blur = conv_to_patch.blur

        self.scale = conv_to_patch.scale
        self.padding = conv_to_patch.padding
        self.modulation = conv_to_patch.modulation
        self.demodulate = conv_to_patch.demodulate
        self.weight = conv_to_patch.weight

        self.offsets_modulation = style_patches[patch_key](
            conv_to_patch.weight
        )

    def forward(self, input, style, offsets=None, is_s_code=False, offset_power=1.):
        batch, in_channel, height, width = input.shape
        
        if self.offsets_modulation.style_space() == 'w' and offsets is not None:
            style = self.offsets_modulation(style, offsets)
        
        if not is_s_code:
            style = self.modulation(style) # style_space = modulation(w+) == affine(w+)
        
        if self.offsets_modulation.style_space() == 's' and offsets is not None:
            style = self.offsets_modulation(style, offsets, offset_power)
        
        # further code is from original ModulatedConv2d
        style = style.view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class OffsetsStyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, offsets, noise=None, is_s_code=False, offset_power=1.):
        out = self.conv(input=input, style=style, offsets=offsets, is_s_code=is_s_code, offset_power=offset_power)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out
    

class OffsetsGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        fse_idx_k=5,
        use_feature_shift_conv=False,
        feature_shift_conv_res=16,
        feature_shift_layer_idx=5
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = OffsetsStyledConv(
            self.channels[4],
            self.channels[4],
            3, style_dim,
            blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer('noise_{}'.format(layer_idx), torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                OffsetsStyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                OffsetsStyledConv(
                    out_channel,
                    out_channel,
                    3,
                    style_dim,
                    blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel
        
        # You can ommit this field if features_in will be a list
        # For the tensor features_in support fill this in
        self.fse_idx_k = fse_idx_k

        self.feature_shift_conv = None
        self.use_feature_shift_conv = use_feature_shift_conv
        self.feature_shift_conv_res = feature_shift_conv_res
        self.feature_shift_layer_idx = feature_shift_layer_idx
        if use_feature_shift_conv:
            # self.feature_shift_conv = OffsetsFeatureShiftConv(
            #     input_channels=self.channels[feature_shift_conv_res],
            #     output_channels=self.channels[feature_shift_conv_res],
            #     kernel_size=1
            # )
            ch = self.channels[feature_shift_conv_res]
            self.feature_shift_conv = FeatureiResnet(
                blocks=[[2 * ch, 2], [(3 * ch) // 2, 2], [ch, 2]],
                inplanes=2 * ch  # 2 * ch because input is concatenated with delta
            )

        self.n_latent = self.log_size * 2 - 2
        
        self.modulation_layers = [self.conv1.conv.modulation, self.to_rgb1.conv.modulation] + \
                                 [layer.conv.modulation for layer in self.convs]            + \
                                 [layer.conv.modulation for layer in self.to_rgbs]
        
    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)
    
    def get_s_code(self, styles, input_is_latent, truncation_latent, truncation, inject_index=None):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]
            print('meh-styles', styles[0].shape)
        
        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t
        
        if isinstance(styles, torch.Tensor):
            latent = styles
        elif len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                print('meh inject', latent.shape)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        
        s_codes = [
            self.conv1.conv.modulation(latent[:, 0]),
            self.to_rgb1.conv.modulation(latent[:, 1])
        ]
        
        i = 1

        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            s_codes.append(conv1.conv.modulation(latent[:, i]))
            s_codes.append(conv2.conv.modulation(latent[:, i + 1]))
            s_codes.append(to_rgb.conv.modulation(latent[:, i + 2]))
            i += 2

        return s_codes

    def get_conv_output(self, conv, input1, input2, latent, offset, noise, offset_power, use_without_offset=True):
        """
        Computes outputs with or without offsets using flag,
        Could be used for Inference with Feature-Style-Encoder, 
        shifting feature maps towards the same direction as the original generator
        Refer to Formula 1 at https://arxiv.org/pdf/2202.02183.pdf
        """
        output1 = conv(input1, latent, offset, noise=noise, offset_power=offset_power)
        output2 = conv(input2, latent, None, noise=noise, offset_power=offset_power) if use_without_offset else None
        return output1, output2

    def forward(
        self,
        styles,
        offsets=None,
        return_latents=False,
        return_features=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        is_s_code=False,
        noise=None,
        randomize_noise=False,
        offset_power=1.,
        features_in=None,
        shift_with_generator_feature_map=False,
        feature_scale=1.0,
        feature_shift_delta=None,
        return_feature_shift_output=False
    ):
        if not is_s_code:
            return self.forward_with_w(
                styles, 
                offsets,
                return_latents, 
                return_features,
                inject_index, 
                truncation, 
                truncation_latent, 
                input_is_latent, 
                noise, 
                randomize_noise,
                offset_power,
                features_in,
                shift_with_generator_feature_map,
                feature_scale,
                feature_shift_delta,
                return_feature_shift_output
            )
        
        return self.forward_with_s(styles, offsets, offset_power, return_latents, noise, randomize_noise)

    def forward_with_w(
        self,
        styles,
        offsets,
        return_latents=False,
        return_features=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        offset_power=1.,
        features_in=None,
        shift_with_generator_feature_map=False,
        feature_scale=1.0,
        feature_shift_delta=None,
        return_feature_shift_output=False
    ):
        # if features_in is not None:
        #     print('features_in shapes:', *[f.shape if f is not None else f for f in features_in])
        
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, 'noise_{}'.format(i)) for i in range(self.num_layers)]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        
        print("latent.shape:", latent.shape)

        if self.use_feature_shift_conv: assert feature_shift_delta is not None, "Provide feature_shift_delta!"

        def _insert_feature(input, input_wo_offset, features):
            output = (1 - feature_scale) * input + feature_scale * features.type_as(input)
            if shift_with_generator_feature_map:
                output = output + input - input_wo_offset
                print(torch.norm(output), torch.norm(input - input_wo_offset))
            return output, False # stop calculating original feature maps without offsets

        def insert_feature(input, input_wo_offset, layer_idx):
            # print(f'In insert feature, input.shape: {input.shape}, layer_idx: {layer_idx}')
            if features_in is not None:
                if isinstance(features_in, torch.Tensor):
                    if layer_idx == self.fse_idx_k:
                        return _insert_feature(input, input_wo_offset, features_in)
                elif isinstance(features_in[0], list):
                    if features_in[0][layer_idx] is not None:
                        output = torch.cat([
                            _insert_feature(input[ind], input_wo_offset[ind], f[layer_idx])[0]
                            for ind, f in enumerate(features_in)
                        ])
                        assert input.shape == output.shape, \
                            f"batched insert_feature on list features_in error: {input.shape} -> {output.shape}"
                        return output, False  # stop calculating original feature maps without offsets
                elif features_in[layer_idx] is not None:
                    print(layer_idx, input.shape, features_in[layer_idx].shape)
                    return _insert_feature(input, input_wo_offset, features_in[layer_idx])
            return input, shift_with_generator_feature_map
        
        def shift_feature(input, input_wo_offset, layer_idx):
            # print(f'In shift feature, input.shape: {input.shape}, layer_idx: {layer_idx}')
            if self.use_feature_shift_conv:
                if self.feature_shift_layer_idx == layer_idx:
                    print(f"Norm before feature_shift: {torch.norm(input)}")
                    assert self.feature_shift_conv_res == input.shape[-2] and input.shape[-2] == input.shape[-1], \
                        f"Shape mismatch for feature_shift with res {self.feature_shift_conv_res}, input.shape: {input.shape}"
                    assert feature_shift_delta.shape == input.shape, \
                        f"Shape mismatch for feature_shift with input.shape: {input.shape} and feature_shift_delta.shape: {feature_shift_delta}"
                    shift_input = torch.cat([input, feature_shift_delta], dim=1)
                    print(f'Shapes in feature_shift, input: {input.shape}, delta: {feature_shift_delta.shape}, shift_input: {shift_input.shape}')

                    output = self.feature_shift_conv(shift_input)
                    print(f'output: {output.shape}')
                    print(f"Norm after feature_shift: {torch.norm(output)}")

                    if return_feature_shift_output:
                        feature_shift_out.append(output)
                    
                    return output, input_wo_offset
            return input, input_wo_offset
        
        feature_shift_out = []
        outs = []
        out = self.input(latent)
        if return_features: outs.append(out)

        out, out_wo_offsets = self.get_conv_output(
            self.conv1, out, out, latent[:, 0],
            offsets['conv_0'] if offsets is not None else None,
            noise=noise[0], offset_power=offset_power,
            use_without_offset=shift_with_generator_feature_map
        )
        if return_features: outs.append(out)

        skip = self.to_rgb1(out, latent[:, 1])
        i = 1

        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            # print(f'{i}-th layer shape: {out.shape}')
            out, shift_with_generator_feature_map = insert_feature(out, out_wo_offsets, i)
            out, out_wo_offsets = shift_feature(out, out_wo_offsets, i)
            out, out_wo_offsets = self.get_conv_output(
                conv1, out, out_wo_offsets, latent[:, i],
                offsets['conv_{}'.format(i)] if offsets is not None else None,
                noise=noise1, offset_power=offset_power,
                use_without_offset=shift_with_generator_feature_map
            )
            if return_features: outs.append(out)
            
            # print(f'{i+1}-th layer shape: {out.shape}')
            out, shift_with_generator_feature_map = insert_feature(out, out_wo_offsets, i + 1)
            out, out_wo_offsets = shift_feature(out, out_wo_offsets, i + 1)
            out, out_wo_offsets = self.get_conv_output(
                conv2, out, out_wo_offsets, latent[:, i + 1],
                offsets['conv_{}'.format(i + 1)] if offsets is not None else None,
                noise=noise2, offset_power=offset_power,
                use_without_offset=shift_with_generator_feature_map
            )
            if return_features: outs.append(out)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_features:
            return image, outs
        elif return_feature_shift_output:
            return image, feature_shift_out[0]
        else:
            return image, None
        
    def forward_with_s(
        self,
        latent,
        offsets,
        offset_power,
        return_latents=False,
        noise=None,
        randomize_noise=True,
    ):
        
        d1 = dict([(0, 0), (1, 2), (2, 3), (3, 5), (4, 6), (5, 8), (6, 9), (7, 11), (8, 12), (9, 14), (10, 15), (11, 17), (12, 18), (13, 20), (14, 21), (15, 23), (16, 24)])
        
        d2 = {v:k for k, v in d1.items()}
        
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        out = self.input(latent[0])        
        out = self.conv1(
            out, latent[0],
            offsets['conv_0'] if offsets is not None else None,
            noise=noise[0], is_s_code=True, offset_power=offset_power
        )
        
        skip = self.to_rgb1(out, latent[1], is_s_code=True)

        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            
            out = conv1(
                out, latent[i],
                offsets['conv_{}'.format(d2[i])] if offsets is not None else None,
                noise=noise1, is_s_code=True, offset_power=offset_power
            )
            
            out = conv2(
                out, latent[i + 1],
                offsets['conv_{}'.format(d2[i + 1])] if offsets is not None else None,
                noise=noise2, is_s_code=True, offset_power=offset_power
            )
            
            skip = to_rgb(out, latent[i + 2], skip, is_s_code=True)
            
            i += 3

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None