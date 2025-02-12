import torch
import torch.nn as nn

from core.utils.common import requires_grad
from core.utils.class_registry import ClassRegistry

from gan_models.StyleGAN2.offsets_model import (
    OffsetsGenerator,
    ModModulatedConv2d,
    DecModulatedConv2d,
    StyleModulatedConv2d
)

from core.stylegan_patches import decomposition_patches, modulation_patches, style_patches

uda_models = ClassRegistry()

# default_arguments = Omegaconf.structured(uda_models.make_dataclass_from_args("GenArgs"))
# default_arguments.GenArgs.stylegan2.size ...


@uda_models.add_to_registry("stylegan2")
class OffsetsTunningGenerator(torch.nn.Module):
    def __init__(self, img_size=1024, latent_size=512, map_layers=8, channel_multiplier=2,
                 fse_idx_k=5, use_feature_shift_conv=False, feature_shift_conv_res=16, feature_shift_layer_idx=5,
                 device='cuda:0', checkpoint_path=None):
        super().__init__()
        
        self.generator = OffsetsGenerator(
            img_size, latent_size, map_layers, channel_multiplier=channel_multiplier, fse_idx_k=fse_idx_k,
            use_feature_shift_conv=use_feature_shift_conv,
            feature_shift_conv_res=feature_shift_conv_res,
            feature_shift_layer_idx=feature_shift_layer_idx
        ).to(device)

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.generator.load_state_dict(checkpoint["g_ema"], strict=False)
            if use_feature_shift_conv:
                self.init_feature_shift_weights()

        self.generator.eval()

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096).to(device)
        self.patched = False
    
    def init_feature_shift_weights(self):
        for m in self.generator.feature_shift_conv.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def patch_layers(self, patch_key):
        """
        Modify ModulatedConv2d Layers with <<patch_key>> patch
        """
        if patch_key in decomposition_patches:
            self._patch_modconv_key(patch_key, DecModulatedConv2d)
        elif patch_key in modulation_patches:
            self._patch_modconv_key(patch_key, ModModulatedConv2d)
        elif patch_key in style_patches:
            self._patch_modconv_key(patch_key, StyleModulatedConv2d)
        elif patch_key == 'original':
            ...
        else:
            raise ValueError(
                f'''
                Incorrect patch_key. Got {patch_key}, possible are {
                {decomposition_patches}, {modulation_patches}, {style_patches}
                }
                '''
            )
        self.patched = True
        
        return self
    
    def _patch_modconv_key(self, patch_key, mod_conv_class):
        self.generator.conv1.conv = mod_conv_class(
            patch_key, self.generator.conv1.conv
        )

        for conv_layer_ix in range(len(self.generator.convs)):
            self.generator.convs[conv_layer_ix].conv = mod_conv_class(
                patch_key, self.generator.convs[conv_layer_ix].conv
            )

    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):
        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])
        if phase == 'shape':
            # layers 1-2
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        if phase == 'no_fine':
            # const + layers 1-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        if phase == 'shape_expanded':
            # const + layers 1-3
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'mapping':
            return list(self.get_all_layers())[0]
        if phase == 'affine':
            styled_convs = list(self.get_all_layers())[4]
            return [s.conv.modulation for s in styled_convs] + [self.generator.conv1.conv.modulation]
        if phase == 'affine_all':
            synt_convs = self.get_all_layers()[4]
            to_rgbs = self.get_all_layers()[6]
            affine_convs = [s.conv.modulation for s in synt_convs] + [self.generator.conv1.conv.modulation]
            affine_to_rgbs = [s.conv.modulation for s in to_rgbs] + [self.generator.to_rgb1.conv.modulation]
            return affine_convs + affine_to_rgbs
        if phase == 'affine_to_rgb':
            styled_convs = self.get_all_layers()[4]
            affine_layers = [s.conv.modulation for s in styled_convs]
            return affine_layers + [self.get_all_layers()[6]] + [self.generator.to_rgb1]
        if phase == 'conv_kernel':
            styled_convs = list(self.get_all_layers())[4]
            return [s.conv.weight for s in styled_convs] + [self.generator.conv1.conv.weight]
        
        if phase == 'feature_shift':
            return [self.generator.feature_shift_conv]

        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers()
        else:
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])

    def freeze_layers(self, layer_list=None):
        """
        Disable training for all layers in list.
        """
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        """
        Enable training for all layers in list.
        """
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def style(self, styles):
        """
        Convert z codes to w codes.
        """
        styles = [self.generator.style(s) for s in styles]
        return styles

    def get_s_code(self, styles, input_is_latent=False, truncation=1):
        return self.generator.get_s_code(
            styles, input_is_latent,
            truncation=truncation,
            truncation_latent=self.mean_latent
        )

    def modulation_layers(self):
        return self.generator.modulation_layers

    def forward(self,
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
                randomize_noise=True,
                offset_power=1.,
                features_in=None,
                shift_with_generator_feature_map=False,
                feature_scale=1.0,
                feature_shift_delta=None,
                return_feature_shift_output=False,
                offsets_coeffs=None):
        
        if offsets is not None:
            assert self.patched, "call MODEL.patch_layers(`patch_key`) before using offsets"

        # ENHANCING DOMAIN OFFSETS ACCORDING TO LATENT DISTRIBUTIONS ANALYSIS
        if (offsets is not None) and (offsets_coeffs is not None):
            assert len(offsets_coeffs) == 18 and len(offsets) == 17
            for coeff, conv_name in zip(offsets_coeffs[1:], offsets.keys()):
                offsets[conv_name]['in'] = coeff * offsets[conv_name]['in']
            
        return self.generator(styles,
                              offsets=offsets,
                              return_latents=return_latents,
                              return_features=return_features,
                              truncation=truncation,
                              truncation_latent=self.mean_latent,
                              noise=noise,
                              randomize_noise=randomize_noise,
                              input_is_latent=input_is_latent,
                              is_s_code=is_s_code,
                              offset_power=offset_power,
                              features_in=features_in,
                              shift_with_generator_feature_map=shift_with_generator_feature_map,
                              feature_scale=feature_scale,
                              feature_shift_delta=feature_shift_delta,
                              return_feature_shift_output=return_feature_shift_output)
