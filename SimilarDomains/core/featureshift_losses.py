import os

import torch
from torch import nn
from torch.nn import functional as F

from torchvision import models
from torchvision import transforms as T

from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from itertools import chain
from typing import Sequence

from core.utils.image_utils import BicubicDownSample
# from models.face_parsing.model import BiSeNet, seg_mean, seg_std


normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

@dataclass
class DefaultPaths:
    psp_path: str = "/home/sasedov/HairFastGAN/pretrained_models/psp_ffhq_encode.pt"
    ir_se50_path: str = "/home/sasedov/HairFastGAN/pretrained_models/ArcFace/ir_se50.pth"
    stylegan_weights: str = "/home/sasedov/HairFastGAN/pretrained_models/stylegan2-ffhq-config-f.pt"
    stylegan_car_weights: str = "/home/sasedov/HairFastGAN/pretrained_models/stylegan2-car-config-f-new.pkl"
    stylegan_weights_pkl: str = (
        "/home/sasedov/HairFastGAN/pretrained_models/stylegan2-ffhq-config-f.pkl"
    )
    arcface_model_path: str = "/home/sasedov/HairFastGAN/pretrained_models/ArcFace/backbone_ir50.pth"
    moco: str = "/home/sasedov/HairFastGAN/pretrained_models/moco_v2_800ep_pretrain.pt"
    

"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    else:
        raise ValueError(
            "Invalid number of layers: {}. Must be one of [50, 100, 152]".format(
                num_layers
            )
        )
    return blocks


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


"""
Modified Backbone implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Backbone(nn.Module):
    def __init__(self, input_size, num_layers, mode="ir", drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )
        if input_size == 112:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Dropout(drop_ratio),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 512),
                nn.BatchNorm1d(512, affine=affine),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Dropout(drop_ratio),
                nn.Flatten(),
                nn.Linear(512 * 14 * 14, 512),
                nn.BatchNorm1d(512, affine=affine),
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, num_layers=50, mode="ir", drop_ratio=0.4, affine=False)
    return model


def IR_101(input_size):
    """Constructs a ir-101 model."""
    model = Backbone(
        input_size, num_layers=100, mode="ir", drop_ratio=0.4, affine=False
    )
    return model


def IR_152(input_size):
    """Constructs a ir-152 model."""
    model = Backbone(
        input_size, num_layers=152, mode="ir", drop_ratio=0.4, affine=False
    )
    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(
        input_size, num_layers=50, mode="ir_se", drop_ratio=0.4, affine=False
    )
    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(
        input_size, num_layers=100, mode="ir_se", drop_ratio=0.4, affine=False
    )
    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(
        input_size, num_layers=152, mode="ir_se", drop_ratio=0.4, affine=False
    )
    return model

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print("Loading ResNet ArcFace")
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        self.facenet.load_state_dict(torch.load(DefaultPaths.ir_se50_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count
    
class FeatReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, recon_1, recon_2):
        return self.loss_fn(recon_1, recon_2).mean()
    
class EncoderAdvLoss:
    def __call__(self, fake_preds):
        loss_G_adv = F.softplus(-fake_preds).mean()
        return loss_G_adv

class AdvLoss:
    def __init__(self, coef=0.0):
        self.coef = coef

    def __call__(self, disc, real_images, generated_images):
        fake_preds = disc(generated_images, None)
        real_preds = disc(real_images, None)
        loss = self.d_logistic_loss(real_preds, fake_preds)

        return {'disc adv': loss}

    def d_logistic_loss(self, real_preds, fake_preds):
        real_loss = F.softplus(-real_preds)
        fake_loss = F.softplus(fake_preds)

        return (real_loss.mean() + fake_loss.mean()) / 2


def get_network(net_type: str):
    if net_type == "alex":
        return AlexNet()
    elif net_type == "squeeze":
        return SqueezeNet()
    elif net_type == "vgg":
        return VGG16()
    else:
        raise NotImplementedError("choose net_type from [alex, squeeze, vgg].")


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__(
            [
                nn.Sequential(nn.Identity(), nn.Conv2d(nc, 1, 1, 1, 0, bias=False))
                for nc in n_channels_list
            ]
        )

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            "mean", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        if os.path.exists("/home/sasedov/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"):
            alexnet = models.alexnet()
            alexnet.load_state_dict(torch.load("/home/sasedov/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"))
            self.layers = alexnet.features
        else:
            self.layers = models.alexnet(True).features

        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(True).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = "alex", version: str = "0.1"):
    # build url
    url = (
        "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
        + f"master/lpips/weights/v{version}/{net_type}.pth"
    )

    pretrained_dir = "/home/sasedov/StyleDomain/SimilarDomains/pretrained/lpips"
    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url,
        model_dir=pretrained_dir,
        progress=True,
        map_location=None if torch.cuda.is_available() else torch.device("cpu")
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace("lin", "")
        new_key = new_key.replace("model.", "")
        new_state_dict[new_key] = val

    return new_state_dict
    

class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    def __init__(self, net_type: str = "alex", version: str = "0.1"):

        assert version in ["0.1"], "v0.1 is only supported now"

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type).to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0]


class LPIPSLoss(LPIPS):
    pass
    
class LPIPSScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = LPIPSLoss()

    def forward(self, x, y):
        out = 0
        for res in [256, 128, 64]:
            x_scale = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
            y_scale = F.interpolate(y, size=(res, res), mode="bilinear", align_corners=False)
            out += self.loss_fn.forward(x_scale, y_scale).mean()
        return out
    
class SyntMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, im1, im2):
        return self.loss_fn(im1, im2).mean()


class R1Loss:
    def __init__(self, coef=10.0):
        self.coef = coef

    def __call__(self, disc, real_images):
        real_images.requires_grad = True

        real_preds = disc(real_images, None)
        real_preds = real_preds.view(real_images.size(0), -1)
        real_preds = real_preds.mean(dim=1).unsqueeze(1)
        r1_loss = self.d_r1_loss(real_preds, real_images)

        loss_D_R1 = self.coef / 2 * r1_loss * 16 + 0 * real_preds[0]
        return {'disc r1 loss': loss_D_R1}

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty


class DilatedMask:
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
        
        cords_x = torch.arange(0, kernel_size).view(1, -1).expand(kernel_size, -1) - kernel_size // 2
        cords_y = cords_x.clone().permute(1, 0)
        self.kernel = torch.as_tensor((cords_x ** 2 + cords_y ** 2) <= (kernel_size // 2) ** 2, dtype=torch.float).view(1, 1, kernel_size, kernel_size).cuda()
        self.kernel /= self.kernel.sum()
    
    def __call__(self, mask):
        smooth_mask = F.conv2d(mask, self.kernel, padding=self.kernel_size // 2)
        return smooth_mask ** 0.25


class FeatureShiftLossBuilder:
    def __init__(self, losses_dict, device='cuda:0'):
        self.losses_dict = losses_dict
        self.device = device
        
        self.EncoderAdvLoss = EncoderAdvLoss()
        self.AdvLoss = AdvLoss()
        self.R1Loss = R1Loss()
        self.FeatReconLoss = FeatReconLoss().to(device).eval()
        self.IDLoss = IDLoss().to(device).eval()
        self.LPIPS = LPIPSScaleLoss().to(device).eval()
        self.SyntMSELoss = SyntMSELoss().to(device).eval()
        self.downsample_256 = BicubicDownSample(factor=4)
        
    def CalcAdvLoss(self, disc, gen_F):
        fake_preds_F = disc(gen_F, None)
        return self.EncoderAdvLoss(fake_preds_F)
    
    def CalcDisLoss(self, disc, real_images, generated_images):
        return self.AdvLoss(disc, real_images, generated_images)
    
    def CalcR1Loss(self, disc, real_images):
        return self.R1Loss(disc, real_images)
    
    def _sum_losses(self, losses, loss_type):
        losses[f"{loss_type}_loss"] = \
            self.losses_dict.get(f"{loss_type}_MSE", 1.0)   * losses[f"{loss_type}_MSE"] + \
            self.losses_dict.get(f"{loss_type}_LPIPS", 1.0) * losses[f"{loss_type}_LPIPS"] + \
            self.losses_dict.get(f"{loss_type}_ID", 1.0)    * losses[f"{loss_type}_ID"] + \
            self.losses_dict.get(f"{loss_type}_Adv", 1.0)   * losses[f"{loss_type}_Adv"]
        return losses

    def _invresion_losses(
            self, batch,
            source_disc,
            calc_inversion_adv_loss=True,
            calc_inversion_disc_loss=False,
            use_e4e_inversion_features_loss=False,
            use_fse_inversion_features_loss=False,
            **kwargs):
        
        source_image_256 = self.downsample_256(batch["images"])
        gen_source_inversion_256 = self.downsample_256(batch["feature_shift_inversion"])

        losses = {
            "inversion_MSE": self.SyntMSELoss(source_image_256, gen_source_inversion_256),
            "inversion_LPIPS": self.LPIPS(source_image_256, gen_source_inversion_256),
            "inversion_ID": self.IDLoss(source_image_256, gen_source_inversion_256),
            "inversion_Adv": self.CalcAdvLoss(source_disc, batch["feature_shift_inversion"]) if calc_inversion_adv_loss else 0.0
        }
        
        if calc_inversion_disc_loss:
            losses["source_disc_loss"] = self.CalcDisLoss(source_disc, batch["images"], batch["feature_shift_inversion"])

        losses = self._sum_losses(losses, "inversion")
        
        assert (not use_e4e_inversion_features_loss) or (not use_fse_inversion_features_loss)

        if use_e4e_inversion_features_loss:
            losses["inversion_features_MSE"] = self.SyntMSELoss(batch["e4e_inversion_features"], batch["feature_shift_inversion_output"])
            losses["inversion_loss"] = losses["inversion_loss"] + \
                self.losses_dict.get(f"inversion_features_MSE", 1.0) * losses["inversion_features_MSE"]
        
        if use_fse_inversion_features_loss:
            losses["inversion_features_MSE"] = self.SyntMSELoss(batch["fse_features"], batch["feature_shift_inversion_output"])
            losses["inversion_loss"] = losses["inversion_loss"] + \
                self.losses_dict.get(f"inversion_features_MSE", 1.0) * losses["inversion_features_MSE"]
        
        if self.losses_dict.get("inversion_features_reg", None) is not None:
            losses["inversion_features_reg"] = batch["feature_shift_inversion_output"].norm(p=2)
            losses["inversion_loss"] = losses["inversion_loss"] + \
                self.losses_dict.get(f"inversion_features_reg", 1.0) * losses["inversion_features_reg"]
        
        return losses


    def _domain_losses(
            self, batch,
            domain_disc,
            calc_domain_adv_loss=False,
            calc_domain_disc_loss=False,
            use_e4e_domain_features_loss=False,
            use_domain_inversion_loss=False,
            **kwargs):
        
        e4e_domain_inversion_256 = self.downsample_256(batch["e4e_domain_inversion"])  # <- do we need normalization on e4e outputs?
        gen_domain_inversion_256 = self.downsample_256(batch["feature_shift_domain_inversion"])

        losses = {
            "domain_MSE": self.SyntMSELoss(e4e_domain_inversion_256, gen_domain_inversion_256),
            "domain_LPIPS": self.LPIPS(e4e_domain_inversion_256, gen_domain_inversion_256),
            "domain_ID": self.IDLoss(e4e_domain_inversion_256, gen_domain_inversion_256),
            "domain_Adv": self.CalcAdvLoss(domain_disc, batch["feature_shift_domain_inversion"]) if calc_domain_adv_loss else 0.0,
        }
        
        if calc_domain_disc_loss:
            losses["domain_dics_loss"] = self.CalcDisLoss(domain_disc, batch["e4e_domain_inversion"], batch["feature_shift_domain_inversion"])

        losses = self._sum_losses(losses, "domain")
        
        if use_e4e_domain_features_loss:
            losses["domain_features_MSE"] = self.SyntMSELoss(batch["e4e_domain_features"], batch["feature_shift_domain_output"])
            losses["domain_loss"] = losses["domain_loss"] + \
                self.losses_dict.get(f"domain_features_MSE", 1.0) * losses["domain_features_MSE"]
        
        if use_domain_inversion_loss:
            source_image_256 = self.downsample_256(batch["images"])

            losses.update({
                "domain_inversion_MSE": self.SyntMSELoss(source_image_256, gen_domain_inversion_256),
                "domain_inversion_LPIPS": self.LPIPS(normalize(source_image_256), gen_domain_inversion_256),
                "domain_inversion_ID": self.IDLoss(normalize(source_image_256), gen_domain_inversion_256),
                "domain_inversion_Adv": 0.0
            })
            losses = self._sum_losses(losses, "domain_inversion")
            losses["domain_loss"] = losses["domain_loss"] + losses["domain_inversion_loss"]
    
        return losses


    def __call__(
            self, batch,
            domain_disc, source_disc,
            calc_domain_adv_loss=False,
            calc_domain_disc_loss=False,
            calc_inversion_adv_loss=True,
            calc_inversion_disc_loss=False,
            use_e4e_inversion_features_loss=False,
            use_fse_inversion_features_loss=False,
            use_e4e_domain_features_loss=False,
            use_domain_inversion_loss=False,
            only_inversion=False,
            **kwargs
        ):

        losses = self._invresion_losses(
            batch, source_disc,
            calc_inversion_adv_loss, calc_inversion_disc_loss,
            use_e4e_inversion_features_loss, use_fse_inversion_features_loss
        )

        if only_inversion:
            losses["loss"] = losses["inversion_loss"]
            return losses

        losses.update(self._domain_losses(
            batch, domain_disc,
            calc_domain_adv_loss,
            calc_domain_disc_loss,
            use_e4e_domain_features_loss,
            use_domain_inversion_loss
        ))

        losses["loss"] = losses["domain_loss"] + losses["inversion_loss"]
        return losses
