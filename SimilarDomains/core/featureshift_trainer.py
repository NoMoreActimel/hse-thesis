import io
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.functional as F
import yaml

from collections import defaultdict
from contextlib import redirect_stdout
from itertools import repeat
from pathlib import Path
from PIL import ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.featureshift_dataset import FeatureShiftImageDataset
from core.parametrizations import BaseParametrization
from core.utils.common import (
    get_trainable_model_state, requires_grad,
    get_stylegan_conv_dimensions
)
from core.utils.example_utils import to_im, read_img
from core.utils.image_utils import BicubicDownSample
from core.utils.loggers import LoggingManager
from core.utils.reading_weights import read_weights
from core.utils.train_log import StreamingMeans, TimeLog, Timer
from core.uda_models import uda_models
from core.featureshift_losses import FeatureShiftLossBuilder

from visualize import stack_to_grid_with_names, text_on_square_image

from examples.draw_util import weights
# from gan_models.StyleGAN2.model import Discriminator
from gan_models.StyleGAN2.dnnlib.util import open_url

def freeze_layers(model):
    for layer in list(model.children()):
        requires_grad(layer, False)

def unfreeze_layers(model):
    for layer in list(model.children()):
        requires_grad(layer, True)

def print_requires_grad(model):
    for layer in list(model.children()):
        print(layer)
        for p in layer.parameters():
            print(p.requires_grad)

def inf_loop(dataloader):
    for loader in repeat(dataloader):
        yield from loader

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return _TFNetworkStub
        module = module.replace('torch_utils', 'gan_models.StyleGAN2.torch_utils')
        module = module.replace('dnnlib', 'gan_models.StyleGAN2.dnnlib')
        return super().find_class(module, name)


def get_image_domains():
    f_stdout = io.StringIO()

    domain_images_path = 'image_domains/'
    domain_images = {}

    for domain_image_filename in tqdm(os.listdir(domain_images_path)):
        if '.' not in domain_image_filename:
            continue
        s_domain, file_ext = domain_image_filename.split('.')
        if file_ext in ['png', 'jpg'] and s_domain != 'anime' and s_domain in weights:
            with redirect_stdout(f_stdout):
                domain_images[s_domain] = read_img(domain_images_path + domain_image_filename, align_input=False)

    domain_images = domain_images.items()
    s_domains = [x[0] for x in domain_images]
    domain_ims = [x[1] for x in domain_images]

    ignored_image_domains = ['joker', 'sketch']
    for domain in ignored_image_domains:
        if domain in s_domains:
            ind = s_domains.index(domain)
            s_domains.pop(ind)
            domain_ims.pop(ind)

    print('Considering the following image domains:', *s_domains)
    return s_domains, domain_ims

def get_text_domains(domain_dir_path='pretrained/checkpoints_iccv'):
    path = Path(domain_dir_path)
    
    s_domains = []
    domain_texts = []
    
    for domain_dir in tqdm(path.iterdir()):
        with open(domain_dir / 'config.yaml', 'r') as file:
            domain_config = yaml.safe_load(file)
        
        if domain_config['training']['target_class'][-4:] not in ['.png', '.jpg']:
            s_domains.append(domain_dir.name[:-7])  # cut out _sdelta in the end
            domain_texts.append(domain_config['training']['target_class'])
            if 'indomain' in s_domains[-1]:
                domain_texts[-1] += ' (indomain)'

    print('Considering the following text domains:', *s_domains)
    return s_domains, domain_texts

class FeatureShiftDomainAdaptationTrainer:
    def __init__(self, config): #, model, dataloaders, optimizer, scheduler, train_metrics, val_metrics):
        super().__init__()

        self.config = config
        self.device = config.training.device

        # self.s_domain = self.config.s_domain
        self.im_domains, self.domain_ims = get_image_domains()
        self.text_domains, self.domain_texts = get_text_domains()
        self.domain_limit = self.config.get("domain_limit", None)
        if self.domain_limit is not None:
            self.im_domains = self.im_domains[:(self.domain_limit + 1) // 2]
            self.domain_ims = self.domain_ims[:(self.domain_limit + 1) // 2] 
            self.text_domains = self.text_domains[:self.domain_limit // 2]
            self.domain_texts = self.domain_texts[:self.domain_limit // 2]
        
        self.s_domains = self.im_domains + self.text_domains
        self.domain_ims_or_texts = self.domain_ims + self.domain_texts
    
        self.fse_idx_k = self.config.train_dataset.inversion.fse_idx_k

        self.latents_type = self.config.get("latents_type", "e4e")

        self.source_generator = None
        self.models_da = None
        self.source_discriminator = None
        self.domain_discriminator = None
        self.trainable = None

        self.loss_builder = None
        self.optimizer = None
        self.scheduler = None
        self.domain_disc_optimizer = None

        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.iter_steps = self.config.training.iter_num
        self.val_freq = self.config.training.val_freq
        self.log_step = self.config.training.log_step
        self.log_images_step = self.config.training.log_images_step
        self.save_step = self.config.training.save_step

        self.use_inversion_adv_loss = self.config.training.use_inversion_adv_loss
        self.inversion_adv_loss_min_step = self.config.training.get("inversion_adv_loss_min_step", 0)

        self.use_domain_adv_loss = self.config.training.use_domain_adv_loss
        self.use_domain_disc_loss = self.config.training.use_domain_disc_loss
        self.domain_adv_loss_min_step = self.config.training.get("domain_adv_loss_min_step", 0)
        self.use_domain_inversion_loss = self.config.training.use_domain_inversion_loss
        
        self.use_e4e_domain_features_loss = self.config.training.use_e4e_domain_features_loss
        self.use_e4e_inversion_features_loss = self.config.training.use_e4e_inversion_features_loss
        self.use_fse_inversion_features_loss = self.config.training.use_fse_inversion_features_loss
        self.only_inversion = self.config.training.only_inversion

        self.calculate_grad_norms = self.config.training.get('calculate_grad_norms', False)
        self.clip_grad_norm = self.config.training.get('clip_grad_norm', None)

        self.logger = None
        self.logging_domains = None
        self.logging_domain_ims_or_texts = None

        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )

        self._setup_dataloaders()

        self._setup_source_generator()
        self._setup_models_da()
        self._setup_discriminators()
        self._setup_trainable()

        self._setup_loss_builder()
        self._setup_optimizers()
        self._setup_logging()

        self.domain_offsets = {
            s_domain: self.models_da[s_domain]()
            for s_domain in self.s_domains
        }

    def _setup_dataloaders(self):
        self.train_dataset = FeatureShiftImageDataset(self.config.train_dataset, self.latents_type)
        self.val_dataset = FeatureShiftImageDataset(self.config.val_dataset, self.latents_type)
        
        self.train_dataloader = inf_loop(DataLoader(
            self.train_dataset,
            batch_size=self.config.train_dataset.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=self.config.train_dataset.get("shuffle", True),
            num_workers=self.config.train_dataset.get("num_workers", 1),
            drop_last=True
        ))
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.val_dataset.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            shuffle=self.config.val_dataset.get("shuffle", False),
            num_workers=self.config.val_dataset.get("num_workers", 1),
            drop_last=False
        )

    def _setup_source_generator(self):
        self.source_generator = uda_models[self.config.training.source_generator](
            **self.config.training.source_generator_args #[self.config.training.source_generator]  # <-- use_feature_shift_conv = False
        )
        self.source_generator.patch_layers(self.config.training.patch_key)       # WTF ?
        self.source_generator.freeze_layers()
        self.source_generator.to(self.device)
    
    def _setup_models_da(self):
        self.models_da = {}
        for s_domain in self.s_domains:
            ckpt = read_weights(weights[s_domain])
            self.models_da[s_domain] = BaseParametrization(
                ckpt['patch_key'],
                get_stylegan_conv_dimensions(ckpt['sg2_params']['img_size']),
            )
            self.models_da[s_domain].load_state_dict(ckpt["state_dict"], strict=False)
            self.models_da[s_domain].to(self.device).eval()
            self.models_da[s_domain].freeze_layers()

    def _setup_trainable(self):
        self.trainable = uda_models[self.config.training.trainable](
            **self.config.training.trainable_generator_args #[self.config.training.trainable]  # <-- use_feature_shift_conv = True
        )
        self.trainable.patch_layers(self.config.training.patch_key)       # WTF ?
        trainable_layers = list(self.trainable.get_training_layers(
            phase=self.config.training.phase
        ))
        print('trainable_layers:', trainable_layers)
        self.trainable.freeze_layers()
        self.trainable.unfreeze_layers(trainable_layers)
        self.trainable.to(self.device)
    
    def _setup_discriminators(self):
        if self.use_inversion_adv_loss:
            with open_url(self.config.training.source_discriminator_path) as f:
                data = _LegacyUnpickler(f).load()
            self.source_discriminator = data['D'].to(self.device).eval()

        if self.use_domain_adv_loss:
            if "domain_discriminator_path" in self.config.training:
                with open_url(self.config.training.domain_discriminator_path) as f:
                    data = _LegacyUnpickler(f).load()
                self.domain_discriminator = data['D'].to(self.device)

        # self.source_discriminator = Discriminator(size=self.config.image_size)
        # checkpoint = torch.load(self.config.training.source_discriminator_path, map_location=self.device)
        # self.source_discriminator.load_state_dict(checkpoint["d_ema"], strict=False)
        # # layers are frozen as this discriminator works on the source images in the inversion loss
        # freeze_layers(self.source_discriminator)
        # self.source_discriminator.to(self.device)
        # self.source_discriminator.eval()
    
        # self.domain_discriminator = Discriminator(size=self.config.image_size)
        # checkpoint = torch.load(self.config.training.domain_discriminator_path, map_location=self.device)
        # self.domain_discriminator.load_state_dict(checkpoint["d_ema"], strict=False)
        # # layers are not frozen as this discriminator finetunes on e4e_domain_inversion, in the domain loss
        # self.domain_discriminator.to(self.device)
    
    def _setup_loss_builder(self):
        self.loss_builder = FeatureShiftLossBuilder(losses_dict=self.config.training.losses_dict, device=self.device)

    def _setup_optimizers(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.trainable.parameters()),
            **self.config.training.optimization_setup.optimizer
        )

        if self.use_domain_disc_loss:
            self.domain_disc_optimizer = torch.optim.Adam(
                self.domain_discriminator.parameters(),
                **self.config.training.optimization_setup.domain_discriminator_optimizer
            )
        
        if self.config.training.optimization_setup.get('scheduler', False):
            lr = self.config.training.optimization_setup.optimizer.lr
            steps = self.config.training.optimization_setup.scheduler.n_steps
            start_lr = self.config.training.optimization_setup.scheduler.start_lr
            alpha = lr - start_lr
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda i: start_lr/lr + min(1, i / steps) * alpha / lr
            )
    
    def _setup_logging(self):
        self.logger = LoggingManager(self.config)

        self.logging_domains = []
        self.logging_domain_ims_or_texts = []
        if self.config.training.get("logging_domains", None) is not None:
            logging_domains = self.config.training.logging_domains
            for domain_ind, s_domain in enumerate(self.s_domains):
                if s_domain in logging_domains:
                    self.logging_domains.append(s_domain)
                    self.logging_domain_ims_or_texts.append(self.domain_ims_or_texts[domain_ind])
        else:
            for _ in range(4):
                domain_ind = random.choice(range(len(self.s_domains)))
                self.logging_domains.append(self.s_domains[domain_ind])
                self.logging_domain_ims_or_texts.append(self.domain_ims_or_texts[domain_ind])

        
    def start_from_checkpoint(self):
        step = 0
        if self.config.training.checkpointing.start_from:
            state_dict = torch.load(self.config.training.checkpointing.start_from, map_location='cpu')
            step = state_dict['step']
            self.trainable.load_state_dict(state_dict['trainable'])
            self.optimizer.load_state_dict(state_dict['trainable_optimizer'])
            print('starting from step {}'.format(step))
        return step
    
    def get_checkpoint(self):
        state_dict = {
            "step": self.current_step,
            "trainable": self.trainable.state_dict(),                     # SAVE THE WHOLE MODEL
            "trainable_optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        return state_dict
    
    def make_checkpoint(self):
        if not self.config.training.checkpointing.is_on:
            return

        ckpt = self.get_checkpoint()
        torch.save(ckpt, os.path.join(self.logger.checkpoint_dir, f"checkpoint_{self.current_step}.pt"))

    def get_trainable(self):
        return self.trainable.generator.feature_shift_conv
    
    def save_trainable(self):
        torch.save(self.get_trainable(), str(
            Path(self.logger.models_dir) / f"models_{self.current_step}.pt"  # SAVE FEATURESHIFT ONLY 
        ))
    
    @torch.no_grad()
    def get_grad_norm(self, parameters, norm_type=2):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]),
            norm_type
        )
        return total_norm.item()

    def _log_images_invesion_only(self, batch, prefix="train"):
        if "fse_features_inversion" not in batch:
            batch["fse_features_inversion"], _ = self.source_generator(
                styles=batch["fse_latents"],
                offsets=None,
                input_is_latent=True,
                features_in=batch["fse_features"],
                shift_with_generator_feature_map=False
            )
        
        row_stack = []
        for i, (image, e4e_invresion, fse_inversion, inversion) in enumerate(zip(
            batch["images"], batch["e4e_inversion"],
            batch["fse_features_inversion"], batch["feature_shift_inversion"]
        )):  
            row_stack.append([])
            for im_tensor in [image, e4e_invresion, fse_inversion, inversion]:
                row_stack[-1].append(np.array(to_im(im_tensor)))
            if i + 1 == 4:
                break
        
        image_size = self.train_dataset.image_size
        stacked_image = stack_to_grid_with_names(
            imgs_list=row_stack, H=image_size, W=image_size,
            row_names=batch["filenames"][:4], 
            column_names=["image", self.latents_type, "fse_features", "featureshift"],
            font=ImageFont.truetype("/home/sasedov/Times.ttf", 25 * image_size // 256),
        )
        dict_to_log = {f"{self.current_domain}/{prefix}_featureshift_inversions/": stacked_image}
        self.logger.log_images(self.current_step, dict_to_log)


    def log_images(self, batch, prefix="train"):
        # images = construct_paper_image_grid(batch["images"])
        # e4e_inversion = construct_paper_image_grid(batch["e4e_inversion"])
        # e4e_domain_inversion = construct_paper_image_grid(batch["e4e_domain_inversion"])
        if self.only_inversion:
            self._log_images_invesion_only(batch, prefix)
            return
        
        if "fse_features_inversion" not in batch:
            batch["fse_features_inversion"], _ = self.source_generator(
                styles=batch["fse_latents"],
                offsets=None,
                input_is_latent=True,
                features_in=batch["fse_features"],
                shift_with_generator_feature_map=False
            )
            batch["fse_features_domain_inversion"], _ = self.source_generator(
                styles=batch["fse_latents"],
                offsets=self.domain_offsets[self.current_domain],
                input_is_latent=True,
                features_in=batch["fse_features"],
                shift_with_generator_feature_map=False
            )
            batch["feature_shift_domain_inversion_fse"], _ = self.trainable(
                styles=batch["fse_latents"],
                offsets=self.domain_offsets[self.current_domain],
                input_is_latent=True,
                features_in=batch["fse_features"],
                shift_with_generator_feature_map=False,
                feature_shift_delta=batch["e4e_features_delta"]
            )

        row_stack = []
        image_size = self.train_dataset.image_size
        if prefix == "train":
            if isinstance(self.current_domain_im_or_text, str):
                row_stack.append([text_on_square_image(
                    self.current_domain_im_or_text, image_size,
                    font=ImageFont.truetype("/home/sasedov/Times.ttf", 25 * image_size // 256),
                    linewidth=20
                )])
            else:
                row_stack.append([np.array(to_im(self.transform(self.current_domain_im_or_text).unsqueeze(0)))])
            for _ in range(6):
                empty = np.zeros_like(row_stack[0][0])
                empty.fill(255)
                row_stack[0].append(empty)
        
        for i, (image, e4e_invresion, e4e_domain_inversion, \
            fse_inversion, fse_domain_inversion, inversion, domain_inversion) in enumerate(zip(
            batch["images"], batch["e4e_inversion"], batch["e4e_domain_inversion"],
            batch["fse_features_inversion"], batch["fse_features_domain_inversion"],
            batch["feature_shift_inversion"], batch["feature_shift_domain_inversion_fse"]
        )):  
            row_stack.append([])
            for im_tensor in [image, e4e_invresion, e4e_domain_inversion,
                              fse_inversion, fse_domain_inversion, inversion, domain_inversion]:
                row_stack[-1].append(np.array(to_im(im_tensor)))
            if i + 1 == 4:
                break
        
        domain_row_name = "" if isinstance(self.current_domain_im_or_text, str) else self.current_domain
        stacked_image = stack_to_grid_with_names(
            imgs_list=row_stack, H=image_size, W=image_size,
            row_names=[domain_row_name] + batch["filenames"][:4] if prefix == "train" else batch["filenames"][:4],
            column_names=[
                "image", self.latents_type, f"{self.latents_type}_domain", 
                "fse_features", "fse_features_domain",
                "featureshift", "featureshift_domain"
            ],
            font=ImageFont.truetype("/home/sasedov/Times.ttf", 25 * image_size // 256),
        )

        if prefix == "train":
            dict_to_log = {f"all_domains/{prefix}_featureshift_inversions/": stacked_image}
        else:
            dict_to_log = {f"{self.current_domain}/{prefix}_featureshift_inversions/": stacked_image}
        
        self.logger.log_images(self.current_step, dict_to_log)


    def train_loop(self):
        training_time_log = TimeLog(
            self.logger, self.config.training.iter_num + 1, event="training"
        )

        recovered_step = self.start_from_checkpoint()
        self.current_step = recovered_step
        iter_info = StreamingMeans()

        for batch in tqdm(self.train_dataloader, desc='train', total=self.iter_steps):
            domain_ind = random.choice(range(len(self.s_domains)))
            self.current_domain = self.s_domains[domain_ind]
            self.current_domain_im_or_text = self.domain_ims_or_texts[domain_ind]

            with Timer(iter_info, "train_iter"):
                self._process_batch(batch, iter_info)
            
            if (self.current_step + 1) % self.config.training.checkpointing.step_backup == 0:
                self.make_checkpoint()

            if (self.current_step + 1) % self.save_step == 0:
                self.save_trainable()
            
            if self.current_step % self.log_images_step == 0:
                with Timer(iter_info, "log_images"):
                    self.log_images(batch, prefix="train")

            if self.current_step % self.log_step == 0:
                self.logger.log_values(
                    self.current_step, self.iter_steps, iter_info
                )
                iter_info.clear()
                training_time_log.now(self.current_step)
            
            if self.current_step % self.val_freq == 0:
                self.eval_loop()
        
            self.current_step += 1
            if self.current_step == self.iter_steps:
                break

        training_time_log.end()
        self.logger.exp_logger.finish()
    
    @torch.no_grad()
    def eval_loop(self):
        iter_info = StreamingMeans()
        for batch in tqdm(self.val_dataloader, desc='val', total=len(self.val_dataloader)):
            domain_ind = random.choice(range(len(self.logging_domains)))
            self.current_domain = self.logging_domains[domain_ind]
            self.current_domain_im_or_text = self.logging_domain_ims_or_texts[domain_ind]

            with Timer(iter_info, "val_iter"):
                self._process_batch(batch, iter_info, iter_info_prefix="val_", eval_mode=True)
        
        with Timer(iter_info, "log_images"):
            self.log_images(batch, prefix="val")
        self.logger.log_values(
            self.current_step, self.iter_steps, iter_info
        )
        iter_info.clear()

    
    def _move_batch_to_device(self, batch):
        for k, v in batch.items():
            if k not in ["filenames"]:
                batch[k] = v.to(self.device)
    
    def _print_cuda_memory_usage(self):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f"Cuda memory stats... Reserved: {r}, Allocated: {a}, Free: {f}")


    def _process_batch(self, batch, iter_info, eval_mode=False, iter_info_prefix=""):
        self._move_batch_to_device(batch)
        batch["e4e_latents"] = batch["e4e_latents"].unsqueeze(0)
        batch["fse_latents"] = batch["fse_latents"].unsqueeze(0)
        batch["fse_latents_on_inv"] = batch["fse_latents_on_inv"].unsqueeze(0)
        # self._print_cuda_memory_usage()

        # if self.latents_type == "fse":
        #     for x in "e4e_latents", "fse_latents", "e4e_inversion", "fse_inversion", "e4e_features", "fse_generator_features":
        #         print(f"{x}.shape: {batch[x].shape}")
        #     batch["e4e_latents"] = batch["fse_latents"]
        #     batch["e4e_inversion"] = batch["fse_inversion"]
        #     batch["e4e_features"] = batch["fse_generator_features"]
        
        if self.use_e4e_inversion_features_loss:
            with torch.no_grad():
                batch["e4e_inversion"], batch["e4e_inversion_features"] = self.source_generator(
                    styles=batch["e4e_latents"],
                    offsets=None,
                    input_is_latent=True,
                    features_in=None,
                    shift_with_generator_feature_map=False,
                    return_latents=False,
                    return_features=True
                )

            batch["e4e_inversion_features"] = [x.unsqueeze(0) for x in batch["e4e_inversion_features"][self.fse_idx_k]]
            batch["e4e_inversion_features"] = torch.cat(batch["e4e_inversion_features"])
            
        zero_features_delta = torch.zeros_like(batch["fse_features"])
        zero_features_delta.requires_grad = False

        batch["feature_shift_inversion"], batch["feature_shift_inversion_output"] = self.trainable(
            styles=batch["fse_latents"],
            offsets=None,
            input_is_latent=True,
            features_in=batch["fse_features"],
            shift_with_generator_feature_map=False,
            feature_shift_delta=zero_features_delta,
            return_feature_shift_output=(
                self.use_e4e_inversion_features_loss or
                self.use_fse_inversion_features_loss or
                "inversion_features_reg" in self.config.training.losses_dict
            )
        )

        if not self.only_inversion:
            with torch.no_grad():
                batch["e4e_domain_inversion"], batch["e4e_domain_features"] = self.source_generator(
                    styles=batch["e4e_latents"],
                    offsets=self.domain_offsets[self.current_domain],
                    input_is_latent=True,
                    features_in=None,
                    shift_with_generator_feature_map=False,
                    return_latents=False,
                    return_features=True
                )
            
            batch["e4e_domain_features"] = [x.unsqueeze(0) for x in batch["e4e_domain_features"][self.fse_idx_k]]
            batch["e4e_domain_features"] = torch.cat(batch["e4e_domain_features"])
            batch["e4e_features_delta"] = batch["e4e_domain_features"] - batch["e4e_features"]

            batch["feature_shift_domain_inversion"], batch["feature_shift_domain_output"] = self.trainable(
                styles=batch["fse_latents_on_inv"],
                offsets=self.domain_offsets[self.current_domain],
                input_is_latent=True,
                features_in=batch["fse_features_on_inv"],
                shift_with_generator_feature_map=False,
                feature_shift_delta=batch["e4e_features_delta"],
                return_feature_shift_output=self.use_e4e_domain_features_loss
            )

        losses = {}
        grad_norms = {}

        if (not self.only_inversion) and self.use_domain_disc_loss:
            self.domain_disc_optimizer.zero_grad()

            losses["domain_dics_loss"] = self.loss_builder.CalcDisLoss(
                self.domain_discriminator,
                batch["e4e_domain_inversion"].detach(),
                batch["feature_shift_domain_inversion"].detach()
            )

            if not eval_mode:
                losses["domain_dics_loss"].backward()
                if self.calculate_grad_norms:
                    grad_norms["domain_discriminator"] = self.get_grad_norm(self.domain_discriminator.parameters())
                self.domain_disc_optimizer.step()

        losses.update(self.loss_builder(
            batch,
            domain_disc=self.domain_discriminator,
            source_disc=self.source_discriminator,
            calc_domain_adv_loss=self.use_domain_adv_loss and self.current_step >= self.domain_adv_loss_min_step,
            calc_domain_disc_loss=self.use_domain_disc_loss and self.current_step >= self.domain_adv_loss_min_step,
            calc_inversion_adv_loss=self.use_inversion_adv_loss and self.current_step >= self.domain_adv_loss_min_step,
            calc_inversion_disc_loss=False,
            use_e4e_inversion_features_loss=self.use_e4e_inversion_features_loss,
            use_fse_inversion_features_loss=self.use_fse_inversion_features_loss,
            use_e4e_domain_features_loss=self.use_e4e_domain_features_loss,
            use_domain_inversion_loss=self.use_domain_inversion_loss,
            only_inversion=self.only_inversion
        ))

        # self._print_cuda_memory_usage()

        iter_info.update({f"{iter_info_prefix}losses/{k}": v for k, v in losses.items()})

        if not eval_mode:
            self.optimizer.zero_grad()
            losses["loss"].backward()
            # self._print_cuda_memory_usage()

            if self.calculate_grad_norms:
                grad_norms["feature_shift_block"] = self.get_grad_norm(self.get_trainable().parameters())
            
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.get_trainable().parameters(), self.clip_grad_norm)

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            if self.calculate_grad_norms:
                iter_info.update({f"{iter_info_prefix}grad_norms/{k}": v for k, v in grad_norms.items()})
