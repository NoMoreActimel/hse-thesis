import json
import dlib
import numpy as np
import os
import PIL
import sys
import torch

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from tqdm import tqdm

from core.utils.common import align_face
from core.utils.example_utils import (
    read_img, to_im, Inferencer,
    project_e4e, project_fse_without_image_generation
)
from core.utils.reading_weights import read_weights
from examples.draw_util import weights


class FeatureShiftImageDataset(Dataset):
    def __init__(self, dataset_config, latents_type="e4e"):
        super().__init__()

        self.config = dataset_config
        self.device = self.config.device
        self.image_size = self.config.image_size
        self.align_input = self.config.align_input

        self.dataset_path = Path(self.config.dataset_path)
        self.processed_path = Path(self.config.processed_path)
        self.processed_path.mkdir(exist_ok=True, parents=True)
        self.index_path = self.processed_path / "index.json"

        self.process_data = self.config.process_data
        if self.process_data:
            assert not os.listdir(self.processed_path), \
                "Required data processing, but the specified direcotry is not empty"

        self.dataset_size_limit = self.config.get('dataset_size_limit', int(1e10))
        self.default_image_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )

        self.latents_type = latents_type

        self.index = self._get_or_load_index()

    def _process_dataset(self):
        dataset_files = os.listdir(self.dataset_path)
        dataset_length = min(len(dataset_files), self.dataset_size_limit)

        inversion_resize = transforms.Resize(self.image_size)
        # inversion_transform = transforms.Compose(
        #     [
        #         transforms.Resize((self.image_size, self.image_size)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #     ]
        # )

        images = []
        file_names = []
        for i, file_name in tqdm(enumerate(dataset_files), total=dataset_length):            
            image = read_img(self.dataset_path / file_name, align_input=self.align_input)    # what if it could not be aligned? 
            images.append(image)
            file_names.append(file_name)

            if (i and (i + 1) % self.config.processing_batch_size == 0) or (i == self.dataset_size_limit):
                e4e_inversion_batch, e4e_latents_batch, e4e_features_batch = self._get_e4e_inversion(images)
                fse_inversion_batch, fse_latents_batch, fse_features_batch, fse_generator_features_batch = self._get_fse_inversion(images, return_generator_features=True)

                print(e4e_inversion_batch.shape, e4e_latents_batch.shape, len(e4e_features_batch), fse_inversion_batch.shape, fse_latents_batch.shape, len(fse_features_batch))

                e4e_inversion_images = [to_im(inversion_resize(e4e_inversion)) for e4e_inversion in e4e_inversion_batch]
                fse_inversion_images = [to_im(inversion_resize(fse_inversion)) for fse_inversion in fse_inversion_batch]

                # get fse latents and features for e4e-inverted images

                fse_latents_on_e4e_inv_batch, fse_features_on_e4e_inv_batch = self._get_fse_inversion(e4e_inversion_images, return_inversion=False)
                fse_latents_on_fse_inv_batch, fse_features_on_fse_inv_batch = self._get_fse_inversion(fse_inversion_images, return_inversion=False)

                for image, file_name, \
                        e4e_inversion_image, e4e_latents, e4e_features, \
                        fse_inversion_image, fse_latents, fse_features, fse_generator_features, \
                        fse_latents_on_e4e_inv, fse_features_on_e4e_inv, \
                        fse_latents_on_fse_inv, fse_features_on_fse_inv in zip(
                    images, file_names,
                    e4e_inversion_images, e4e_latents_batch, e4e_features_batch,
                    fse_inversion_images, fse_latents_batch, fse_features_batch, fse_generator_features_batch,
                    fse_latents_on_e4e_inv_batch, fse_features_on_e4e_inv_batch,
                    fse_latents_on_fse_inv_batch, fse_features_on_fse_inv_batch,
                ):
                    image_name, image_filetype = file_name.split('.')

                    image_dir_path = self.processed_path / image_name
                    image_dir_path.mkdir(exist_ok=True, parents=True)

                    image.save(image_dir_path / f"image.{image_filetype}")

                    e4e_inversion_image.save(image_dir_path / f"e4e_inversion.{image_filetype}", format=image_filetype)
                    torch.save(e4e_latents, image_dir_path / f"e4e_latents.pt")
                    torch.save(e4e_features, image_dir_path / f"e4e_features.pt")

                    fse_inversion_image.save(image_dir_path / f"fse_inversion.{image_filetype}", format=image_filetype)
                    # torch.save(fse_inversion, image_dir_path / f"fse_inversion.pt")
                    torch.save(fse_latents, image_dir_path / f"fse_latents.pt")
                    torch.save(fse_features, image_dir_path / f"fse_features.pt")
                    torch.save(fse_generator_features, image_dir_path / f"fse_generator_features.pt")

                    torch.save(fse_latents_on_e4e_inv, image_dir_path / f"fse_latents_on_e4e_inv.pt")
                    torch.save(fse_features_on_e4e_inv, image_dir_path / f"fse_features_on_e4e_inv.pt")
                    torch.save(fse_latents_on_fse_inv, image_dir_path / f"fse_latents_on_fse_inv.pt")
                    torch.save(fse_features_on_fse_inv, image_dir_path / f"fse_features_on_fse_inv.pt")
            
                images = []
                file_names = []
            
            if i + 1 == self.dataset_size_limit:
                break
    
    def _prepare_image(self, image):
        # NOT USED
        resize = transforms.Resize(self.image_size)
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
        return np.array(to_im(resize(transform(image)), padding=0))
    

    def _get_e4e_inversion(self, images):
        e4e_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        images = torch.cat([e4e_transform(image).unsqueeze(0) for image in images])
        print("e4e transform:", images.shape)

        _, e4e_latents = project_e4e(
            images,
            self.config.inversion.e4e_model_path,
            use_transforms=False,
            device=self.device
        )
        e4e_latents = e4e_latents.unsqueeze(0)

        ckpt = read_weights(weights['jojo'])
        ckpt_ffhq = {'sg2_params': ckpt['sg2_params']}
        ckpt_ffhq['sg2_params']['checkpoint_path'] = weights['ffhq']
        model = Inferencer(ckpt, self.device)


        (e4e_inversion, _), (e4e_features, __) = model(
            latents=e4e_latents,
            input_is_latent=True,
            features_in=None,
            shift_with_generator_feature_map=False,
            return_latents=False,
            return_features=True,
            return_generator_features=True
        )

        e4e_latents = e4e_latents.squeeze(0)
        print('e4e_latents.shape:', e4e_latents.shape)
        e4e_features = e4e_features[self.config.inversion.fse_idx_k]
        print('e4e_features.shape:', e4e_features.shape)

        return e4e_inversion, e4e_latents, e4e_features
        

    def _get_fse_inversion(self, images, return_inversion=True, return_generator_features=False):
        fse_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
        images = torch.cat([fse_transform(image).unsqueeze(0) for image in images])
        print("fse transform:", images.shape)

        _, fse_latents, fse_features = project_fse_without_image_generation(
            images,
            model_path=self.config.inversion.fse_model_path,
            fse_config_path=self.config.inversion.fse_config_path,
            arcface_model_path=self.config.inversion.arcface_model_path,
            stylegan_model_path=self.config.inversion.checkpoint_path,
            use_transforms=False,
            device=self.device
        )
        
        if return_inversion:
            ckpt = read_weights(weights['jojo'])
            ckpt_ffhq = {'sg2_params': ckpt['sg2_params']}
            ckpt_ffhq['sg2_params']['checkpoint_path'] = weights['ffhq']
            model = Inferencer(ckpt, self.device)

            # INVERSE USING FSE LATENTS ONLY, AS WE WOULD USE IT FOR THE GENERATOR SHIFT MEARUSEMENT

            fse_inversion, fse_generator_features = model(
                latents=fse_latents,
                input_is_latent=True,
                features_in=None,
                shift_with_generator_feature_map=False,
                return_features=return_generator_features,
                return_generator_features=return_generator_features
            )
            if return_generator_features:
                fse_inversion = fse_inversion[0]
                fse_generator_features = fse_generator_features[0]

        if images.ndim == 4:
            fse_latents = fse_latents.squeeze(0)
            print('fse_latents.shape:', fse_latents.shape)
            fse_features = [x[self.config.inversion.fse_idx_k] for x in fse_features]
            print('fse_features[i].shape:', fse_features[0].shape)

            if return_generator_features:
                fse_generator_features = fse_generator_features[self.config.inversion.fse_idx_k]
                print('fse_generator_features.shape:', fse_generator_features.shape)

        if return_inversion:
            if return_generator_features:
                return fse_inversion, fse_latents, fse_features, fse_generator_features
            return fse_inversion, fse_latents, fse_features
        else:
            if return_generator_features:
                return fse_latents, fse_features, fse_generator_features
            return fse_latents, fse_features
    
    # @staticmethod
    # def transform_features_to_list(features, layer_idx):
    #     assert features.ndim == 4, f"Works on batched features only, but features.shape: {features.shape}"
    #     for f in features:
    #         pass

    def _get_or_load_index(self):
        if self.index_path.exists():
            with self.index_path.open() as f:
                index = json.load(f)
            if len(index) == self.dataset_size_limit:
                return index

        if self.process_data:
            self._process_dataset()
        
        index = []
        for image_dir_path in self.processed_path.iterdir():
            if os.path.isdir(image_dir_path):
                index.append(os.path.basename(os.path.normpath(image_dir_path)))
                if len(index) == self.dataset_size_limit:
                    break
        
        with self.index_path.open('w') as f:
            json.dump(index, f)
        
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, ind):
        image_dir_path = self.processed_path / self.index[ind]
        filename = os.path.basename(os.path.normpath(image_dir_path))

        image_path = list(image_dir_path.glob("image.*"))[0]
        e4e_inversion_path = list(image_dir_path.glob("e4e_inversion.*"))[0]
        fse_inversion_path = list(image_dir_path.glob("fse_inversion.*"))[0]

        image = self.default_image_transform(Image.open(image_path))
        e4e_inversion = self.default_image_transform(Image.open(e4e_inversion_path))
        fse_inversion = self.default_image_transform(Image.open(fse_inversion_path))

        e4e_latents = torch.load(image_dir_path / f"e4e_latents.pt", map_location="cpu")
        fse_latents = torch.load(image_dir_path / f"fse_latents.pt", map_location="cpu")

        e4e_features = torch.load(image_dir_path / f"e4e_features.pt", map_location="cpu")
        fse_features = torch.load(image_dir_path / f"fse_features.pt", map_location="cpu")

        if self.latents_type == "fse":
            fse_generator_features = torch.load(image_dir_path / f"fse_generator_features.pt", map_location="cpu")
        else:
            fse_generator_features = None

        fse_latents_on_inv = torch.load(image_dir_path / f"fse_latents_on_{self.latents_type}_inv.pt", map_location="cpu")
        fse_features_on_inv = torch.load(image_dir_path / f"fse_features_on_{self.latents_type}_inv.pt", map_location="cpu")

        return {
            "filename": filename,
            "image": image,
            "e4e_inversion": e4e_inversion,
            "e4e_latents": e4e_latents,
            "e4e_features": e4e_features,
            "fse_inversion": fse_inversion,
            "fse_latents": fse_latents,
            "fse_features": fse_features,
            "fse_generator_features": fse_generator_features,
            "fse_latents_on_inv": fse_latents_on_inv,
            "fse_features_on_inv": fse_features_on_inv
        }
    
    def collate_fn(self, batch_items):
        if self.latents_type == "fse":
            batch = {
                "filenames": [item["filename"] for item in batch_items],
                "images": torch.cat([item["image"].unsqueeze(0) for item in batch_items]),
                "e4e_inversion": torch.cat([item["fse_inversion"].unsqueeze(0) for item in batch_items]),
                "e4e_latents": torch.cat([item["fse_latents"].unsqueeze(0) for item in batch_items]),
                "fse_latents": torch.cat([item["fse_latents"].unsqueeze(0) for item in batch_items]),
                "fse_features": torch.cat([item["fse_features"] for item in batch_items]),
                "fse_latents_on_inv": torch.cat([item["fse_latents_on_inv"].unsqueeze(0) for item in batch_items]),
                "fse_features_on_inv": torch.cat([item["fse_features_on_inv"] for item in batch_items]),
                "e4e_features": torch.cat([item["fse_generator_features"].unsqueeze(0) for item in batch_items])
            }
            return batch
        batch = {
            "filenames": [item["filename"] for item in batch_items],
            "images": torch.cat([item["image"].unsqueeze(0) for item in batch_items]),
            "e4e_inversion": torch.cat([item["e4e_inversion"].unsqueeze(0) for item in batch_items]),
            "e4e_latents": torch.cat([item["e4e_latents"].unsqueeze(0) for item in batch_items]),
            "e4e_features": torch.cat([item["e4e_features"].unsqueeze(0) for item in batch_items]),
            "fse_inversion": torch.cat([item["fse_inversion"].unsqueeze(0) for item in batch_items]),
            "fse_latents": torch.cat([item["fse_latents"].unsqueeze(0) for item in batch_items]),
            "fse_features": torch.cat([item["fse_features"] for item in batch_items]),
            "fse_latents_on_inv": torch.cat([item["fse_latents_on_inv"].unsqueeze(0) for item in batch_items]),
            "fse_features_on_inv": torch.cat([item["fse_features_on_inv"] for item in batch_items])
        }
        return batch
