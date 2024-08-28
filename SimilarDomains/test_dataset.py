import io
import os
import torch

from contextlib import redirect_stdout
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from core.utils.example_utils import run_alignment


class SimpleImageDataset(Dataset):
    def __init__(self, image_dir_path, transform=None):
        super().__init__()

        self.image_dir = image_dir_path
        self.transform = transform

        self.filenames = sorted(os.listdir(image_dir_path))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = self.image_dir + filename
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "filename": filename}


def align_image_dataset(dataset_path, aligned_dataset_path, predictor_path="pretrained/shape_predictor_68_face_landmarks.dat"):
    dataset_path = Path(dataset_path)
    aligned_dataset_path = Path(aligned_dataset_path)
    aligned_dataset_path.mkdir(parents=True, exist_ok=True)

    f_stdout = io.StringIO()

    print(f'Starting to align images from {dataset_path} into {aligned_dataset_path}...')
    for filename in tqdm(os.listdir(dataset_path)):
        with redirect_stdout(f_stdout):
            aligned_image = run_alignment(dataset_path / filename, predictor_path=predictor_path)
            aligned_image.save(aligned_dataset_path / filename)
    
    print(f'Done!')
