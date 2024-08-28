from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from tqdm import tqdm

import io
import os
import yaml

from examples.draw_util import weights
from core.utils.example_utils import read_img


def get_image_domains():
    f_stdout = io.StringIO()

    domain_images_path = 'image_domains/'
    domain_images = {}

    for domain_image_filename in tqdm(os.listdir(domain_images_path)):
        s_domain, file_ext = domain_image_filename.split('.')
        if file_ext in ['png', 'jpg'] and s_domain != 'anime' and s_domain in weights:
            with redirect_stdout(f_stdout):
                domain_images[s_domain] = read_img(domain_images_path + domain_image_filename, align_input=True)

    domain_images = domain_images.items()
    s_domains = [x[0] for x in domain_images]
    domain_ims = [x[1] for x in domain_images]

    print('Considering the following domains:', *s_domains)
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
    
    return s_domains, domain_texts