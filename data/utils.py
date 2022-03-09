import os
import wget
import gzip
import torch
import gdown
import kaggle
import shutil
import zipfile
import tarfile
import hashlib
import numpy as np
import urllib.request
import xml.dom.minidom
from tqdm.std import tqdm
from PIL import Image, ImageDraw
from torch.functional import Tensor
from torch.utils.data.dataset import Dataset, Subset
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple


def google_ping_test():
    url = "https://drive.google.com/"
    try:
        status = urllib.request.urlopen(url).code
        return True
    except Exception as err:
        return False


def download_data_from_url(url, root):
    wget.download(url, os.path.join(root, url.split('/')[-1]))


def download_data_from_url_with_path(url, path):
    wget.download(url, path)


def download_data_from_kaggle_competition(competition, root):
    kaggle.api.competition_download_files(competition, root)


def download_data_from_kaggle_dataset(dataset, root):
    kaggle.api.dataset_download_files(dataset, root)


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def chack_file_with_md5(file_name, md5):
    return md5 == calculate_md5(file_name)


def download_data_from_google_drive(url, path):
    if not google_ping_test():
        print('Can not connect to google,may you should set proxy')
    gdown.download(url, output=path)


def polygon2mask_default(w: int, h: int, polygons: list) -> Image.Image:
    binary_mask = Image.new('L', (w, h), 0)
    for polygon in polygons:
        ImageDraw.Draw(binary_mask).polygon(polygon, outline=255, fill=255)
    return binary_mask


def xml_to_binary_mask(w, h, filename: str, polygon2mask=polygon2mask_default) -> Image.Image:
    xml_file = filename
    xDoc = xml.dom.minidom.parse(xml_file).documentElement
    Regions = xDoc.getElementsByTagName('Region')
    xy = []
    for i, Region in enumerate(Regions):
        verticies = Region.getElementsByTagName('Vertex')
        xy.append(np.zeros((len(verticies), 2)))
        for j, vertex in enumerate(verticies):
            xy[i][j][0], xy[i][j][1] = float(vertex.getAttribute('X')), float(vertex.getAttribute('Y'))
    polygons = []
    for zz in xy:
        polygon = []
        for k in range(len(zz)):
            polygon.append((zz[k][0], zz[k][1]))
        polygons.append(polygon)
    return polygon2mask(w, h, polygons)


def untar_file(tar_src, dst_dir):
    tar = tarfile.open(tar_src)
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name, dst_dir)
    tar.close()


def to_tensor_3D(img: np.ndarray):
    assert isinstance(img, np.ndarray) and len(img.shape) < 5 and len(img.shape) > 2
    if len(img.shape) == 3:
        return torch.tensor(img).unsqueeze(3).permute(3, 2, 1, 0).contiguous()
    else:
        return torch.tensor(img).permute(3, 2, 1, 0).contiguous()


def ungz_file(gz_src: str, dst_dir: str):
    with gzip.open(gz_src, 'rb') as f_in:
        with open(os.path.join(dst_dir, os.path.basename(gz_src)[:-3]), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


from torch.utils.data import random_split as rsp
class Subset_with_T(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset =dataset
        self.transforms = None

    def set_transforms(self,transforms:Callable):
        self.transforms = transforms

    def __getitem__(self, index) -> Any:
        if self.transforms is not None:
            data = self.dataset[index]
            if isinstance(data,(Tuple,List)):
                return self.transforms(*data)
            else:
                return self.transforms(data)
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)



def random_split(dataset: Dataset, lengths: Sequence[int]) -> List[Subset_with_T]:
    subsets = rsp(dataset,lengths)
    return tuple([Subset_with_T(_) for _ in subsets])
