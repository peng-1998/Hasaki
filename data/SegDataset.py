import os
import shutil
from glob import glob
from typing import Callable, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import Dataset

from .utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ISBI_2016_Task1(Dataset):
    """
    Use the data on ISBI 2016.About more:https://challenge.isic-archive.com/landing/2016/37/ \\
    The size of train images:576<=width<=4288,540<=height<=2848.\\
    There are 900 image in train data and 379 in test data.\\
    The shape of mask image is H\times W and pixels value are 255 for target area and 0 for background. \\
    If transform is None,you will get size [3, H, W],[1, H, W] for tensors of image and label. \\
    You may need run label = label[0] or label=label.squeeze(dim=0) when label is tensor in callable transform.
    """

    urls = {
        'train image': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip',
        'train mask' : 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip',
        'test image' : 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip',
        'test mask'  : 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip'
    }

    md5s = {
        'train image': '2029f387e62dcc062b1370b1efc1f7fb',
        'train mask' : 'fbd77134298f3511479a37bac93109c7',
        'test image' : 'efebcaeaae751007401a40a60d391f93',
        'test mask'  : '492a7711a2e19b96114cab6c96bd1ad5'
        }

    def __init__(
        self,
        root     : str,
        mode     : str = 'train',
        transform: Optional[Callable] = None,
        download : bool = False,
    ) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        self.root      = root
        self.mode      = mode
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, ISBI_2016_Task1.urls[f'{self.mode} image'].split('/')[-1])
        if os.path.exists(file_name[:-4]):
            print(f'{self.mode} images are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading {self.mode} images ...')
                    download_data_from_url(ISBI_2016_Task1.urls[f'{self.mode} image'], self.root)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, ISBI_2016_Task1.md5s[f'{self.mode} image']):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)
        file_name = os.path.join(self.root, ISBI_2016_Task1.urls[f'{self.mode} mask'].split('/')[-1])
        if os.path.exists(file_name[:-4]):
            print(f'{self.mode} masks are already exist,cancel download.')
        else:
            print(f'Downloading {self.mode} masks ...')
            if not os.path.exists(file_name):
                while True:
                    download_data_from_url(ISBI_2016_Task1.urls[f'{self.mode} mask'], self.root)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, ISBI_2016_Task1.md5s[f'{self.mode} mask']):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        images_path = os.path.join(self.root, ISBI_2016_Task1.urls[f'{self.mode} image'].split('/')[-1].replace('.zip', ''))
        masks_path  = os.path.join(self.root, ISBI_2016_Task1.urls[f'{self.mode} mask'].split('/')[-1].replace('.zip', ''))
        images      = glob(os.path.join(images_path, '*.jpg'))
        self.data   = [(_, _.replace(images_path, masks_path)[:-4] + '_Segmentation.png') for _ in images]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        imgs = [Image.open(_) for _ in self.data[index]]
        return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)


class ISIC_2017_Task1(Dataset):
    """
    Use the data on ISIC 2017.About more:https://challenge.isic-archive.com/landing/2017/42/ \\
    The size of train images:576<=width<=6688,540<=height<=4497.\\
    There are 2000 image in train data and 150 in validation data and 600 in test data.\\
    The shape of mask image is H\times W and pixels value are 255 for target area and 0 for background. \\
    If transform is None,you will get size [3, H, W],[1, H, W] for tensors of image and label. \\
    You may need run label = label[0] or label=label.squeeze(dim=0) when label is tensor in callable transform.
    """

    urls = {
        'train image': 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip',
        'train mask' : 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip',
        'val image'  : 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip',
        'val mask'   : 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip',
        'test image' : 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip',
        'test mask'  : 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip'
    }

    md5s = {
        'train image': 'a14a7e622c67a358797ae59abb8a0b0c',
        'train mask' : '77fdbeb6fbec4139937224416b250f4c',
        'val image'  : '8d6419d942112f709894c0d82f6c9038',
        'val mask'   : '64d3e68fa2deeb8a5e89aa8dec2efd44',
        'test image' : '5f6a0b5e1f2972bd1f5ea02680489f09',
        'test mask'  : 'b1742de6bd257faca3b2b21a4aa3b781'
    }

    def __init__(
        self,
        root     : str,
        mode     : str = 'train',
        transform: Optional[Callable] = None,
        download : bool = False,
    ) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\','test' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()

        if mode not in ['train', 'val', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        self.root      = root
        self.mode      = mode
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, ISIC_2017_Task1.urls[f'{self.mode} image'].split('/')[-1])
        if os.path.exists(file_name[:-4]):
            print(f'{self.mode} images are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading {self.mode} images ...')
                    download_data_from_url(ISIC_2017_Task1.urls[f'{self.mode} image'], self.root)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, ISIC_2017_Task1.md5s[f'{self.mode} image']):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)
        file_name = os.path.join(self.root, ISIC_2017_Task1.urls[f'{self.mode} mask'].split('/')[-1])
        if os.path.exists(file_name[:-4]):
            print(f'{self.mode} masks are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading {self.mode} masks ...')
                    download_data_from_url(ISIC_2017_Task1.urls[f'{self.mode} mask'], self.root)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, ISIC_2017_Task1.md5s[f'{self.mode} mask']):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        images_path = os.path.join(self.root, ISIC_2017_Task1.urls[f'{self.mode} image'].split('/')[-1].replace('.zip', ''))
        masks_path  = os.path.join(self.root, ISIC_2017_Task1.urls[f'{self.mode} mask'].split('/')[-1].replace('.zip', ''))
        images      = glob(os.path.join(images_path, '*.jpg'))
        self.data   = [(_, _.replace(images_path, masks_path)[:-4] + '_segmentation.png') for _ in images]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        imgs = [Image.open(_) for _ in self.data[index]]
        return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)


class ISIC_2018_Task1(Dataset):
    """
    Use the data on ISIC 2018.Abot more:https://challenge2018.isic-archive.com/task1/ \\
    The size of train images:576<=width<=6748,540<=height<=4499.\\
    There are 2594 image in train data and 100 in validation data and 32 in test data.\\
    The shape of mask image is H\times W and pixels value are 255 for target area and 0 for background.\\
    All image in test data are not have mask and some of they are bad. \\
    If transform is None,you will get size [3, H, W],[1, H, W] for tensors of image and label. \\
    You may need run label = label[0] or label=label.squeeze(dim=0) when label is tensor in callable transform.
    """

    urls = {
        'train image': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip',
        'train mask' : 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip',
        'val image'  : 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip',
        'val mask'   : 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip',
        'test image' : 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip'
    }

    md5s = {
        'train image': '8b5be801f37b58ccf533df2928a5906b',
        'train mask' : 'ee5e5db7771d48fa2613abc7cb5c24e2',
        'val image'  : 'd8f15a08981c8d6c0646b5e3228ef20d',
        'val mask'   : '2323d7ed9350de0a463d529ef45a62f4',
        'test image' : '0a9ba80e1aee58716fbfcdde7131c45a'
        }

    def __init__(
        self,
        root     : str,
        mode     : str = 'train',
        transform: Optional[Callable] = None,
        download : bool = False,
    ) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\','test' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train and val mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        if mode not in ['train', 'val', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        self.root      = root
        self.mode      = mode
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')

        self._load_data()

    def _download(self) -> None:
        if self.mode in ['train', 'val']:
            file_name = os.path.join(self.root, ISIC_2018_Task1.urls[f'{self.mode} image'].split('/')[-1])
            if os.path.exists(file_name[:-4]):
                print(f'{self.mode} images are already exist,cancel download.')
            else:
                if not os.path.exists(file_name):
                    while True:
                        print(f'Downloading {self.mode} images ...')
                        download_data_from_url(ISIC_2018_Task1.urls[f'{self.mode} image'], self.root)
                        print('The file is being verified using MD5 ...')
                        if chack_file_with_md5(file_name, ISIC_2018_Task1.md5s[f'{self.mode} image']):
                            break
                        else:
                            print('Error with file,trying download again.')
                print('Unzipping ...')
                unzip_file(file_name, self.root)
                os.remove(file_name)
            file_name = os.path.join(self.root, ISIC_2018_Task1.urls[f'{self.mode} mask'].split('/')[-1])
            if os.path.exists(file_name[:-4]):
                print(f'{self.mode} masks are already exist,cancel download.')
            else:
                if not os.path.exists(file_name):
                    while True:
                        print(f'downloading {self.mode} masks ...')
                        download_data_from_url(ISIC_2018_Task1.urls[f'{self.mode} mask'], self.root)
                        print('The file is being verified using MD5 ...')
                        if chack_file_with_md5(file_name, ISIC_2018_Task1.md5s[f'{self.mode} mask']):
                            break
                        else:
                            print('error with file,trying download again.')
                print('unzipping ...')
                unzip_file(file_name, self.root)
                os.remove(file_name)
        else:
            file_name = os.path.join(self.root, ISIC_2018_Task1.urls[f'{self.mode} image'].split('/')[-1])
            if os.path.exists(file_name[:-4]):
                print(f'{self.mode} images are already exist,cancel download.')
            else:
                if not os.path.exists(file_name):
                    while True:
                        print(f'downloading {self.mode} images ...')
                        download_data_from_url(ISIC_2018_Task1.urls[f'{self.mode} image'], self.root)
                        print('The file is being verified using MD5 ...')
                        if chack_file_with_md5(file_name, ISIC_2018_Task1.md5s[f'{self.mode} image']):
                            break
                        else:
                            print('error with file,trying download again.')
                print('unzipping ...')
                unzip_file(file_name, self.root)
                os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode in ['train', 'val']: 
            images_path = os.path.join(self.root, ISIC_2018_Task1.urls[f'{self.mode} image'].split('/')[-1].replace('.zip', ''))
            masks_path  = os.path.join(self.root, ISIC_2018_Task1.urls[f'{self.mode} mask'].split('/')[-1].replace('.zip', ''))
            images      = glob(os.path.join(images_path, '*.jpg'))
            self.data   = [(_, _.replace(images_path, masks_path)[:-4] + '_segmentation.png') for _ in images]
        else: 
            images_path = os.path.join(self.root, ISIC_2018_Task1.urls[f'{self.mode} image'].split('/')[-1].replace('.zip', ''))
            self.data   = glob(os.path.join(images_path, '*.jpg'))

    def __getitem__(self, index) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.mode in ['train', 'val']:
            imgs = [Image.open(_) for _ in self.data[index]]
            return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)
        else: 
            img = Image.open(self.data[index])
            return TF.to_tensor(img) if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class KITS_2019(Dataset):
    """
    Use the data of KITS 2019.\\
    About more:https://github.com/neheller/kits19 or https://kits19.grand-challenge.org/ \\
    There are 210 data in this dataset.
    """

    imaging_url      = "https://kits19.sfo2.digitaloceanspaces.com/master_{:05d}.nii.gz"
    segmentation_url = 'https://github.com/neheller/kits19'
    num_data         = 210

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root:The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data if download is True.
        """

        super().__init__()
        self.root      = root
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        if os.path.exists(os.path.join(self.root, 'case_00209/imaging.nii.gz')):
            print('Datas are already exist,cancel download.')
            return
        print('Downloading masks...')
        import git
        with git.Repo.init(path=self.root) as repo:
            repo.clone_from(KITS_2019.segmentation_url, os.path.join(self.root, 'kits19'))
        for i in range(KITS_2019.num_data):
            print(f'Downloading {i+1}-th image')
            save_path = os.path.join(self.root, f'case_{i:05d}')
            shutil.move(os.path.join(self.root, 'kits19', 'data', f'case_{i:05d}'), self.root)
            while not os.path.exists(os.path.join(save_path, 'imaging.nii.gz')):
                download_data_from_url(KITS_2019.imaging_url.format(i), save_path)
                os.rename(os.path.join(save_path, f'master_{i:05d}.nii.gz'), os.path.join(save_path, f'imaging.nii.gz'))
        shutil.rmtree(os.path.join(self.root, 'kits19'))
        shutil.rmtree(os.path.join(self.root, '.git'))

    def _load_data(self) -> None:
        imgs      = glob(os.path.join(self.root, '*/imaging.nii.gz'))
        self.data = [(_, _[:-14] + 'segmentation.nii.gz') for _ in imgs]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
        return tuple(imgs) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)


class Kvasir_SEG(Dataset):
    """
    Use the data of Kvasir-SEG dataset.\\
    About more:https://datasets.simula.no/kvasir-seg/ \\
    The size of images:332<=width<=1920,352<=height<=1072.\\
    There are 1000 images in this dataset.\\
    The shape of mask image is H\times W and pixels value are 255 for target area and 0 for background. \\
    If transform is None,you will get size [3, H, W],[3, H, W] for tensors of image and label. \\
    You may need run label = label[0] when label is tensor in callable transform.
    """

    data_url = 'https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip'

    md5 = '6323d9094df93b35d43069a566ee1ca3'

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root:The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        self.root      = root
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, Kvasir_SEG.data_url.split('/')[-1])
        if os.path.exists(file_name[:-4]):
            print(f'Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print('Downloading data ...')
                    download_data_from_url(Kvasir_SEG.data_url, self.root)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, Kvasir_SEG.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        image_path = os.path.join(self.root, 'Kvasir-SEG/images')
        mask_path  = os.path.join(self.root, 'Kvasir-SEG/masks')
        masks      = glob(os.path.join(mask_path, '*.jpg'))
        self.data  = [(os.path.join(image_path, mask.split('/')[-1]), mask) for mask in masks]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        imgs = [Image.open(_) for _ in self.data[index]]
        return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)


class CVC_ClinicDB(Dataset):
    """
    Use the polyp data of CVC ClinicDB dataset.About more:https://polyp.grand-challenge.org/CVCClinicDB/ \\
    The size is (384,288) for each image. \\
    There are 612 image in this dataset. \\
    The shape of mask image is H\times W and pixels value are 255 for target area and 0 for background. \\
    If transform is None,you will get size [3,384,288],[3,384,288] for tensors of image and label. \\
    You may need run label = label[0] or label = label.squeeze(dim=0) when label is tensor in callabel transform.
    """

    data_url = 'balraj98/cvcclinicdb'

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root:The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        self.root      = root
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        if os.path.exists(os.path.join(self.root, 'PNG')):
            print('Datas are already exist,cancel download.')
        else:
            file_name = os.path.join(self.root, 'cvcclinicdb.zip')
            if not os.path.exists(file_name):
                print('Downloading data ...')
                download_data_from_kaggle_dataset(self.data_url, self.root)
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        image_path = os.path.join(self.root, 'PNG/Original')
        mask_path  = os.path.join(self.root, 'PNG/Ground Truth')
        masks      = glob(os.path.join(mask_path, '*'))
        self.data  = [(os.path.join(image_path, mask.split('/')[-1]), mask) for mask in masks]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        imgs = [Image.open(_) for _ in self.data[index]]
        return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)


class ETIS_Larib(Dataset):
    """
    Use the polyp data of ETIS_Larib dataset.\\
    The size is (966,1225) for each image. \\
    There are 196 image in this dataset. \\
    The shape of mask image is H\times W and pixels value are 255 for target area and 0 for background. \\
    If transform is None,you will get size [3, 966, 1225],[1, 966, 1225] for tensors of image and label. \\
    You may need run label = label[0] or label = label.squeeze(dim=0) when label is tensor in callabel transform.
    """

    data_url = 'https://drive.google.com/uc?id=1skc7wtyX8a1Vy2LdA_0BOZ7wvdc_Ytto'

    md5 = 'ae02b3fc04a1a68fd7440d322a5cdb4b'

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root:The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        self.root      = root
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'ETIS-LaribPolypDB.zip')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(ETIS_Larib.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, ETIS_Larib.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        image_path = os.path.join(self.root, 'ETIS-LaribPolypDB/ETIS-LaribPolypDB')
        mask_path  = os.path.join(self.root, 'ETIS-LaribPolypDB/Ground Truth')
        masks      = glob(os.path.join(mask_path, '*'))
        self.data  = [(os.path.join(image_path, mask.split('/')[-1][1:]), mask) for mask in masks]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        imgs = [Image.open(_) for _ in self.data[index]]
        return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)


class MoNuSeg(Dataset):
    """
    Use the data of MoNuSeg dataset.About more:https://monuseg.grand-challenge.org/ \\
    The size is (1000,1000) for each image. \\
    There are 37 image in train data and 14 in test data. \\
    The shape of mask image is H\times W and pixels value are 255 for target area and 0 for background. \\
    If transform is None,you will get size [3,1000,1000],[1,1000,1000] for tensors of image and label. \\
    You may need run label = label[0] or label = label.squeeze(dim=0) when label is tensor in callable transform.
    """

    urls = {
        'train data': 'https://drive.google.com/uc?id=1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA', 
        'test data' : 'https://drive.google.com/uc?id=1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw'
        }

    md5s = {
        'train data': 'dfe849bcb72daec2a78ae5ba0ad10ade', 
        'test data' : '0721da0cdbd267299e0463419d70e1a5'
        }

    file_names = {
        'train data': 'MoNuSeg 2018 Training Data.zip', 
        'test data' : 'MoNuSegTestData.zip'
        }

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root      = root
        self.mode      = mode
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, MoNuSeg.file_names[f'{self.mode} data'])
        if os.path.exists(file_name[:-4]):
            print(f'{self.mode} data are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MoNuSeg.urls[f'{self.mode} data'], file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MoNuSeg.md5s[f'{self.mode} data']):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, MoNuSeg.file_names[f'{self.mode} data'].replace('.zip', ''), 'Tissue Images')
            mask_path = os.path.join(self.root, MoNuSeg.file_names[f'{self.mode} data'].replace('.zip', ''), 'Annotations')
            masks = glob(os.path.join(mask_path, '*'))
            self.data = [(os.path.join(image_path, mask.split('/')[-1].replace('xml', 'tif')), mask) for mask in masks]
        else:
            image_path = os.path.join(self.root, MoNuSeg.file_names[f'{self.mode} data'].replace('.zip', ''))
            mask_path = os.path.join(self.root, MoNuSeg.file_names[f'{self.mode} data'].replace('.zip', ''))
            masks = glob(os.path.join(mask_path, '*.xml'))
            self.data = [(os.path.join(image_path, mask.split('/')[-1].replace('xml', 'tif')), mask) for mask in masks]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        img, mask = self.data[index]
        img = Image.open(img)
        mask = xml_to_binary_mask(img.width, img.height, mask)
        if self.transform is not None:
            return self.transform(img, mask)
        else:
            return TF.to_tensor(img), TF.to_tensor(mask)

    def __len__(self) -> int:
        return len(self.data)


class MSD_Brain(Dataset):
    """
    Use the Brain dataset in Medical Segmentation Decathlon. \\
    About more:http://medicaldecathlon.com/ \\
    There are 484 images in train dataset and 266 in test dataset.\\
    Every image or label has the shape : (240, 240, 155, 4). \\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU'
    md5 = '240a19d752f0d9e9101544901065d872'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task01_BrainTumour.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_Brain.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_Brain.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task01_BrainTumour/imagesTr')
            mask_path = os.path.join(self.root, 'Task01_BrainTumour/labelsTr')
            masks = glob(os.path.join(mask_path, 'BRATS*'))
            self.data = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task01_BrainTumour', 'imagesTs', 'BRATS*'))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class MSD_Heart(Dataset):
    """
    Use the Heart dataset in Medical Segmentation Decathlon. \\
    About more:http://medicaldecathlon.com/ \\
    There are 20 images in train dataset and 10 in test dataset.\\
    Every image or label has the shape  (320, 320, about 100) such  (320, 320, 110).\\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY'
    md5 = '06ee59366e1e5124267b774dbd654057'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task02_Heart.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_Heart.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_Heart.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task02_Heart/imagesTr')
            mask_path = os.path.join(self.root, 'Task02_Heart/labelsTr')
            masks = glob(os.path.join(mask_path, 'la*'))
            self.data = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task02_Heart', 'imagesTs', 'la*'))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class MSD_Liver(Dataset):
    """
    Use the Liver dataset in Medical Segmentation Decathlon.\\
    About more:http://medicaldecathlon.com/ \\
    There are 130 images in train dataset and 70 in test dataset.\\
    Every image or label has the shape (512, 512, any) such as (512, 512, 610). \\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu'
    md5 = 'a90ec6c4aa7f6a3d087205e23d4e6397'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task03_Liver.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_Liver.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_Liver.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task03_Liver/imagesTr')
            mask_path = os.path.join(self.root, 'Task03_Liver/labelsTr')
            masks = glob(os.path.join(mask_path, 'liver*'))
            self.data = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task03_Liver', 'imagesTs', 'liver*'))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class MSD_Hippocampus(Dataset):
    """
    Use the Hippocampus dataset in Medical Segmentation Decathlon.\\
    About more:http://medicaldecathlon.com/ \\
    There are 260 images in train dataset and 130 in test dataset.\\
    Every image or label has the shape (any,any,any) such as (37, 56, 36).\\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C'
    
    md5 = '9d24dba78a72977dbd1d2e110310f31b'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task04_Hippocampus.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_Hippocampus.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_Hippocampus.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task04_Hippocampus/imagesTr')
            mask_path = os.path.join(self.root, 'Task04_Hippocampus/labelsTr')
            masks = glob(os.path.join(mask_path, 'hippocampus*'))
            self.data = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task04_Hippocampus', 'imagesTs', 'hippocampus*'))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class MSD_Prostate(Dataset):
    """
    Use the Prostate dataset in Medical Segmentation Decathlon. \\
    About more:http://medicaldecathlon.com/ \\
    There are 32 images in train dataset and 16 in test dataset. \\
    Every image or label has the shape (320, 320, any, 2) such as (320, 320, 20, 2). \\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a'
    md5 = '35138f08b1efaef89d7424d2bcc928db'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task05_Prostate.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_Prostate.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_Prostate.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task05_Prostate/imagesTr')
            mask_path = os.path.join(self.root, 'Task05_Prostate/labelsTr')
            masks = glob(os.path.join(mask_path, '*'))
            self.data = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task05_Prostate', 'imagesTs', '*'))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class MSD_Lung(Dataset):
    """
    Use the Lung dataset in Medical Segmentation Decathlon. \\
    About more:http://medicaldecathlon.com/ \\
    There are 63 images in train dataset and 32 in test dataset. \\
    Every image or label has the shape (512, 512, any) such as (512, 512, 241). \\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi'
    md5 = '8afd997733c7fc0432f71255ba4e52dc'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task06_Lung.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_Lung.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_Lung.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task06_Lung/imagesTr')
            mask_path = os.path.join(self.root, 'Task06_Lung/labelsTr')
            masks = glob(os.path.join(mask_path, '*'))
            self.data = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task06_Lung', 'imagesTs', '*'))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class MSD_Pancreas(Dataset):
    """
    Use the Lung dataset in Medical Segmentation Decathlon. \\
    About more:http://medicaldecathlon.com/ \\
    There are 281 images in train dataset and 139 in test dataset. \\
    Every image or label has the shape (512, 512, any) such as (512, 512, 101). \\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL'
    md5 = '4f7080cfca169fa8066d17ce6eb061e4'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task07_Pancreas.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_Pancreas.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_Pancreas.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task07_Pancreas/imagesTr')
            mask_path = os.path.join(self.root, 'Task07_Pancreas/labelsTr')
            masks = glob(os.path.join(mask_path, 'pancreas*'))
            self.data = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task07_Pancreas', 'imagesTs', 'pancreas*'))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class MSD_Spleen(Dataset):
    """
    Use the Spleen dataset in Medical Segmentation Decathlon. \\
    About more:http://medicaldecathlon.com/ \\
    There are 41 images in train dataset and 20 in test dataset. \\
    Every image or label has the shape (512, 512, any) such as (512, 512, 112). \\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE'
    md5 = '410d4a301da4e5b2f6f86ec3ddba524e'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)

            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task09_Spleen.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_Spleen.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_Spleen.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task09_Spleen/imagesTr')
            mask_path = os.path.join(self.root, 'Task09_Spleen/labelsTr')
            masks = glob(os.path.join(mask_path, 'spleen*'))
            self.data = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task09_Spleen', 'imagesTs', 'spleen*'))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class MSD_HepaticVessel(Dataset):
    """
    Use the Hepatic Vessel dataset in Medical Segmentation Decathlon. \\
    About more:http://medicaldecathlon.com/ \\
    There are 303 images in train dataset and 140 in test dataset. \\
    Every image or label has the shape (512, 512, any) such as (512, 512, 45). \\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS'

    md5 = '641d79e80ec66453921d997fbf12a29c'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)

            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task08_HepaticVessel.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_HepaticVessel.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_HepaticVessel.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task08_HepaticVessel/imagesTr')
            mask_path = os.path.join(self.root, 'Task08_HepaticVessel/labelsTr')
            masks = glob(os.path.join(mask_path, 'hepaticvessel*'))
            self.data = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task08_HepaticVessel', 'imagesTs', 'hepaticvessel*'))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class MSD_Colon(Dataset):
    """
    Use the Colon dataset in Medical Segmentation Decathlon. \\
    About more:http://medicaldecathlon.com/ \\
    There are 126 images in train dataset and 64 in test dataset. \\
    Every image or label has the shape (512, 512, any) such as (512, 512, 87). \\
    Only train dataset have lables.
    """

    data_url = 'https://drive.google.com/uc?id=1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y'

    md5 = 'bad7a188931dc2f6acf72b08eb6202d0'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label) when train mode and get (image) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root      = root
        self.mode      = mode
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)

            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'Task10_Colon.tar')
        if os.path.exists(file_name[:-4]):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_google_drive(MSD_Colon.data_url, file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MSD_Colon.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Untaring ...')
            untar_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'Task10_Colon/imagesTr')
            mask_path  = os.path.join(self.root, 'Task10_Colon/labelsTr')
            masks      = glob(os.path.join(mask_path, 'colon*'))
            self.data  = [(mask, os.path.join(image_path, mask.split('/')[-1])) for mask in masks]
        else:
            self.data = glob(os.path.join(self.root, 'Task10_Colon', 'imagesTs', 'colon*'))

    def __getitem__(self, index) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.mode == 'train':
            imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
            return tuple(imgs) if self.transform is None else self.transform(*imgs)
        else:
            img = to_tensor_3D(np.array(nib.load(self.data[index]).get_fdata()))
            return img if self.transform is None else self.transform(img)

    def __len__(self) -> int:
        return len(self.data)


class DRIVE(Dataset):
    """
    DRIVE dataset,about more:https://drive.grand-challenge.org/ \\
    Go to https://drive.grand-challenge.org/ to register and join the challenge to get the dataset. \\
    The root path should be:\\
        root/training/images \\
        root/training/mask \\ 
        root/training/1st_manual \\
        root/test/images \\
        root/test/mask \\
    The images are rgb format with size 565\times 584,but label and mask only have one channel. \\
    If transform is None , you will get size [3, 584, 565],[1, 584, 565],[1, 584, 565] for tensors of image,mask and label. \\
    You may need run x=x.squeeze(dim=0) for mask and label in callable transform.
    """

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, mask, label) when train mode and get (image, mask) when test mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.data = []

        self._load_data()

    def _load_data(self) -> None:
        if self.mode == 'train':
            images = os.path.join(self.root, 'training', 'images')
            masks = os.path.join(self.root, 'training', 'mask')
            labels = os.path.join(self.root, 'training', '1st_manual')
            self.data = [(_, _.replace(images, masks)[:-4] + '_mask.gif', _.replace(images, labels)[:-12] + 'manual1.gif') for _ in glob(os.path.join(images, '*.tif'))]
        else:
            images = os.path.join(self.root, 'test', 'images')
            masks = os.path.join(self.root, 'test', 'mask')
            self.data = [(_, _.replace(images, masks)[:-4] + '_mask.gif') for _ in glob(os.path.join(images, '*.tif'))]

    def __getitem__(self, index) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        if self.mode == 'train':
            img, mask, label = self.data[index]
            img, mask, label = Image.open(img), Image.open(mask), Image.open(label)
            if self.transform is not None:
                return self.transform(img, mask, label)
            else:
                return TF.to_tensor(img), TF.to_tensor(mask), TF.to_tensor(label)
        else:
            img, mask = self.data[index]
            img, mask = Image.open(img), Image.open(mask)
            if self.transform is not None:
                return self.transform(img, mask)
            else:
                return TF.to_tensor(img), TF.to_tensor(mask)

    def __len__(self) -> int:
        return len(self.data)


class CHASE_DB1(Dataset):
    """
    Use the eye fundus data of CHASE_DB1. \\
    About more:https://www.idiap.ch/software/bob/docs/bob/bob.db.chasedb1/master/index.html \\
    There 28 (14 pair) images in this dataset. \\
    Each image has 2 different labels.\\
    The size is (960,999) for each image. \\
    The shape of mask image is H\times W and pixels value are 255 for target area and 0 for background. \\
    If transform is None,you will get size [3, 960, 999],[1, 960, 999],[1, 960, 999] for tensors of image,label1 and label2. \\
    You may need run x = x[0] or x=x.squeeze(dim=0) in callable transform when labels are tensor.
    """

    data_url = 'https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip'

    md5 = 'd9e47c4bac125b29996fae4380a68db1'

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label1, label2).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        self.root = root
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'CHASEDB1.zip')
        if os.path.exists(os.path.join(self.root, 'Image_01L.jpg')):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_url(CHASE_DB1.data_url, self.root)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, CHASE_DB1.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unziping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        images = glob(os.path.join(self.root, '*.jpg'))
        self.data = [(_, _.replace('.jpg', '_1stHO.png'), _.replace('.jpg', '_2ndHO.png')) for _ in images]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        img, mask1, mask2 = self.data[index]
        img, mask1, mask2 = Image.open(img), Image.open(mask1), Image.open(mask2)
        if self.transform is not None:
            return self.transform(img, mask1, mask2)
        else:
            return TF.to_tensor(img), TF.to_tensor(mask1), TF.to_tensor(mask2)

    def __len__(self) -> int:
        return len(self.data)


class GlaS(Dataset):
    """
    The gland segmentation dataset GlaS. \\
    About more:https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/ \\
    There are 85 images in train data and 80 in test data. \\
    The size of images:567<=width<=775 and 430<=height<=522. \\
    If transform is None,you will get size [3, H, W],[1, H, W] for tensors of image and label. \\
    You may need run label = label[0] or label=label.squeeze(dim=0) when label is tensor in callable transform.
    """

    data_url = 'https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip'

    md5 = '495b2a9f3d694545fbec06673fb3f40f'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'warwick_qu_dataset_released_2016_07_08.zip')
        if os.path.exists(os.path.join(self.root, 'Warwick QU Dataset (Released 2016_07_08)')):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_url(GlaS.data_url, self.root)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, GlaS.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unziping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            masks = glob(os.path.join(self.root, 'Warwick QU Dataset (Released 2016_07_08)', 'train*anno.bmp'))
            self.data = [(_.replace('_anno', ''), _) for _ in masks]
        else:
            masks = glob(os.path.join(self.root, 'Warwick QU Dataset (Released 2016_07_08)', 'test*anno.bmp'))
            self.data = [(_.replace('_anno', ''), _) for _ in masks]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        img, mask = self.data[index]
        img, mask = Image.open(img), Image.open(mask)
        if self.transform is not None:
            return self.transform(img, mask)
        else:
            return TF.to_tensor(img), TF.to_tensor(mask)

    def __len__(self) -> int:
        return len(self.data)


class CoNSeP(Dataset):
    """
    The H&E image segmentation dataset CoNSeP. \
    About more:https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/ \
    There 27 images in train data and 14 in test data. \
    The size of each image is (1000\times 1000). \
    If transform is None,you will get size [3,1000,1000],[1,1000,1000] for tensors of image and label. \
    You may need run label = label[0] or label = label.squeeze(dim=0) when label is tensor in callable transform.
    """

    data_url = 'https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep.zip'

    md5 = '54dc1511bdad92186490e667b1156445'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'consep.zip')
        if os.path.exists(os.path.join(self.root, 'CoNSeP')):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_url(CoNSeP.data_url, self.root)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, CoNSeP.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unziping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            images = glob(os.path.join(self.root, 'CoNSeP', 'Train', 'Images', '*'))
            masks = os.path.join(self.root, 'CoNSeP', 'Train', 'Labels')
            self.data = [(_, os.path.join(masks, os.path.basename(_).replace('png', 'mat'))) for _ in images]
        else:
            images = glob(os.path.join(self.root, 'CoNSeP', 'Test', 'Images', '*'))
            masks = os.path.join(self.root, 'CoNSeP', 'Test', 'Labels')
            self.data = [(_, os.path.join(masks, os.path.basename(_).replace('png', 'mat'))) for _ in images]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        from scipy.io import loadmat
        img, mask = self.data[index]
        img, mask = Image.open(img), Image.fromarray(loadmat(mask)['type_map'])
        if self.transform is not None:
            return self.transform(img, mask)
        else:
            return TF.to_tensor(img), TF.to_tensor(mask)

    def __len__(self) -> int:
        return len(self.data)


class PanNuke(Dataset):
    """
    Use the H&E images of PanNuke. \\
    About more:https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/ \\
    There are 7901 images in this dataset. \\
    The size of each image is (256\times 256). \\
    If transform is None,you will get size [3,256,256],[any,256,256] for tensors of images and labels. \\
    You may need run label = label.sum(dim=0) after to_tensor(label) in callable transform.'
    """
    urls = {
        'fold_1': 'https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip',
        'fold_2': 'https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip',
        'fold_3': 'https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip'
        }

    md5s = {
        'fold_1': 'e1b16ef84db3e3368d9f5bd4e61ed65c', 
        'fold_2': 'f1839d332c4b8e12c7c01882020ac457', 
        'fold_3': '441950a966ee73d6e9780196815a0b20'
        }

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root:The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        self.root = root
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        if os.path.exists(os.path.join(self.root, 'images')):
            print('Data are already exist,cancel download.')
            return
        for key in PanNuke.urls.keys():
            file_name = os.path.join(self.root, os.path.basename(PanNuke.urls[key]))
            if os.path.exists(file_name[:-4]):
                print(f'{key} are already exist,cancel download.')
            else:
                if not os.path.exists(file_name):
                    while True:
                        print(f'Downloading {key} ...')
                        download_data_from_url(PanNuke.urls[key], self.root)
                        print('The file is being verified using MD5 ...')
                        if chack_file_with_md5(file_name, PanNuke.md5s[key]):
                            break
                        else:
                            print('Error with file,trying download again.')
                print('Unziping ...')
                unzip_file(file_name, self.root)
                os.remove(file_name)
        print('The data files are too big,spliting to small file.')
        index = 0
        os.makedirs(os.path.join(self.root, 'images'))
        os.makedirs(os.path.join(self.root, 'masks'))
        for i in range(1, 4):
            images_path = os.path.join(self.root, f'Fold {i}', 'images', f'fold{i}', 'images.npy')
            images = np.load(images_path)
            for j, image in enumerate(images):
                img = Image.fromarray(np.uint8(image))
                img.save(os.path.join(self.root, 'images', f'{str(j+index).zfill(5)}.png'))
            del images
            masks_path = os.path.join(self.root, f'Fold {i}', 'masks', f'fold{i}', 'masks.npy')
            masks = np.load(masks_path)
            for j, mask in enumerate(masks):
                np.save(os.path.join(self.root, 'masks', f'{str(j+index).zfill(5)}.npy'), mask)
            index += len(masks)
            shutil.rmtree(os.path.join(self.root, f'Fold {i}'))

    def _load_data(self) -> None:
        images = glob(os.path.join(self.root, 'images/*.png'))
        masks_path = os.path.join(self.root, 'masks')
        self.data = [(_, os.path.join(masks_path, os.path.basename(_).replace('png', 'npy'))) for _ in images]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        img, mask = self.data[index]
        img, mask = Image.open(img), np.load(mask)
        return (TF.to_tensor(img), TF.to_tensor(mask)) if self.transform is None else self.transform(img, mask)

    def __len__(self) -> int:
        return len(self.data)


class DSB2018(Dataset):
    """
    Use the data of 2018 Data Science Bowl on kaggle. \\
    About more:https://www.kaggle.com/c/data-science-bowl-2018/data \\
    There are 670 images in train data and 3084 in test data. \\
    The size of images:161<=width<=1388 and 205<=height<=1040. \\
    There are not labels for test data. \\
    If transform is None,you will get size [4,256,256],[1,256,256] for tensors of images and labels. \\
    You may need run label = label[0] or label = label.squeeze(dim=0) when label is tensor.'
    """

    data_url = 'data-science-bowl-2018'

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root:The path dataset exist or save.
            mode:\'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image,label) when mode is train and get (image) when anther mode.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        if os.path.exists(os.path.join(self.root, self.mode)):
            print('Datas are already exist,cancel download.')
        else:
            file_name = os.path.join(self.root, 'data-science-bowl-2018.zip')
            if not os.path.exists(file_name):
                print(f'Downloading data...')
                download_data_from_kaggle_competition(DSB2018.data_url, self.root)
            print('Unzipping...')
            unzip_file(os.path.join(file_name), self.root)
            os.remove(file_name)
            os.makedirs(os.path.join(self.root, 'train'))
            unzip_file(os.path.join(self.root, 'stage1_train.zip'), os.path.join(self.root, 'train'))
            os.remove(os.path.join(self.root, 'stage1_train.zip'))
            os.makedirs(os.path.join(self.root, 'test'))
            unzip_file(os.path.join(self.root, 'stage1_test.zip'), os.path.join(self.root, 'test'))
            os.remove(os.path.join(self.root, 'stage1_test.zip'))
            unzip_file(os.path.join(self.root, 'stage2_test_final.zip'), os.path.join(self.root, 'test'))
            os.remove(os.path.join(self.root, 'stage2_test_final.zip'))

    def _load_data(self) -> None:
        if self.mode == 'train':
            images = glob(os.path.join(self.root, 'train', '*', 'images', '*'))
            for path in images:
                masks = glob(os.path.join(*(path.split('/')[:-2]), 'masks', '*'))
                self.data.append((path, masks))
        else:
            images = glob(os.path.join(self.root, 'test', '*', 'images', '*'))
            self.data = images

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == 'train':
            img, mask = self.data[index]
            img = Image.open(img)
            mask = [(i + 1) * np.array(Image.open(_), dtype=np.int32) / 255 for i, _ in enumerate(mask)]
            mask = np.array(mask).sum(axis=0)
            if self.transform is not None:
                return self.transform(img, mask)
            else:
                return TF.to_tensor(img), TF.to_tensor(mask)
        else:
            img = self.data[index]
            img = Image.open(img)
            if self.transform is not None:
                return self.transform(img)
            else:
                return TF.to_tensor(img)

    def __len__(self) -> int:
        return len(self.data)


class BraTS(Dataset):
    """
    Use the data of Multimodal Brain Tumor Segmentation Challenge 2018,2019 and 2020. \\
    About more: https://www.kaggle.com/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015 and https://www.med.upenn.edu/sbia/brats2018/data.html and  https://www.med.upenn.edu/cbica/brats2019/data.html and https://www.med.upenn.edu/cbica/brats2020/data.html \\
    Each data contains 4 images and a label. \\
    |version|images count| \\
    |-------|------------| \\
    |2018   |1425        | \\
    |2019   |335         | \\
    |2020   |369         |
    """

    data_url = 'andrewmvd/brain-tumor-segmentation-in-mri-brats-2015'

    dirs = {'2018': 'MICCAI_BraTS_2018_Data_Training', '2019': 'MICCAI_BraTS_2019_Data_Training', '2020': 'MICCAI_BraTS2020_TrainingData'}

    def __init__(self, root: str, version: str = '2018', transform: Optional[Callable] = None, download: bool = False):
        """
        Args::
            root:The path dataset exist or save.
            version: '2018','2019' or '2020'
            transform: A callable function or object.It will get parameter (flair,t1,t1ce,t2,label).
        """

        assert version in BraTS.dirs.keys()

        super().__init__()
        self.root      = root
        self.version   = version
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)

            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        if os.path.exists(os.path.join(self.root, BraTS.dirs[self.version])):
            print('Datas are already exist,cancel download.')
        else:
            file_name = os.path.join(self.root, 'brain-tumor-segmentation-in-mri-brats-2015.zip')
            if not os.path.exists(file_name):
                print(f'Downloading data...')
                download_data_from_kaggle_dataset(BraTS.data_url, self.root)
            print('Unzipping...')
            unzip_file(os.path.join(file_name), self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        if self.version == '2018' or self.version == '2019':
            masks = glob(os.path.join(self.root, BraTS.dirs[self.version], '*/*/*seg.nii'))
        else:
            masks = glob(os.path.join(self.root, BraTS.dirs[self.version], '*/*seg.nii'))
        self.data = [(_[:-7] + 'flair.nii', _[:-7] + 't1.nii', _[:-7] + 't1ce.nii', _[:-7] + 't2.nii', _) for _ in masks]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
        return tuple(imgs) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)


class TNBC(Dataset):
    """
    Nuclei Segmentation use Triple negative breast cancer images. \\
    About moer:https://zenodo.org/record/1175282#.YZ8N4ZFBxhF \\
    There are 50 images in this dataset,and 512\times 512 pixels for each image and label. \\
    If transform is None,you will get size [4, 512, 512],[1, 512, 512] for tensors of images and labels. \\
    You may need run label = label[0] or label=label.squeeze(dim=0) when label is tensor in callable transform.
    """

    data_url = 'https://zenodo.org/record/2579118/files/TNBC_NucleiSegmentation.zip'
    md5 = '1605712a752b201b57eacc8f866adb4f'

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root:The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        self.root = root
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, 'TNBC_NucleiSegmentation.zip')
        if os.path.exists(os.path.join(self.root, 'TNBC_dataset')):
            print('Datas are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print(f'Downloading data ...')
                    download_data_from_url(TNBC.data_url, self.root)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, TNBC.md5):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unziping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        images    = glob(os.path.join(self.root, 'TNBC_dataset', 'Slide*', '*'))
        self.data = [(_, _.replace('Slide', 'GT')) for _ in images]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        imgs = [Image.open(_) for _ in self.data[index]]
        return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)


class MoNuSAC(Dataset):
    """
    Use the data of MoNuSAC dataset.About more:https://monusac-2020.grand-challenge.org/ \\
    The size of images:35<=width<=2162  and 33<=height<=2500. \\
    There are 209 image in train data and 82 in test data. \\
    The shape of mask image is H\times W and pixels value are 255 for target area and 0 for background. \\
    If transform is None,you will get size [4,any,any],[1,any,any] for tensors of images and labels. \\
    You may need run label = label[0] or label = label.squeeze(dim=0) when label is tensor in callable transform.
    """

    urls = {
        'train data': 'https://drive.google.com/uc?id=1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq', 
        'test data': 'https://drive.google.com/uc?id=1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ'
        }

    md5s = {
        'train data': '9237b6b6e2c00b4ebbb23c311a3f1704', 
        'test data': '857011bc9aecf1b2073733826bae7749'
        }

    file_names = {
        'train data': 'MoNuSAC_images_and_annotations.zip', 
        'test data': 'MoNuSAC Testing Data and Annotations.zip'
        }

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        file_name = os.path.join(self.root, MoNuSAC.file_names[f'{self.mode} data'])
        if os.path.exists(file_name[:-4]):
            print(f'{self.mode} data are already exist,cancel download.')
        else:
            if not os.path.exists(file_name):
                while True:
                    print('Downloading data ...')
                    download_data_from_google_drive(MoNuSAC.urls[f'{self.mode} data'], file_name)
                    print('The file is being verified using MD5 ...')
                    if chack_file_with_md5(file_name, MoNuSAC.md5s[f'{self.mode} data']):
                        break
                    else:
                        print('Error with file,trying download again.')
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        mask_path = os.path.join(self.root, MoNuSAC.file_names[f'{self.mode} data'].replace('.zip', ''))
        masks     = glob(os.path.join(mask_path, '*/*.xml'))
        for mask in masks:
            if os.path.exists(os.path.join(mask.replace('xml', 'tif'))):
                self.data.append((os.path.join(*(mask.split('/')[:-1]), os.path.basename(mask).replace('xml', 'tif')), mask))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        (img, mask) = self.data[index]
        img         = Image.open(img)
        mask        = xml_to_binary_mask(img.width, img.height, mask)
        if self.transform is not None:
            return self.transform(img, mask)
        else:
            return TF.to_tensor(img), TF.to_tensor(mask)

    def __len__(self) -> int:
        return len(self.data)


class CryoNuSeg(Dataset):
    """
    Use the H&E images of CryoNuSeg. \\
    About moer: https://www.kaggle.com/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images \\
    There are 30 images in this dataset,and 512\times 512 pixels for each image and label. \\
    If transform is None,you will get size [3, 512, 512],[1, 512, 512] for tensors of images and labels. \\
    You may need run label = label[0] or label = label.squeeze(dim=0) when label is tensor in callable transform.
    """

    data_url = 'ipateam/segmentation-of-nuclei-in-cryosectioned-he-images'

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root:The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        self.root = root
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        if os.path.exists(os.path.join(self.root, 'tissue images')):
            print('Datas are already exist,cancel download.')
        else:
            file_name = os.path.join(self.root, 'segmentation-of-nuclei-in-cryosectioned-he-images.zip')
            if not os.path.exists(file_name):
                print('Downloading data ...')
                download_data_from_kaggle_dataset(CryoNuSeg.data_url, self.root)
            print('Unzipping ...')
            unzip_file(file_name, self.root)
            os.remove(file_name)

    def _load_data(self) -> None:
        images = glob(os.path.join(self.root, 'tissue images', '*.tif'))
        self.data = [(_, os.path.join(self.root, 'Annotator 1 (biologist second round of manual marks up)', 'Annotator 1 (biologist second round of manual marks up)', 'mask binary', os.path.basename(_).replace('tif', 'png'))) for _ in images]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        image, mask = self.data[index]
        image, mask = Image.open(image), Image.open(mask)
        if self.transform is not None:
            return self.transform(image, mask)
        else:
            return TF.to_tensor(image), TF.to_tensor(mask)

    def __len__(self) -> int:
        return len(self.data)


class ISBI_2012_EM_stacks(Dataset):
    """
    The EM stacks dataset in ISBI 2012. \\
    About more: https://imagej.net/events/isbi-2012-segmentation-challenge \\
    There are 30 images and labels in train data and 30 images in test. \\
    The sizes of all image and labels are 512 \times 512. \\
    If transform is None,you will get size [1, 512, 512],[1, 512, 512] for tensors of images and labels. \\
    You may need run label = label[0] or label = label.squeeze(dim=0) when label is tensor in callable transform.
    """

    data_url = 'https://github.com/zhixuhao/unet'

    def __init__(
        self,
        root: str,
        mode: str = 'train',
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        Args::
            root:The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label).
            download: Will download data or unpack package when data packages exist if download is True.
        """


        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root      = root
        self.mode      = mode
        self.transform = transform
        self.download  = download
        self.data      = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        if os.path.exists(os.path.join(self.root, self.mode)):
            print('Datas are already exist,cancel download.')
            return
        import git
        with git.Repo.init(path=self.root) as repo:
            print('Downloading data ...')
            repo.clone_from(ISBI_2012_EM_stacks.data_url, os.path.join(self.root, 'unet'))
        shutil.move(os.path.join(self.root, 'unet', 'data', 'membrane', 'train'), self.root)
        shutil.move(os.path.join(self.root, 'unet', 'data', 'membrane', 'test'), self.root)
        shutil.rmtree(os.path.join(self.root, 'unet'))
        shutil.rmtree(os.path.join(self.root, '.git'))
        [os.remove(_) for _ in glob(os.path.join(self.root, 'test', '*predict.png'))]

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, 'train', 'image')
            label_path = os.path.join(self.root, 'train', 'label')
            images     = glob(os.path.join(image_path, '*.png'))
            self.data  = [(_, os.path.join(label_path, os.path.basename(_))) for _ in images]
        else:
            self.data = glob(os.path.join(self.root, 'test', '*.png'))

    def __getitem__(self, index) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.mode == 'train':
            imgs = [Image.open(_) for _ in self.data[index]]
            return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)
        else:
            image = Image.open(self.data[index])
            return TF.to_tensor(image) if self.transform is None else self.transform(image)

    def __len__(self) -> int:
        return len(self.data)


class ISBI_CellTracking_2D(Dataset):
    """
    Segmentation part of ISBI CellTracking,this class only use 2D data. \\
    Use one of datas in BF-C2DL-HSC,BF-C2DL-MuSC,DIC-C2DH-HeLa,Fluo-C2DL-MSC,Fluo-N2DH-GOWT1,Fluo-N2DL-HeLa,PhC-C2DH-U373,PhC-C2DL-PSC,Fluo-N2DH-SIM+. \\
    About more:http://celltrackingchallenge.net/2d-datasets/ 
    """

    urls = {
        'BF-C2DL-HSC': {
            'train': 'http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip',
            'test': 'http://data.celltrackingchallenge.net/challenge-datasets/BF-C2DL-HSC.zip'
        },
        'BF-C2DL-MuSC': {
            'train': 'http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip',
            'test': 'http://data.celltrackingchallenge.net/challenge-datasets/BF-C2DL-MuSC.zip'
        },
        'DIC-C2DH-HeLa': {
            'train': 'http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip',
            'test': 'http://data.celltrackingchallenge.net/challenge-datasets/DIC-C2DH-HeLa.zip'
        },
        'Fluo-C2DL-MSC': {
            'train': 'http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip',
            'test': 'http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C2DL-MSC.zip'
        },
        'Fluo-N2DH-GOWT1': {
            'train': 'http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip',
            'test': 'http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-GOWT1.zip'
        },
        'Fluo-N2DL-HeLa': {
            'train': 'http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip',
            'test': 'http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DL-HeLa.zip'
        },
        'PhC-C2DH-U373': {
            'train': 'http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip',
            'test': 'http://data.celltrackingchallenge.net/challenge-datasets/PhC-C2DH-U373.zip'
        },
        'PhC-C2DL-PSC': {
            'train': 'http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip',
            'test': 'http://data.celltrackingchallenge.net/challenge-datasets/PhC-C2DL-PSC.zip'
        },
        'Fluo-N2DH-SIM+': {
            'train': 'http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip',
            'test': 'http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-SIM+.zip'
        }
    }
    md5s = {
        'BF-C2DL-HSC': {
            'train': '7399c8cd0452f80c5e46d43a952dd632',
            'test' : 'dc56332d884bca2b1b19dd9c2ed4e37f'
        },
        'BF-C2DL-MuSC': {
            'train': 'ea0935b1044a00a430f4283da9574857',
            'test' : '73f0205115f9ac75d60aaaed1b896eff'
        },
        'DIC-C2DH-HeLa': {
            'train': '86c35bbbc4be2e64f3c1c0cb52f77918',
            'test' : '1d6a91bfde978f9e23afa3ec1e3ca054'
        },
        'Fluo-C2DL-MSC': {
            'train': '22f0f061368fbd305ac6869311e5d089',
            'test' : '9003b6c9edcacb6199feb131820014ad'
        },
        'Fluo-N2DH-GOWT1': {
            'train': 'df6b8f9ddc7fd4fe4f201aab64e85832',
            'test' : 'e392e7cb3e9634582b57454954b98191'
        },
        'Fluo-N2DL-HeLa': {
            'train': '93862a64c8ce3a72eddff741d7504cb2',
            'test' : 'f336d1c29f10659f3a5d590e6fc37d46'
        },
        'PhC-C2DH-U373': {
            'train': 'bd7b8bb361eea2235f49078c60065cae',
            'test' : 'b428080e86dbe8422bce307ad593ec2e'
        },
        'PhC-C2DL-PSC': {
            'train': 'adfbc176f18d9c4522f4a41c784ba86c',
            'test' : '8991e3d340ddd0b0bdbb0e7340f950d9'
        },
        'Fluo-N2DH-SIM+': {
            'train': 'cfac3a2d0d59aa8c500a25ac6a9c95c3',
            'test' : '69a3a65de332c4ec2f689682c5ce1a82'
        }
    }

    def __init__(
        self,
        root: str,
        mode: str = 'train',
        version: str = 'BF-C2DL-HSC',
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        Args::
            root: The path dataset exist or save.
            mode: \'train\' or \'test\' 
            version: one of in [\'BF-C2DL-HSC\',\'BF-C2DL-MuSC\',\'DIC-C2DH-HeLa\',\'Fluo-C2DL-MSC\',\'Fluo-N2DH-GOWT1\',\'Fluo-N2DL-HeLa\',\'PhC-C2DH-U373\',\'PhC-C2DL-PSC\',\'Fluo-N2DH-SIM+\'].
            transform: A callable function or object.It will get parameter (image, label) when mode is train and get (image) when anther mode.
            download: Will download data or unpack package when data packages exist if download is True.
        """

        assert mode in ['train', 'test'] and version in ISBI_CellTracking_2D.urls.keys()

        super().__init__()
        self.root = root
        self.mode = mode
        self.version = version
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        if os.path.exists(os.path.join(self.root, self.version, self.mode, self.version)):
            print('Datas are already exist,cancel download.')
            return
        file_name = os.path.join(self.root, self.version, self.mode, ISBI_CellTracking_2D.urls[self.version][self.mode].split('/')[-1])
        if not os.path.exists(file_name):
            while True:
                print('Downloading data...')
                download_data_from_url(ISBI_CellTracking_2D.urls[self.version][self.mode], os.path.join(self.root, self.version, self.mode))
                print('The file is being verified using MD5 ...')
                if chack_file_with_md5(file_name, ISBI_CellTracking_2D.md5s[self.version][self.mode]):
                    break
                else:
                    print('Error with file,trying download again.')
        print('Unzipping ...')
        unzip_file(file_name, os.path.join(self.root, self.version, self.mode))
        os.remove(file_name)

    def _load_data(self) -> None:
        if self.mode == 'train':
            image_path = os.path.join(self.root, self.version, self.mode, self.version, '01')
            label_path = os.path.join(self.root, self.version, self.mode, self.version, '01_ST/SEG')
            if self.version == 'Fluo-N2DH-SIM+':
                label_path = os.path.join(self.root, self.version, self.mode, self.version, '01_GT/SEG')
            images      = glob(os.path.join(image_path, '*'))
            self.data  += [(_, os.path.join(label_path, 'man_seg' + os.path.basename(_)[1:])) for _ in images]
            image_path  = os.path.join(self.root, self.version, self.mode, self.version, '02')
            label_path  = os.path.join(self.root, self.version, self.mode, self.version, '02_ST/SEG')
            if self.version == 'Fluo-N2DH-SIM+':
                label_path = os.path.join(self.root, self.version, self.mode, self.version, '02_GT/SEG')
            images     = glob(os.path.join(image_path, '*'))
            self.data += [(_, os.path.join(label_path, 'man_seg' + os.path.basename(_)[1:])) for _ in images]
        else:
            self.data = glob(os.path.join(self.root, self.version, self.mode, self.version, '*', '*'))

    def __getitem__(self, index) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.mode == 'train':
            imgs = [Image.open(_) for _ in self.data[index]]
            return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)
        else:
            image = Image.open(self.data[index])
            return TF.to_tensor(image) if self.transform is None else self.transform(image)

    def __len__(self) -> int:
        return len(self.data)


class STARE(Dataset):
    """
    The blood vessel segmentation dataset from STructured Analysis of the Retina. \\
    About more:https://cecas.clemson.edu/~ahoover/stare/probing/index.html \\
    There are 20 images in this dataset ,each image has 2 labels that from Adam Hoover (ah) and Valentina Kouznetsova (vk).
    If transform is None,you will get size [3, 605, 700],[1, 605, 700],[1, 605, 700] for tensors of image and label. \\
    You may need run label = label[0] or label=label.squeeze(dim=0) when label is tensor in callable transform.'
    """

    urls = {
        'images'  : 'https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar',
        'label_ah': 'https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar',
        'label_vk': 'https://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar'
    }

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False) -> None:
        """
        Args::
            root:The path dataset exist or save.
            transform: A callable function or object.It will get parameter (image, label_ah, label_vk).
            download: Will download data or unpack package when data packages exist if download is True.
        """

        super().__init__()
        self.root = root
        self.transform = transform
        self.download = download
        self.data = []

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            self._download()
        elif not os.path.exists(root):
            raise Exception('Got a nonexistent folder but did not select download!')
        self._load_data()

    def _download(self) -> None:
        if os.path.exists(os.path.join(self.root, 'images')):
            print('Datas are already exist,cancel download.')
        else:
            for key in STARE.urls.keys():
                file_name = os.path.join(self.root, os.path.basename(STARE.urls[key]))
                if not os.path.exists(file_name):
                    print(f'Downloading {key} ...')
                    download_data_from_url(STARE.urls[key], self.root)
                print(f'\nUntaring {key} ...')
                image_dir = os.path.join(self.root, key)
                if os.path.exists(image_dir):
                    os.mkdir(image_dir)
                untar_file(file_name, image_dir)
                os.remove(file_name)
                for gz in glob(os.path.join(image_dir, '*.gz')):
                    ungz_file(gz, image_dir)
                    os.remove(gz)

    def _load_data(self) -> None:
        images        = glob(os.path.join(self.root, 'images', '*.ppm'))
        label_ah_path = os.path.join(self.root, 'label_ah')
        label_vk_path = os.path.join(self.root, 'label_vk')
        self.data = [(_, os.path.join(label_ah_path, os.path.basename(_)[:-3] + 'ah.ppm'), os.path.join(label_vk_path, os.path.basename(_)[:-3] + 'vk.ppm')) for _ in images]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        imgs = [Image.open(_) for _ in self.data[index]]
        return tuple([TF.to_tensor(_) for _ in imgs]) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)


class ACDC_2017(Dataset):
    """
    The segmentation of Automated Cardiac Diagnosis Challenge(ACDC). \\
    About more:https://www.creatis.insa-lyon.fr/Challenge/acdc/participation.html \\
    Please register and download data from above web page. \\
    The format for root path should be:: \\
        root/training/patientxxx/Info.cfg \\
        root/training/patientxxx/patientxxx_4d.nii.gz \\
        ... \\
        root/testing/patientxxx/Info.cfg \\
        root/testing/patientxxx/patientxxx_4d.nii.gz \\
        ...
    """

    def __init__(self, root: str, mode: str = 'train', transform: Optional[Callable] = None) -> None:
        """
        Args::
            root:The path dataset exist or save.
            mode:\'train\' or \'test\' 
            transform: A callable function or object.It will get parameter (image4d,imageES,imageED,labelES,labelED) when mode is train and get (image4d,imageES,imageED) when anther mode.
        """

        if mode not in ['train', 'test']:
            raise Exception(f'No such mode named \'{mode}\'')

        super().__init__()
        self.root      = root
        self.mode      = mode
        self.transform = transform
        self.data      = []

        self._load_data()

    def _load_data(self) -> None:
        if self.mode == 'train':
            infos = glob(os.path.join(self.root,'training','*','Info.cfg'))
            for infof in infos:
                with open(infof,'r') as f:
                    info = f.read().splitlines()
                    info = [_.split(':') for _ in info]
                    for item in info:
                        if item[0].strip() == 'ED':
                            ED = item[1].strip()
                        if item[0].strip() == 'ES':
                            ES = item[1].strip()
                img4d   = glob(os.path.join(*(infof.split('/')[:-1]),'*_4d.nii.gz'))[0]
                imges   = glob(os.path.join(*(infof.split('/')[:-1]),f'*{ES.zfill(2)}.nii.gz'))[0]
                labeles = glob(os.path.join(*(infof.split('/')[:-1]),f'*{ES.zfill(2)}_gt.nii.gz'))[0]
                imged   = glob(os.path.join(*(infof.split('/')[:-1]),f'*{ED.zfill(2)}.nii.gz'))[0]
                labeled = glob(os.path.join(*(infof.split('/')[:-1]),f'*{ED.zfill(2)}_gt.nii.gz'))[0]
                self.data.append((img4d,imges,imged,labeles,labeled))
        else:
            infos = glob(os.path.join(self.root,'testing','*','Info.cfg'))
            for infof in infos:
                with open(infof,'r') as f:
                    info = f.read().splitlines()
                    info = [_.split(':') for _ in info]
                    for item in info:
                        if item[0].strip() == 'ED':
                            ED = item[1].strip()
                        if item[0].strip() == 'ES':
                            ES = item[1].strip()
                img4d = glob(os.path.join(*(infof.split('/')[:-1]),'*_4d.nii.gz'))[0]
                imges = glob(os.path.join(*(infof.split('/')[:-1]),f'*{ES.zfill(2)}.nii.gz'))[0]
                imged = glob(os.path.join(*(infof.split('/')[:-1]),f'*{ED.zfill(2)}.nii.gz'))[0]
                self.data.append((img4d,imges,imged))

    def __getitem__(self, index) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        imgs = [to_tensor_3D(np.array(nib.load(_).get_fdata())) for _ in self.data[index]]
        return tuple(imgs) if self.transform is None else self.transform(*imgs)

    def __len__(self) -> int:
        return len(self.data)
