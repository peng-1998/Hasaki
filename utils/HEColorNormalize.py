from typing import Union
import torch
from PIL import Image
from torch.functional import Tensor
from torch.nn.modules.module import Module
import torchvision.transforms.functional as TF
from torchnmf.nmf import NMF


class HE_ColorNormalization(Module):
    '''
    Image normalization using H&E staining based on nonnegative matrix factorization, ref. Article: https://ieeexplore.ieee.org/document/7164042/figures#figures
    Non-negative matrix factorization performe:
    $$
    \min _{W, H} \frac{1}{2}\|V-W H\|_{F}^{2}+\lambda \sum_{j=1}^{r}\|H(j,)\|_{1}, W, H \geq 0
    $$
    '''
    def __init__(self, target_img: str = None, use_cuda: bool = False, W_target: Tensor = None, r: int = 2, Lambda: float = 0.1) -> None:
        '''
        Args:
            target_img:The path of target image.When W_target is None, this parameter cannot be None.
            use_cuda:Using CUDA acceleration, non-negative matrix factorization takes several seconds to iterate.
            W_target:The target matrix W.When target_img is None, this parameter cannot be None.
            r:Matrix W,H will be W:(C,r) H:(r,N).
            Lambda:Coefficients of sparse control items.
        '''
        super().__init__()
        assert target_img or W_target
        self.r = r * 2
        self.Lambda = Lambda

        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        if W_target:
            self.W_target = W_target.to(self.device)
        else:
            img_target = Image.open(target_img)
            img_target = TF.to_tensor(img_target).clamp(0.01, 0.99).to(self.device)
            assert img_target.shape[0] == 3
            img_target = img_target.view(3, -1)
            img_target = -torch.log(img_target)
            nmft = NMF(img_target.shape, self.r).to(self.device)
            nmft.sparse_fit(img_target, sW=Lambda)
            self.W_target = nmft.H.data
            self.Ht_RM = torch.quantile(nmft.W.data, 0.99)

    def forward(self, pic: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        if isinstance(pic, Tensor):
            assert pic.shape[0] == 3
            source_img_size = pic.shape
            source_img = pic.clamp(0.01, 0.99).to(self.device).view(3, -1)
        elif isinstance(pic, Image.Image):
            source_img = TF.to_tensor(pic).clamp(0.01, 0.99).to(self.device)
            assert source_img.shape[0] == 3
            source_img_size = source_img.shape
            source_img = source_img.view(3, -1)
        else:
            raise Exception('Unsupported types')
        source_img = -torch.log(source_img)
        nmfs = NMF(source_img.shape, self.r).to(self.device)
        nmfs.sparse_fit(source_img, max_iter=100)
        H_s = nmfs.W.T.data
        H_sn = H_s * self.Ht_RM / torch.quantile(H_s.data, 0.99)
        V = self.W_target.mm(H_sn)
        V = torch.exp(-V).view(source_img_size).clamp(0, 1)
        if isinstance(pic, Tensor):
            return V.to(pic.device).data
        else:
            return TF.to_pil_image(V.cpu().data)