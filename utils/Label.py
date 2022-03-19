from torch import Tensor
import torch

try:
    from cc_torch import connected_components_labeling
    x = torch.tensor([[0, 0], [1, 1]], dtype=torch.uint8, device='cuda')
    connected_components_labeling(x)
    SEGMENT_BACKEND = 'cc_torch'
except:
    SEGMENT_BACKEND = 'scipy'


def _labeling_2d_with_cc_troch(image: Tensor) -> Tensor:
    from cc_torch import connected_components_labeling
    image = image.to(dtype=torch.uint8, device='cuda')
    image = connected_components_labeling(image)
    bincount = image.view(-1).bincount()
    index = torch.where(bincount != 0)[0]
    for idx in range(len(index)):
        image[image == index[idx]] = idx
    return image.long().to(image.device)


def _labeling_2d_with_scipy(image: Tensor) -> Tensor:
    from scipy.ndimage import label
    result, _ = label(image.data.cpu().numpy())
    return torch.from_numpy(result).long().to(image.device)


def labeling_2d(image: Tensor) -> Tensor:
    '''
    Args:
        image:An binary 2D image with shape (H,W),the dtype should be bool or int and long
    Return:
        Tensor of objects indexes with same shape with image and dtype long
    '''
    if SEGMENT_BACKEND == 'cc_torch':
        return _labeling_2d_with_cc_troch(image)
    return _labeling_2d_with_scipy(image)