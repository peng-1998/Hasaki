from torch import Tensor
import torch


def mask_center(mask: Tensor):
    '''
    Args:   
        mask:a ndim 0-1 value Tensor or bool Tensor where object area is 1 or True
    Retruen:
        a float Tenser with shape [n]
    '''
    return mask.nonzero().float().mean(dim=0)


from torch.nn.functional import conv3d, conv2d


def mask_edge(mask: Tensor):
    '''
    Args:   
        mask:a 2d or 3d 0-1 value Tensor or bool Tensor where object area is 1 or True
    Retruen:
        a long type Tensor with mask's shape,where the edge of mask is 1
    '''
    if len(mask.shape) == 2:
        kernel = torch.tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        return (mask.bool() ^ (conv2d(mask[None, None, :, :], -kernel.to(mask.device), None, 1, 1) < 0)).long()
    elif len(mask.shape) == 3:
        kernel = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        return (mask.bool() ^ (conv3d(mask[None, None, :, :, :], -kernel.to(mask.device), None, 1, 1) < 0)).long()
    else:
        raise Exception(f'mask is expected to be 2d or 3d,but get Tensor with shape {list(mask.shape)}')


def instances_centers(instances: Tensor):
    '''
    Args:
        instances: nd Tensor, use 1 for frist instance and 2 for the second ...
    Return:
        Tensor with shape [k,n], k is the count of instances
    '''
    assert instances.max() > 0
    centers = []
    for i in range(instances.max()):
        centers.append(mask_center(instances == (i + 1)))
    return torch.stack(centers)



