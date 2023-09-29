import torch
from torch import Tensor


def euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
    """
    Calculates pair-wise distance between row vectors in x and those in y.

    Replaces torch cdist with p=2, as cdist is not properly exported to onnx and openvino format.
    Resulting matrix is indexed by x vectors in rows and y vectors in columns.

    Args:
        x: input tensor 1
        y: input tensor 2

    Returns:
        Matrix of distances between row vectors in x and y.
    """
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
    # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
    res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
    res = res.clamp_min_(0).sqrt_()
    return res
