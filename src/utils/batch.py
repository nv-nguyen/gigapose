import torch
import numpy as np
from einops import rearrange


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        if isinstance(self.data, np.ndarray):
            return np.ceil(len(self.data) / self.batch_size).astype(int)
        elif isinstance(self.data, torch.Tensor):
            length = self.data.shape[0]
            return np.ceil(length / self.batch_size).astype(int)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)


def gather(features, index_patches):
    """
    Args:
    - features: (B, C, H, W)
    - index_patches: (B, N, 2) where N is the number of patches, and 2 is the (x, y) index of the patch
    Output:
    - selected_features: (BxN, C) where index_patches!= -1
    """
    B, C, H, W = features.shape
    features = rearrange(features, "b c h w -> b (h w) c")

    index_patches = index_patches.clone()
    x, y = index_patches[:, :, 0], index_patches[:, :, 1]
    mask = torch.logical_and(x != -1, y != -1)
    index_patches[index_patches == -1] = H - 1  # dirty fix so that gather does not fail

    # Combine index_x and index_y into a single index tensor
    index = y * W + x

    # Gather features based on index tensor
    flatten_features = torch.gather(
        features, dim=1, index=index.unsqueeze(-1).repeat(1, 1, C)
    )

    # reshape to (BxN, C)
    flatten_features = rearrange(flatten_features, "b n c -> (b n) c")
    mask = rearrange(mask, "b n -> (b n)")
    return flatten_features[mask]
