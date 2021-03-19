import os
import numpy as np
import torch
from typing import Tuple


class LSTMDataset(torch.utils.data.Dataset):

    def __init__(self, rootdir: str) -> None:
        self.rootdir   = rootdir
        self.fileslist = os.listdir(rootdir)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        npzfile = np.load(os.path.join(self.rootdir, self.fileslist[index]))
        inputs  = torch.as_tensor(npzfile["inputs"].astype(np.float32))
        targets = torch.as_tensor(npzfile["targets"].astype(np.float32))
        return inputs, targets

    def __len__(self) -> int:
        return len(self.fileslist)
