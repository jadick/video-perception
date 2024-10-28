# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

#from compressai.registry import register_dataset


#@register_dataset("Vimeo90kDataset")
class Vimeo90kDataset(Dataset):
    """Load a Vimeo-90K structured dataset.

    Vimeo-90K dataset from
    Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, William T. Freeman:
    `"Video Enhancement with Task-Oriented Flow"
    <https://arxiv.org/abs/1711.09078>`_,
    International Journal of Computer Vision (IJCV), 2019.

    Training and testing image samples are respectively stored in
    separate directories:

    .. code-block::

        - rootdir/
            - sequence/
                - 00001/001/im1.png
                - 00001/001/im2.png
                - 00001/001/im3.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'valid')
        tuplet (int): order of dataset tuplet (e.g. 3 for "triplet" dataset)
    """

    def __init__(self, root, transform=None, split="train", tuplet=3):
        list_path = Path(root) / self._list_filename(split, tuplet)

        with open(list_path) as f:
            self.samples = [
                f"{root}/sequences/{line.rstrip()}/im{idx}.png"
                for line in f
                if line.strip() != ""
                for idx in range(1, tuplet + 1)
            ]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

    def _list_filename(self, split: str, tuplet: int) -> str:
        tuplet_prefix = {3: "tri", 7: "sep"}[tuplet]
        list_suffix = {"train": "trainlist", "valid": "testlist"}[split]
        return f"{tuplet_prefix}_{list_suffix}.txt"
    
class VideoFolder_diffusion(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directorie containing mnay sub-directorie like Vimeo90K Dataset:

    .. code-block::

        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    ...
                    -0932/
                    -0933/
                    ...
                - 00011/
                    ...
                - 00012/
                    ...

    training and testing (valid) clips are withdrew from sub-directory navigated by
    corresponding input files listing relevant folders.

    This class returns a set of three video frames in a tuple.
    Random interval can be applied to if subfolders includes more than 6 frames.

    Args:
        root (string): root directory of the dataset
        rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        split="train",
    ):
        #print (transform)
        if transform is None:
            raise RuntimeError("Transform must be applied")
        #split = "tri_"+split+"list
        splitfile = Path(f"{root}/tri_trainlist.txt")
        #print ("WHUTTTTTTTTTTTTTTTTTTTTT", splitfile)
        splitdir = Path(f"{root}/sequences")

        if not splitfile.is_file():
            raise RuntimeError(f'Invalid file "{root}"')

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        with open(splitfile, "r") as f_in:
            self.sample_folders = [Path(f"{splitdir}/{f.strip()}") for f in f_in]

        self.max_frames = 3  # hard coding for now
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]
        #print (frame_paths[0])
        frame_paths.append(Path(str(frame_paths[0]).replace("vimeo_triplet", "vimeo_triplet_diffusion")))
        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )
        
        #print (frames.min())
        #print (frames.max())
        I_frame = frames[...,:3]
        
        
        frames = torch.chunk(self.transform(frames), 4)
        #print (frames[0].shape)
        #if self.rnd_temp_order:
        #    if random.random() < 0.5:
        #        return frames[::-1]

        return torch.stack(frames)
    

    def __len__(self):
        return len(self.sample_folders)

    
class VideoFolder(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directorie containing mnay sub-directorie like Vimeo90K Dataset:

    .. code-block::

        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    ...
                    -0932/
                    -0933/
                    ...
                - 00011/
                    ...
                - 00012/
                    ...

    training and testing (valid) clips are withdrew from sub-directory navigated by
    corresponding input files listing relevant folders.

    This class returns a set of three video frames in a tuple.
    Random interval can be applied to if subfolders includes more than 6 frames.

    Args:
        root (string): root directory of the dataset
        rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        split="train",
    ):
        #print (transform)
        if transform is None:
            raise RuntimeError("Transform must be applied")
        #split = "tri_"+split+"list
        splitfile = Path(f"{root}/tri_trainlist.txt")
        #print ("WHUTTTTTTTTTTTTTTTTTTTTT", splitfile)
        splitdir = Path(f"{root}/sequences")

        if not splitfile.is_file():
            raise RuntimeError(f'Invalid file "{root}"')

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        with open(splitfile, "r") as f_in:
            self.sample_folders = [Path(f"{splitdir}/{f.strip()}") for f in f_in]

        self.max_frames = 3  # hard coding for now
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]

        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )
        
        #print (frames.min())
        #print (frames.max())
        I_frame = frames[...,:3]
        
        
        frames = torch.chunk(self.transform(frames), self.max_frames)
        #print (frames[0].shape)
        #if self.rnd_temp_order:
        #    if random.random() < 0.5:
        #        return frames[::-1]

        return torch.stack(frames)
    

    def __len__(self):
        return len(self.sample_folders)
    