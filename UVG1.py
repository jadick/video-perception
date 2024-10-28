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
import torch
import random
import os
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import glob


class UVG(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        split="train",
        max_frames=3
    ):
        #print (transform)
        if transform is None:
            raise RuntimeError("Transform must be applied")
        #split = "tri_"+split+"list
        vid_dirs = glob.glob(root + "/*")
        self.max_frames = max_frames  # hard coding for now
        self.transform = transform
        self.samples = []

        for vid_dir in vid_dirs:
            num_frames = len(os.listdir(vid_dir))
            for i in range(num_frames - self.max_frames+1):
                sample = {
                    'dir': vid_dir,
                    'frame_indices': list(range(i, i + self.max_frames))
                }
                self.samples.append(sample)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        sample = self.samples[index]
        dir_ = sample['dir']
        frame_index = sample['frame_indices']

        frames = []
        frame_paths = []
        frames_0 = []

        #frame_paths.append(Path(str(frame_paths[0]).replace("vimeo_triplet", "vimeo_triplet_diffusion")))


        for idx in frame_index:
            frame_path = os.path.join(dir_, f"frame_{idx+1:04d}.png")
            frame_paths.append(frame_path)
        frame_path_0 = os.path.join(dir_, f"frame_{frame_index[0]+1:04d}.png").replace("uvg", "uvg_diffusion")
        frame_paths.append(frame_path_0)

        #frames = np.concatenate(
        #    [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        #)

        frames = np.concatenate(
            [np.asarray(Image.open(frame_paths[i]).convert("RGB"))[:, 3:3+448,:] if i <3 else  np.asarray(Image.open(frame_paths[i]).convert("RGB"))\
             for i in range(len(frame_paths))], axis=-1
        )

        #print (frames.shape)
        frames = torch.chunk(self.transform(frames), 4)

        return torch.stack(frames) #{"frames": frames, "path": str(frame_paths[0])}

    def __len__(self):
        return len(self.samples)
