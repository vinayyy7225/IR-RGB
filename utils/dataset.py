import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class LandsatDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Landsat patches saved in NPZ format.
    Includes data augmentation (flips and rotations) for satellite scenes during training.
    """
    def __init__(self, data_dir, is_train=True, scale_factor=4):
        """
        data_dir: Path to directory containing train or val NPZ files.
        is_train: True for training set, False for validation set.
        scale_factor: Upscaling factor for super-resolution simulation.
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.scale_factor = scale_factor
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # Load npz file
        with np.load(file_path) as data:
            # rgb: (256, 256, 3), range [0, 1]
            rgb = data["rgb"]
            # ir: (256, 256, 2), range [0, 1]
            ir = data["ir"]
            # nir: (256, 256, 1), range [0, 1]
            nir = data["nir"]
            
        # Data augmentation for training:
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                rgb = np.fliplr(rgb)
                ir = np.fliplr(ir)
                nir = np.fliplr(nir)
            # Random vertical flip
            if random.random() > 0.5:
                rgb = np.flipud(rgb)
                ir = np.flipud(ir)
                nir = np.flipud(nir)
            # Random 90, 180, 270 degree rotation
            k = random.randint(0, 3)
            if k > 0:
                rgb = np.rot90(rgb, k)
                ir = np.rot90(ir, k)
                nir = np.rot90(nir, k)
                
        # Convert to torch tensors and transpose to channel-first (C, H, W)
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
        ir_tensor = torch.from_numpy(np.ascontiguousarray(ir)).permute(2, 0, 1).float()
        nir_tensor = torch.from_numpy(np.ascontiguousarray(nir)).permute(2, 0, 1).float()
        
        # Normalize RGB to [-1, 1] for GAN training
        rgb_gan = rgb_tensor * 2.0 - 1.0
        
        # Simulate low-resolution thermal input for Real-ESRGAN
        # Downsample HR thermal (ir_tensor) by scale_factor
        h, w = ir_tensor.shape[1], ir_tensor.shape[2]
        lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
        
        # Add batch dim, interpolate, then remove batch dim
        ir_unsqueezed = ir_tensor.unsqueeze(0)
        lr_ir_tensor = F.interpolate(ir_unsqueezed, size=(lr_h, lr_w), mode='bilinear', align_corners=False).squeeze(0)
        
        return {
            "rgb": rgb_gan,          # target RGB in [-1, 1], shape (3, 256, 256)
            "rgb_01": rgb_tensor,    # target RGB in [0, 1], shape (3, 256, 256)
            "ir_hr": ir_tensor,      # target HR thermal in [0, 1], shape (2, 256, 256)
            "ir_lr": lr_ir_tensor,   # input LR thermal in [0, 1], shape (2, 64, 64)
            "nir": nir_tensor,        # input NIR in [0, 1], shape (1, 256, 256)
            "filename": os.path.basename(file_path)
        }
