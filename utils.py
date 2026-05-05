# Code credit : PyTorch-GAN Git repo

import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image

# For evaluation metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


# ===================================
# Evaluation Metrics
# ===================================

# Global LPIPS model (lazy initialization)
_lpips_model = None

def get_lpips_model():
    """Lazy initialization of LPIPS model"""
    global _lpips_model
    if _lpips_model is None:
        try:
            _lpips_model = lpips.LPIPS(net='alex')  # Using AlexNet backbone
            if torch.cuda.is_available():
                _lpips_model = _lpips_model.cuda()
        except Exception as e:
            print(f"Warning: Could not initialize LPIPS model: {e}")
            _lpips_model = None
    return _lpips_model


def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: numpy array of shape (C, H, W) or (H, W, C) with values in [0, 1]
        img2: numpy array of shape (C, H, W) or (H, W, C) with values in [0, 1]
        data_range: the data range of the input image (default: 1.0 for [0,1] range)
    
    Returns:
        PSNR value in dB
    """
    try:
        # Convert from (C, H, W) to (H, W, C) if needed
        if img1.shape[0] == 3 or img1.shape[0] == 1:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        
        # Ensure images are in [0, 1] range
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        psnr_value = peak_signal_noise_ratio(img1, img2, data_range=data_range)
        return psnr_value
    except Exception as e:
        print(f"Warning: PSNR calculation failed: {e}")
        return 0.0


def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: numpy array of shape (C, H, W) or (H, W, C) with values in [0, 1]
        img2: numpy array of shape (C, H, W) or (H, W, C) with values in [0, 1]
        data_range: the data range of the input image (default: 1.0 for [0,1] range)
    
    Returns:
        SSIM value between -1 and 1 (typically 0 to 1)
    """
    try:
        # Convert from (C, H, W) to (H, W, C) if needed
        if img1.shape[0] == 3 or img1.shape[0] == 1:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        
        # Ensure images are in [0, 1] range
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        # For multichannel images
        channel_axis = 2 if img1.ndim == 3 else None
        
        ssim_value = structural_similarity(
            img1, img2, 
            data_range=data_range,
            channel_axis=channel_axis,
            multichannel=True if channel_axis is not None else False
        )
        return ssim_value
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        return 0.0


def calculate_lpips(img1, img2):
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS) between two images.
    Lower LPIPS means more similar images.
    
    Args:
        img1: numpy array of shape (C, H, W) with values in [0, 1]
        img2: numpy array of shape (C, H, W) with values in [0, 1]
    
    Returns:
        LPIPS distance (lower is better, typically 0 to 1)
    """
    try:
        model = get_lpips_model()
        if model is None:
            return 0.0
        
        # Ensure correct shape (C, H, W)
        if img1.shape[0] != 3:
            if len(img1.shape) == 3 and img1.shape[2] == 3:
                img1 = np.transpose(img1, (2, 0, 1))
                img2 = np.transpose(img2, (2, 0, 1))
        
        # Convert to torch tensors and normalize to [-1, 1] for LPIPS
        img1_tensor = torch.from_numpy(img1).float().unsqueeze(0)  # Add batch dimension
        img2_tensor = torch.from_numpy(img2).float().unsqueeze(0)
        
        # LPIPS expects input in [-1, 1], so convert from [0, 1] to [-1, 1]
        img1_tensor = img1_tensor * 2.0 - 1.0
        img2_tensor = img2_tensor * 2.0 - 1.0
        
        if torch.cuda.is_available():
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()
        
        with torch.no_grad():
            lpips_value = model(img1_tensor, img2_tensor).item()
        
        return lpips_value
    except Exception as e:
        print(f"Warning: LPIPS calculation failed: {e}")
        return 0.0
