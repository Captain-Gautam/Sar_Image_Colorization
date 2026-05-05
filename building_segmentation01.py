"""
unsupervised_unet_selftrain.py - Memory Optimized Version

Two-stage unsupervised building segmentation:
1) Extract deep pixel features (ResNet50) and cluster -> pseudo masks
2) Train a U-Net on pseudo masks (self-training) to get refined masks

Requirements:
pip install torch torchvision scikit-learn pillow numpy matplotlib scikit-image tqdm
"""

import os
import warnings
warnings.filterwarnings("ignore")

import glob
import math
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from sklearn.cluster import MiniBatchKMeans
from skimage.morphology import remove_small_objects, closing, square

# -------------------------
# CONFIG
# -------------------------
dataset_path = r"D:\SAR_Mtech\v_2\urban\s2"
output_dir = r"D:\SAR_Mtech\output_bs"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# hyperparams
IMG_SIZE = 256
BATCH = 8
NUM_WORKERS = 2
N_CLUSTERS = 5
MIN_OBJ_SIZE = 300
UNET_EPOCHS = 15
LR = 1e-3
SUBSAMPLE_PIXELS = 0.1  # Use 10% of pixels for KMeans
KMEANS_TRAIN_IMAGES = 500  # Use first 500 images for KMeans training

# -------------------------
# Utilities
# -------------------------
def imread_rgb(path, resize=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), Image.BILINEAR)
    return np.array(img)

def save_mask_png(mask, out_path):
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)

# -------------------------
# Dataset (file list)
# -------------------------
image_files = sorted([p for p in glob.glob(os.path.join(dataset_path, "*")) 
                      if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
if len(image_files) == 0:
    raise SystemExit(f"No images found under {dataset_path}")

print(f"Found {len(image_files)} images")

# -------------------------
# Feature extractor (ResNet50 pretrained)
# -------------------------
class ResNetFeat(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
    def forward(self, x):
        f = self.backbone(x)
        f = nn.functional.interpolate(f, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return f

feat_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

feat_model = ResNetFeat().to(device)
feat_model.eval()

# -------------------------
# Stage 1: Train KMeans incrementally (memory efficient)
# -------------------------
print("Stage 1: Training KMeans on subsampled pixels...")
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=10000, random_state=42)

train_images = image_files[:min(KMEANS_TRAIN_IMAGES, len(image_files))]
with torch.no_grad():
    for p in tqdm(train_images, desc="KMeans training"):
        img = Image.open(p).convert("RGB")
        inp = feat_transform(img).unsqueeze(0).to(device)
        f = feat_model(inp).cpu().numpy()[0]  # C x H x W
        C, H, W = f.shape
        arr = f.reshape(C, -1).T  # (H*W) x C
        
        # Subsample pixels
        n_pixels = arr.shape[0]
        n_sample = int(n_pixels * SUBSAMPLE_PIXELS)
        idx = np.random.choice(n_pixels, n_sample, replace=False)
        arr_sampled = arr[idx].astype(np.float32)
        
        kmeans.partial_fit(arr_sampled)
        
        # Clear memory
        del arr, arr_sampled, f, inp
        if device == "cuda":
            torch.cuda.empty_cache()

# -------------------------
# Stage 2: Generate pseudo masks
# -------------------------
pseudo_dir = os.path.join(output_dir, "pseudo_masks")
os.makedirs(pseudo_dir, exist_ok=True)
pseudo_paths = []

print("Stage 2: Generating pseudo masks...")
with torch.no_grad():
    for p in tqdm(image_files, desc="Creating masks"):
        img = Image.open(p).convert("RGB")
        inp = feat_transform(img).unsqueeze(0).to(device)
        f = feat_model(inp).cpu().numpy()[0]  # C x H x W
        C, H, W = f.shape
        arr = f.reshape(C, -1).T  # (H*W) x C
        
        # Predict in chunks to avoid memory issues
        chunk_size = 10000
        labels = []
        for i in range(0, arr.shape[0], chunk_size):
            chunk = arr[i:i+chunk_size].astype(np.float32)
            labels.append(kmeans.predict(chunk))
        lbl = np.concatenate(labels).reshape(H, W)
        
        # Choose brightest cluster
        img_small = imread_rgb(p, resize=IMG_SIZE) / 255.0
        gray = img_small.mean(axis=2)
        mean_by_cluster = []
        for c in range(N_CLUSTERS):
            sel = gray[lbl == c]
            mean_by_cluster.append(sel.mean() if sel.size>0 else 0.0)
        
        chosen = int(np.argmax(mean_by_cluster))
        mask = (lbl == chosen).astype(np.uint8)

        # Morphological cleanup
        mask = remove_small_objects(mask.astype(bool), min_size=MIN_OBJ_SIZE)
        mask = closing(mask, square(3))
        mask = mask.astype(np.uint8)

        # Save
        base = Path(p).stem
        outp = os.path.join(pseudo_dir, base + "_pseudo.png")
        save_mask_png(mask, outp)
        pseudo_paths.append(outp)
        
        # Clear memory
        del arr, f, inp, lbl, mask
        if device == "cuda":
            torch.cuda.empty_cache()

print("Saved pseudo masks to:", pseudo_dir)

# -------------------------
# Dataset class for UNet training
# -------------------------
class TrainDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=True):
        self.imgs = image_paths
        self.masks = mask_paths
        self.augment = augment
        self.to_tensor = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        if self.augment:
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        img_t = self.to_tensor(img)
        mask_t = (self.to_tensor(mask) > 0.5).float()
        return img_t, mask_t

masks_sorted = []
for p in image_files:
    base = Path(p).stem
    masks_sorted.append(os.path.join(pseudo_dir, base + "_pseudo.png"))

train_ds = TrainDataset(image_files, masks_sorted, augment=True)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

# -------------------------
# UNet Model
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.up2 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv_up2 = DoubleConv(base*8, base*4)
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv_up1 = DoubleConv(base*4, base*2)
        self.up0 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv_up0 = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, out_channels, 1)
    def forward(self, x):
        c1 = self.inc(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        u2 = self.up2(c4); u2 = torch.cat([u2, c3], dim=1); u2 = self.conv_up2(u2)
        u1 = self.up1(u2); u1 = torch.cat([u1, c2], dim=1); u1 = self.conv_up1(u1)
        u0 = self.up0(u1); u0 = torch.cat([u0, c1], dim=1); u0 = self.conv_up0(u0)
        return self.outc(u0)

def dice_loss_logits(input, target, eps=1e-6):
    probs = torch.sigmoid(input)
    num = 2 * (probs * target).sum(dim=(2,3))
    den = probs.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    loss = 1 - (num / den)
    return loss.mean()

model_unet = UNet(in_channels=3, out_channels=1, base=32).to(device)
optim_unet = torch.optim.Adam(model_unet.parameters(), lr=LR)
bce = nn.BCEWithLogitsLoss()

# -------------------------
# Train U-Net
# -------------------------
print("Stage 3: Training U-Net on pseudo-labels...")
for epoch in range(UNET_EPOCHS):
    model_unet.train()
    epoch_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{UNET_EPOCHS}"):
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = model_unet(imgs)
        loss = bce(logits, masks) + dice_loss_logits(logits, masks) * 0.8
        optim_unet.zero_grad()
        loss.backward()
        optim_unet.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{UNET_EPOCHS}  avg_loss: {epoch_loss/len(train_loader):.4f}")

model_path = os.path.join(output_dir, "unet_selftrained.pth")
torch.save(model_unet.state_dict(), model_path)
print("Saved U-Net model to:", model_path)

# -------------------------
# Generate refined masks
# -------------------------
print("Stage 4: Generating refined masks...")
refined_dir = os.path.join(output_dir, "refined_masks")
os.makedirs(refined_dir, exist_ok=True)
model_unet.eval()
with torch.no_grad():
    for img_p in tqdm(image_files, desc="Refining"):
        img = Image.open(img_p).convert("RGB")
        inp = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])(img).unsqueeze(0).to(device)
        logits = model_unet(inp)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
        mask_small = (prob > 0.5).astype(np.uint8)
        mask_small = remove_small_objects(mask_small.astype(bool), min_size=MIN_OBJ_SIZE)
        mask_small = closing(mask_small, square(3)).astype(np.uint8)
        orig = Image.open(img_p)
        ow, oh = orig.size
        mask_full = Image.fromarray((mask_small*255).astype(np.uint8)).resize((ow, oh), Image.NEAREST)
        out_name = Path(img_p).stem + "_refined.png"
        out_path = os.path.join(refined_dir, out_name)
        mask_full.save(out_path)

print("Refined masks saved to:", refined_dir)

# -------------------------
# Visualization
# -------------------------
def show_examples(n=4):
    idxs = np.linspace(0, len(image_files)-1, n, dtype=int)
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(idxs):
        img_p = image_files[idx]
        img = Image.open(img_p).convert("RGB")
        mask_p = os.path.join(refined_dir, Path(img_p).stem + "_refined.png")
        mask = Image.open(mask_p).convert("L")
        plt.subplot(n, 2, 2*i+1)
        plt.imshow(img); plt.axis("off"); plt.title("orig")
        plt.subplot(n, 2, 2*i+2)
        plt.imshow(mask, cmap="gray"); plt.axis("off"); plt.title("refined mask")
    plt.tight_layout()
    plt.show()

show_examples(4)