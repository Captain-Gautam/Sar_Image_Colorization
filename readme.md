# Unpaired and Unsupervised: A Dual-Model Framework for SAR Image Colorization and Downstream Building Verification

## 📌 Project Overview
Synthetic Aperture Radar (SAR) imagery is highly valuable as it can be captured in all weather and lighting conditions. However, SAR images are grayscale, contain complex speckle patterns, and are notoriously difficult to interpret visually. Optical images, while providing natural colors and easy interpretability, are limited by cloud cover and daylight availability.

This repository implements an **unsupervised and unpaired learning framework** to translate SAR images to Optical (RGB) images. By colorizing SAR images, we bridge the gap between sensor modalities, improving human interpretability and boosting the performance of downstream tasks like building verification without relying on scarce, perfectly paired datasets.

**Authors:** Jyoti Triklani, Siddhi Hirani, Gautam Prajapati  
**Institution:** School of Engineering and Applied Science (SEAS), Ahmedabad University  
**Course:** CSE 618 Artificial Intelligence Laboratory  

---

## 🏗️ Framework Architecture

Our dual-model approach explores two distinct generative pipelines for the SAR -> Optical translation, followed by a robust segmentation pipeline:

### 1. CycleGAN (Unpaired Image Translation)
Learns the bidirectional mapping (SAR <-> RGB) using two Generators and two Discriminators (PatchGAN). 
* **Generators (`models.py`)**: Uses a ResNet-based architecture (initially 9 blocks, optimized down to 5/7 blocks to prevent overfitting and adapt to hardware constraints).
* **Losses (`cyclegan_r9.py`)**: Combines Adversarial Loss, Cycle-Consistency Loss (reconstruction), and Identity Loss (to maintain color/brightness).

### 2. DDRM (Denoising Diffusion Restoration Model)
As an alternative/enhancement, we explored OpenAI's pretrained guided diffusion model (LSUN Churches dataset). 
* The CycleGAN output undergoes a forward diffusion process (adding Gaussian noise).
* A U-Net denoiser estimates the clean image at each iteration, utilizing Singular Value Decomposition (SVD) to combine input-guided range space with diffusion-hallucinated null space.

### 3. Downstream Building Verification (U-Net)
Evaluates the practical utility of the generated images.
* Uses a **U-Net** architecture trained entirely on **pseudo-masks** (no manual ground truth required).
* Pixel features are extracted via a pretrained ResNet50, clustered using K-Means to identify buildings, and subsequently refined by the U-Net to output clean segmentation maps.

---

## 📊 Dataset
The models are trained and evaluated on **Sentinel-1 (SAR)** and **Sentinel-2 (Optical)** image pairs, segregated by terrain:
* **Categories:** Agriculture, Barrenland, Urban, Grassland.
* **Volume:** 4,000 images per category per modality -> **16,000 total images**.
* **Resolution:** 256x256 pixels.
* **Dataset Split:** The data is systematically divided into a **70% Training**, **15% Validation**, and **15% Testing** split to ensure robust model evaluation and prevent data leakage.
* *Note: While the dataset contains pairs, the models are trained in an unpaired manner to simulate real-world constraints.*

---

## ⚙️ Repository Structure

```text
├── cyclegan_r9.py       # Main training and validation script for the CycleGAN model
├── models.py            # PyTorch implementation of GeneratorResNet and Discriminator (PatchGAN)
├── datasets.py          # Custom PyTorch Dataset class for loading unaligned A/B domain images
├── utils.py             # ReplayBuffer, LR Schedulers, and metric calculators (PSNR, SSIM, LPIPS)
└── README.md            # Project documentation