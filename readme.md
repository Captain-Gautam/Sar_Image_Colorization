# Unpaired and Unsupervised: A Dual-Model Framework for SAR Image Colorization and Downstream Building Verification

**Authors:** Jyoti Triklani, Siddhi Hirani, Gautam Prajapati  
**Institution:** School of Engineering and Applied Science (SEAS), Ahmedabad University  
**Course:** CSE 618 Artificial Intelligence Laboratory  

## Project Overview
This project presents a comprehensive deep learning pipeline for processing Synthetic Aperture Radar (SAR) images to extract meaningful topographical structures, specifically focusing on building segmentation. The pipeline consists of three sequential stages:

1. **SAR Image Colorization (CycleGAN):** Converting raw, grayscale SAR images into realistic optical-like color images using generative models (ResNet-based Generators and PatchGAN Discriminators). This step bridges the domain gap.
2. **Denoising & Refinement (DDRM):** Applying diffusion models to denoise and significantly enhance the colorized images. Uses SVD and a U-Net denoiser for higher-fidelity imagery.
3. **Building Segmentation (U-Net):** Utilizing a U-Net trained entirely on unsupervised pseudo-masks (extracted via ResNet50 and K-Means clustering) to perform accurate building footprint segmentation without manual labels.

## Data Directories, Dataset Split & Characteristics
The framework is trained and evaluated on **Sentinel-1 (SAR)** and **Sentinel-2 (Optical)** image pairs originally segregated into four terrain categories: Agriculture, Barrenland, Urban, and Grassland.
- **Volume**: 4,000 images per category per modality (**16,000 total images** at `256x256` resolution).
- **Split**: The dataset utilizes a strict **70%, 15%, 15% split** for training, validation, and testing.

```text
data/
└── sar_optical_image/
    ├── train/              # 70% of the dataset
    │   ├── A/              # Input SAR images
    │   └── B/              # Target Optical images / Masks
    ├── val/                # 15% of the dataset
    │   ├── A/
    │   └── B/
    └── test/               # 15% of the dataset
        ├── A/
        └── B/
```

## Codebase Structure
- `cyclegan_r9.py` - Core script for training the SAR to Optical image colorization models.
- `diffusion.py` - Implements the diffusion model for denoising and high-resolution refinement.
- `building_segmentation01.py` - Script dedicated to training and evaluating the building segmentation network.
- `models.py` - Contains the neural network architectures used throughout the pipeline.
- `datasets.py` - Pytorch Dataset and DataLoader classes handling the 70/15/15 splits and augmentations.
- `utils.py` - Helper functions for metrics, checkpointing, and image visualization.

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd SAR_CODE
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install standard dependencies:**
   Ensure you install compatible versions of PyTorch (ideally with CUDA support for GPU acceleration).
   ```bash
   pip install torch torchvision
   pip install numpy pandas matplotlib opencv-python pillow tqdm
   ```

## Usage

### Phase 1: SAR to Optical Colorization
Begin by training the domain translation model to convert SAR inputs into optical pseudo-images.
```bash
python cyclegan_r9.py
```

### Phase 2: Denoising and Sharpening
Once colorized, use the diffusion model to refine the generated images, increasing sharpness and removing structural hallucinations or noise.
```bash
python diffusion.py
```

### Phase 3: Building Segmentation
Finally, run the segmentation model on the refined optical images to detect and mask building footprints.
```bash
python building_segmentation01.py
```

## Experimental Results

The framework was evaluated on 200 test samples, with performance metrics for both domain translation and building segmentation:

### Quantitative Results for SAR-to-Optical Translation
| Model | PSNR (dB) | SSIM |
| :--- | :--- | :--- |
| **CycleGAN (7 Residual Blocks)** | `10.33 ± 1.28` | `0.104 ± 0.029` |
| **CycleGAN (5 Residual Blocks)** | `9.48 ± 1.66` | `0.027 ± 0.029` |

*Note: PSNR and SSIM values are lower than paired tasks, which is expected due to the fundamental domain gap, lack of pixel-wise correspondence in unpaired training, and the one-to-many mapping complexity.*

### Denoising (DDRM)
The DDRM successfully cleans up adversarial training irregularities and removes color bleeding artifacts while preserving semantic content, drastically improving texture consistency and sharpness.

### Building Segmentation
Using the fully unsupervised pseudo-labeling and U-Net architecture, the segmentation yielded highly competitive results without any manual annotations:
- **IoU:** `0.763 ± 0.045`
- **Dice Coefficient:** `0.865 ± 0.038`
