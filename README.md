# Nerve Vessel Segmentation using Pix2Pix GAN

This repository contains a Jupyter Notebook that implements a Pix2Pix Generative Adversarial Network (GAN) for segmenting blood vessels in fundus images. The model learns to transform original fundus images into their corresponding vessel segmentation masks.

## Project Overview
The goal of this project is to accurately segment blood vessels from fundus (retinal) images. This is a crucial step in diagnosing and monitoring various ocular diseases. A Pix2Pix GAN architecture is employed, consisting of a generator (U-Net) and a discriminator (PatchGAN), to perform this image-to-image translation task.

## Dataset
The model is trained and evaluated using the [Fundus Image Dataset for Vessel Segmentation](https://www.kaggle.com/datasets/andrewmvd/fundus-image-vessel-segmentation). This dataset contains pairs of original fundus images and their corresponding ground truth vessel masks. 


- Images are resized to 256×256 pixels for processing
- Dataset is publicly available on Kaggle (check Kaggle page for licensing details)

## Model Architecture
### Generator
U-Net architecture with:
- **Encoder:** Downsampling blocks (Conv2D → BatchNorm → LeakyReLU)
- **Bottleneck:** Central convolutional layer
- **Decoder:** Upsampling blocks (Conv2DTranspose → BatchNorm → Dropout → Concatenate skip connections → ReLU)
- **Output Layer:** Conv2DTranspose → tanh activation (output range: [-1, 1])

### Discriminator
PatchGAN architecture:
- Input: Concatenated source image + target mask (real or generated)
- Layers: Conv2D → LeakyReLU → BatchNorm
- Output: Single patch with sigmoid activation (real/fake probability)

## Training Details
- **Optimizer:** Adam (`lr=0.0002`, `beta_1=0.5`)
- **Loss Functions:**
  - Discriminator: Binary cross-entropy
  - Generator: Adversarial loss (BCE) + L1 loss (MAE) 
  - Loss weights: `[adversarial=1, L1=100]`
- **Data Preprocessing:** 
  - Image scaling: [0, 255] → [-1, 1]
- **Training Parameters:**
  - Epochs: 40
  - Batch size: 1
- **Model Saving:** Best generator saved as `pix2pix_best_generator.h5`

## Performance Metrics
### Training Progress
| Steps | Generator Loss (g_loss) |
|-------|-------------------------|
| 100   | 33.597                  |
| 500   | 19.035                  |
| 1000  | 16.547                  |
| 2000  | 12.605                  |
| 3000  | 10.300                  |
| 4000  | 8.839                   |
| 5000  | 7.660                   |
| 6000  | 6.813                   |
| 7000  | 6.207                   |
| 7838  | 5.816                   |

### Final Evaluation Metrics
| Metric | Value  |
|--------|--------|
| MSE    | 0.0648 |
| SSIM   | 0.8877 |
| PSNR   | 18.05  |

## Usage
1. **Prerequisites:** Install required libraries (see [Requirements](#requirements))
2. **Dataset Setup:**
   - Download [Fundus Image Dataset](https://www.kaggle.com/datasets/andrewmvd/fundus-image-vessel-segmentation)

3. **Execution Options:**
- **Local Jupyter:** Run notebook cells sequentially
- **Kaggle:** 
  1. Upload notebook to Kaggle Kernel
  2. Add dataset to notebook's data sources
  3. Path: `/kaggle/input/fundus-image-dataset-for-vessel-segmentation/`

