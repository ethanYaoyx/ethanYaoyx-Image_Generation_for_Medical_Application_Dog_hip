# RTMpose: Dog Hip Keypoints Detection

The aim of this study is to utilize the Keypoints Detection model to predict the Norberg Angle prediction model. The Norberg Angle is illustrated in the figure below. This study build a custom model based on RTMpose and RTMdetection.

<img src="https://github.com/YoushanZhang/AiAI/assets/74528993/3c3fd898-7857-4f2a-88fd-723165ddfb4f" width="450" height="250">



## Overview

RTMpose is an advanced model designed to accurately detect keypoints on dog hip radiographs. The project is based on the RTMDet and MMPose frameworks, specifically fine-tuned for keypoint detection tasks. This repository contains the training code, model configurations, and evaluation metrics used to achieve state-of-the-art performance in detecting dog hip landmarks.

## Table of Contents

- [Introduction](#introduction)
- [GetStarted](#GetStarted)
- [Usage](#usage)
- [Training](#training)
- [prediction](#prediction)
- [Results](#results)

## Introduction

The RTMpose model is part of a research project aimed at improving the accuracy and efficiency of detecting keypoints on canine hip radiographs. The project leverages advanced techniques in deep learning, including data augmentation, transfer learning, and custom model architectures, to handle the variability in radiographic images.

## GetStarted

To get started, Ensure you have the following installed:

Python 3.7
PyTorch 1.10

Next, run the following notebooks in sequence to set up the necessary dependencies:
1. 'Install_MMDetection.ipynb'
2. 'Install_MMPose.ipynb'

## Usage
The dataset link: https://drive.google.com/drive/folders/1K3w8p0B5WuNIce2RXmrMN3LIKb12DXXp?usp=sharing.

Here is an overview of the dataset used in this project:

|                     | **Test(set3)** | **Valid(set2)** | **Training(set1)** | **More Images(set4)** | **DDPM** | **Dreambooth** | **Stable Diffusion** | **new** | **total** |
|---------------------|----------------|-----------------|--------------------|-----------------------|----------|----------------|----------------------|---------|-----------|
| **For Train**       | 84             | 31              | 219                | 608                   | 358      | 967            | 200                  | 0       | 2467      |
| **For Valid**       | 0              | 0               | 0                  | 105                   | 0        | 0              | 0                    | 0       | 105       |
| **For Test**        | 0              | 0               | 0                  | 0                     | 0        | 0              | 0                    | 121     | 121       |
| **total**           | 84             | 31              | 219                | 713                   | 358      | 967            | 200                  | 121     | 2693      |

The weight link: 

## training
All model configurations are stored in the configs/ directory.
First for training box object detection:

```python
import os
os.chdir('mmdetection')
!python tools/train.py data/rtmdet_m_dog_hip_all.py
```
Second for training keypoint detection:
```python
import os
os.chdir('mmpose')
!python tools/train.py data/rtmpose_m_dog_hip_all.py
```
## prediction
You can find the prediction in the 'prediction_keypoint_.ipynb' file.

## Results
 **Model**      | **RTMpose-s** | **RTMpose-l** | **RTMpose-m** | **RTMpose-X** |
|----------------|---------------|---------------|---------------|---------------|
| **MAE Angle1** | 3.74          | 4.16          | 3.304         | 3.54          |
| **MAE Angle2** | 3.018         | 3.25          | 2.994         | 2.95          |
| **MAE Angle**  | 3.38          | 3.71          | 3.149         | 3.25          |
| **MSE point**  | 8.046         | 8.643         | 8.609         | 8.511         |
| **MSE Angle**  | 16.52         | 20.61         | 15.26         | 15.31         |
| **MSE Radius** | 6.98          | 6.91          | 5.72          | 6.99          |
| **MAPE Angle** | 0.0405        | 0.0415        | 0.036         | 0.06          |
| **MAPE point** | 0.0123        | 0.0127        | 0.012         | 0.014         |
| **MAPE Radius**| 0.0357        | 0.036         | 0.031         | 0.035         |
| **R2 Angle**   | 0.942         | 0.91          | 0.884         | 0.806         |
| **Error < 1**  | 16.53%        | 15.00%        | 19.01%        | 15.07%        |
| **Error < 5**  | 77.69%        | 95.83%        | 80.17%        | 79.75%        |
| **Error < 10** | 99.59%        | 99.71%        | 99.17%        | 99.59%        |
