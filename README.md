# auto_hand_tracking
Automatic Hand Tracking Demo with SAM2

This project demonstrates how to process a video using the SAM2 video predictor and MediaPipe Hand Landmarker. It includes frame extraction from mp4 video, hand landmark detection using mediapipe, mask generation using SAM2, and rendering the processed frames back into a video.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)

---

## Overview

This pipeline processes videos to detect hand landmarks, generates segmentation masks using SAM2, and reassembles the processed frames into an annotated video.
---

## Features

- **Frame Extraction**: Decomposes a video into individual frames.
- **Hand Detection**: Detects and visualizes hand landmarks with annotations.
- **SAM2 Mask Generation**: Generates and propagates segmentation masks across frames.
- **Masked Frame Rendering**: Saves processed frames with masks applied.
- **Video Creation**: Combines processed frames back into a video.

---

## Installation

### Prerequisites
- Python 3.12
- CUDA-enabled GPU (for faster processing)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/declanraj/auto_hand_tracking.git
   cd auto_hand_tracking
2. Install Dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate sam

   # Install specific Torch versions
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Install SAM2 model
   git clone https://github.com/facebookresearch/sam2.git && cd sam2
   pip install -e .

   # Download model checkpoints:
   cd checkpoints && \
   ./download_ckpts.sh && \
   cd ..
   ```
## Usage
Run script.py to generate masked video
   ```bash
   cd ..
   # Ensure the video of interest (test.mp4) is in the source_video folder
   mv script.py ./sam2/script.py
   cd sam2
   # Download the Hand Landmark model
   !wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

   python -u script.py
   ```