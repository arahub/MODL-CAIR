# MODL-CAIR

## Overview

This is the implementation of MODL-CAIR that reproduces the proposed image retargeting method described in our manuscript.

**Manuscript:**

https://publication.com/ *(To be updated after publication)*

A Multi-Operator Deep-Learning Approach to Preserve Visual Integrity in Content-Aware Image Retargeting Task

---

## System Requirements

### 1. Hardware Requirements

The code has been tested on a standard desktop computer. A CUDA-compatible GPU is recommended to accelerate Mask R-CNN inference, although the code also supports CPU execution.

### 2. Software Requirements

The source code has been tested with:

- Python 3.11
- torch 2.8.0
- torchvision 0.23.0
- numpy 1.26.4
- Pillow 11.0.0
- imageio 2.36.1
- numba 0.61.2

---

## Installation Guide

1. Clone or download this repository.

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

The installation typically takes a few minutes depending on the internet connection and hardware configuration.

---

## Usage

Open `modl_cair.py` and specify the following parameters:

```python
INPUT_PATH = "/path/to/input/image.png"
OUTPUT_PATH = "/path/to/output/image.png"
SCALE = 0.75
CONFIDENCE_THRESHOLD = 0.2
```

Run the modl_cair.py script:

```bash
python modl_cair.py
```

The retargeted image will be saved to the location specified by `OUTPUT_PATH`.

---

## Citation

If you use this source code in your research, please cite the corresponding publication.

```
Citation information will be updated after publication.
```
