# MODL-CAIR
A Multi-Operator Deep-Learning Approach to Preserve Visual Integrity in Content-Aware Image Retargeting Task

A Multi-Operator Deep-Learning Approach to Preserve Visual Integrity in Content-Aware Image Retargeting Task

1. Overview
This is the implementation of MODL-CAIR that reproduces the proposed image retargeting method described in our manuscript.

The manuscript is available at:
Manuscript: To be updated after publication.

2. System Requirements
2.1. Hardware Requirements
The code has been tested on a standard desktop computer. However, a CUDA-compatible GPU is recommended to accelerate Mask R-CNN inference, although the code also supports CPU execution.

2.2. Software Requirements
The source code has been tested with:
- Python 3.11
- torch 2.8.0
- torchvision 0.23.0
- numpy 1.26.4
- Pillow 11.0.0
- imageio 2.36.1
- numba 0.61.2

3. Installation Guide
3.1. Clone or download this repository.
3.2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

The installation typically takes a few minutes depending on the internet connection and hardware configuration.

4. Usage
Open “modl_cair.py” and specify the following parameters:

```python
INPUT_PATH = "/path/to/input/image.png"
OUTPUT_PATH = "/path/to/output/image.png"
SCALE = 0.75
CONFIDENCE_THRESHOLD = 0.2
```

Run the “modl_cair.py” script:

```bash
python modl_cair.py
```

The retargeted image will be saved to the location specified by “OUTPUT_PATH”.

5. Citation
If you use this source code in your research, please cite the corresponding publication.
Citation information will be updated after publication.
<img width="468" height="644" alt="image" src="https://github.com/user-attachments/assets/d60ce59e-25f3-43df-b9c6-2dda0ac5ac2a" />
