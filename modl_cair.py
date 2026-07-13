import os
import numpy as np
import imageio.v3 as iio
from PIL import Image
import torch
from torchvision import models
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from numba import njit

# parameters
INPUT_PATH = "/path/to/your/image.png"
OUTPUT_PATH = "/path/to/output/result.png"
SCALE = 0.75
CONFIDENCE_THRESHOLD = 0.2

# Mask R-CNN
model_maskrcnn = None
device = None
transform = None

def initialize_maskrcnn():
    global model_maskrcnn, device, transform
    if model_maskrcnn is None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model_maskrcnn = models.detection.maskrcnn_resnet50_fpn(weights=weights)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_maskrcnn.to(device)
        model_maskrcnn.eval()
        transform = weights.transforms()

def generate_mask(image_np, conf_threshold=0.2):
    global model_maskrcnn, device, transform
    if model_maskrcnn is None:
        initialize_maskrcnn()
    pil_image = Image.fromarray(image_np)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model_maskrcnn(image_tensor)[0]
    scores = predictions['scores'].cpu().numpy()
    masks = predictions['masks'].cpu().numpy()
    keep = scores >= conf_threshold
    masks = masks[keep]
    combined_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    for i in range(masks.shape[0]):
        m = masks[i, 0] > 0.5
        combined_mask = np.where(m, 255, combined_mask)
    return combined_mask

# Sobel kernels
_SOBEL_DU = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]], dtype=np.float32)
_SOBEL_DV = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]], dtype=np.float32)

@njit(cache=True, fastmath=False, parallel=False)
def _reflect_index(i, n):
    if i < 0:
        return -i - 1
    if i >= n:
        return 2 * n - i - 1
    return i

# energy calculation
@njit(cache=True, fastmath=False, parallel=False)
def calc_energy(img):
    r, c, ch = img.shape
    out = np.zeros((r, c), np.float32)
    for i in range(r):
        for j in range(c):
            total_sq = np.float32(0.0)
            for k in range(ch):
                gx = np.float32(0.0)
                gy = np.float32(0.0)
                for di in range(-1, 2):
                    ii = _reflect_index(i + di, r)
                    ki = di + 1
                    for dj in range(-1, 2):
                        jj = _reflect_index(j + dj, c)
                        kj = dj + 1
                        val = np.float32(img[ii, jj, k])
                        gx += _SOBEL_DU[ki, kj] * val
                        gy += _SOBEL_DV[ki, kj] * val
                total_sq += gx * gx + gy * gy
            out[i, j] = np.sqrt(total_sq)
    return out

# Dynamic programming
@njit(cache=True, fastmath=False, parallel=False)
def forward_energy_minimum_seam(img, banned):
    r, c, ch = img.shape
    M = np.zeros((r, c), dtype=np.float32)
    back = np.zeros((r, c), dtype=np.int32)
    INF = np.float32(1e10)
    for j in range(c):
        M[0, j] = banned[0, j]
        back[0, j] = 0
    for i in range(1, r):
        for j in range(c):
            jl = _reflect_index(j - 1, c)
            jr = _reflect_index(j + 1, c)
            hdiff_sq = np.float32(0.0)
            for k in range(ch):
                diff = np.float32(img[i, jr, k]) - np.float32(img[i, jl, k])
                hdiff_sq += diff * diff
            hdiff = np.sqrt(hdiff_sq)
            i_prev = _reflect_index(i - 1, r)
            if j > 0:
                vdiff_sq = np.float32(0.0)
                for k in range(ch):
                    diff = np.float32(img[i_prev, j, k]) - np.float32(img[i, jl, k])
                    vdiff_sq += diff * diff
                cost_left = M[i-1, j-1] + hdiff + np.sqrt(vdiff_sq) + banned[i, j]
            else:
                cost_left = INF
            cost_up = M[i-1, j] + hdiff + banned[i, j]
            if j < c-1:
                vdiff_sq = np.float32(0.0)
                for k in range(ch):
                    diff = np.float32(img[i_prev, j, k]) - np.float32(img[i, jr, k])
                    vdiff_sq += diff * diff
                cost_right = M[i-1, j+1] + hdiff + np.sqrt(vdiff_sq) + banned[i, j]
            else:
                cost_right = INF
            if cost_left <= cost_up and cost_left <= cost_right:
                M[i, j] = cost_left
                back[i, j] = j-1
            elif cost_up <= cost_left and cost_up <= cost_right:
                M[i, j] = cost_up
                back[i, j] = j
            else:
                M[i, j] = cost_right
                back[i, j] = j+1
    return M, back

# Backtrack & seam validation
@njit(cache=True, fastmath=False, parallel=False)
def backtrack_and_check(M, back, mask):
    r, c = M.shape
    seam = np.zeros(r, np.int32)
    j_min = 0
    v_min = M[r-1, 0]
    for j in range(1, c):
        if M[r-1, j] < v_min:
            v_min = M[r-1, j]
            j_min = j
    seam[r-1] = j_min
    bad = (mask[r-1, j_min] == 255)
    j = j_min
    for i in range(r-2, -1, -1):
        j = back[i+1, j]
        if j < 0:
            j = 0
        elif j >= c:
            j = c-1
        seam[i] = j
        if mask[i, j] == 255:
            bad = True
    return seam, (not bad)

# Seam removal
@njit(cache=True, fastmath=False, parallel=False)
def remove_seam_numba(img, mask, seam):
    r, c, ch = img.shape
    new_img = np.zeros((r, c-1, ch), img.dtype)
    new_mask = np.zeros((r, c-1), mask.dtype)
    for i in range(r):
        j_seam = seam[i]
        for j in range(j_seam):
            for k in range(ch):
                new_img[i, j, k] = img[i, j, k]
            new_mask[i, j] = mask[i, j]
        for j in range(j_seam, c-1):
            for k in range(ch):
                new_img[i, j, k] = img[i, j+1, k]
            new_mask[i, j] = mask[i, j+1]
    return new_img, new_mask

# column removal
def crop_c(img, mask, scale_c):
    r, c0, _ = img.shape
    target = int(c0 * scale_c)
    remove_count = c0 - target
    banned = np.zeros((r, c0), np.float32)
    col_mapping = np.tile(np.arange(c0), (r, 1))

    removed_seam_count = 0
    for _ in range(remove_count):
        M, back = forward_energy_minimum_seam(img, banned)
        seam, ok = backtrack_and_check(M, back, mask)
        if ok:
            removed_seam_count += 1
            new_col_mapping = np.zeros((r, col_mapping.shape[1]-1), dtype=np.int32)
            for i in range(r):
                js = seam[i]
                new_col_mapping[i, :js] = col_mapping[i, :js]
                new_col_mapping[i, js:] = col_mapping[i, js+1:]
            col_mapping = new_col_mapping
            img, mask = remove_seam_numba(img, mask, seam)
            new_banned = np.zeros((r, img.shape[1]), np.float32)
            for i in range(r):
                js = seam[i]
                new_banned[i, :js] = banned[i, :js]
                new_banned[i, js:] = banned[i, js+1:]
            banned = new_banned
        else:
            for i in range(r):
                banned[i, seam[i]] = np.float32(1e10)

    if removed_seam_count < remove_count:
        current_width = img.shape[1]
        img_pil = Image.fromarray(img.astype('uint8'))
        img_pil = img_pil.resize((target, r), Image.LANCZOS)
        img = np.array(img_pil)
    return img

# row removal
def crop_r(img, mask, scale_r):
    t_img = np.transpose(img, (1, 0, 2))
    t_mask = mask.T
    out = crop_c(t_img, t_mask, scale_r)
    out = np.transpose(out, (1, 0, 2))
    return out

# main
def main():
    img = iio.imread(INPUT_PATH)
    mask = generate_mask(img, CONFIDENCE_THRESHOLD)

    result = crop_c(img, mask, SCALE)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    iio.imwrite(OUTPUT_PATH, result, extension='.png')

if __name__ == '__main__':
    main()
