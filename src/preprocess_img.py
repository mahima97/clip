# preprocess_img.py
"""
Image preprocessing utilities:
- Resize by shorter side
- Center crop to a square
- Convert to CHW float32 tensor in [0, 1]
"""

from PIL import Image
import numpy as np

def resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    """
    Resize so the shorter side equals `size`, preserving aspect ratio.
    """
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_h, new_w = size, int(w * size / h)
    return img.resize((new_w, new_h), Image.BICUBIC)

def center_crop(img: Image.Image, size: int) -> Image.Image:
    """
    Center-crop the image to a square of side `size`.
    Assumes the image is at least `size` in both dimensions.
    """
    w, h = img.size
    left = int((w - size) / 2)
    top = int((h - size) / 2)
    return img.crop((left, top, left + size, top + size))

class ImagePreprocess:
    """
    Callable image preprocessor for CLIP-style inputs.
    Converts a PIL image to a (C,H,W) float32 array in [0,1].
    """
    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, img: Image.Image):
        img = img.convert("RGB")
        img = resize_shorter_side(img, self.size)
        img = center_crop(img, self.size)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return arr.transpose(2, 0, 1)  # CHW
