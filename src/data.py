# data.py
"""
Dataset class for COCO-style image-caption data.
Each sample returns an image tensor and tokenized caption IDs.

Supports:
- Randomly sampling one caption per image (default, matches CLIP training).
- Expanding all captions per image as separate samples (--all_captions_each_epoch).
"""

import os, json, random
import torch
from torch.utils.data import Dataset
from PIL import Image
from preprocess_img import ImagePreprocess


class CocoClipDataset(Dataset):
    """
    Dataset wrapper for COCO-style annotations used in CLIP training.

    Args:
        data_root (str): Root directory containing "images/" and annotations file.
        ann_path (str): Path to annotations.json (COCO captions format).
        bpe (SimpleBPE): Tokenizer object for encoding captions.
        ctx_len (int): Maximum sequence length for tokenized captions.
        image_size (int): Size for image preprocessing (default=224).
        train (bool): If True, randomly samples one caption per image each epoch.
                      If False, always uses the first caption.
        all_captions_each_epoch (bool): If True, expands dataset so each caption
                                        is a separate sample.

    Raises:
        ValueError: If no valid image/caption pairs are found.
    """

    def __init__(self, data_root, ann_path, bpe,
                 ctx_len=77, image_size=224,
                 train=True, all_captions_each_epoch=False):
        self.bpe = bpe
        self.ctx_len = ctx_len
        self.pre = ImagePreprocess(size=image_size)
        self.train = train
        self.all_caps = all_captions_each_epoch

        # Load annotations
        if not os.path.isabs(ann_path):
            ann_path = os.path.join(data_root, ann_path)
        with open(ann_path, "r") as f:
            data = json.load(f)

        # Map image_id -> filename
        id2file = {im["id"]: im["file_name"] for im in data["images"]}

        # Collect captions per file
        file2caps = {}
        for a in data["annotations"]:
            fn = id2file.get(a["image_id"])
            if fn is None: 
                continue
            file2caps.setdefault(fn, []).append(a["caption"])

        # Build dataset items
        img_dir = os.path.join(data_root, "images")
        self.items = []
        if self.all_caps:
            # Expand each caption into its own training pair
            for fn, caps in file2caps.items():
                fp = os.path.join(img_dir, fn)
                if not os.path.isfile(fp): 
                    continue
                for cap in caps:
                    self.items.append((fp, cap))
        else:
            # One entry per image; captions list is kept
            for fn, caps in file2caps.items():
                fp = os.path.join(img_dir, fn)
                if not os.path.isfile(fp): 
                    continue
                self.items.append((fp, caps))

        if not self.items:
            raise ValueError("No image/caption pairs found. Check paths.")

    def __len__(self):
        """
        Returns:
            int: Number of items in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Load a single sample (image tensor + caption IDs).

        Args:
            idx (int): Index into dataset.

        Returns:
            tuple:
                - image (torch.FloatTensor): Shape [3, H, W], values in [0,1].
                - ids (torch.LongTensor): Tokenized caption IDs [ctx_len].
                - fp (str): Path to the image file.
                - cap (str): The raw caption string.
        """
        if self.all_caps:
            fp, cap = self.items[idx]
        else:
            fp, caps = self.items[idx]
            cap = random.choice(caps) if self.train else caps[0]

        arr = self.pre(Image.open(fp).convert("RGB"))  # CHW in [0,1]
        ids = self.bpe.encode(cap, max_len=self.ctx_len)

        return torch.from_numpy(arr).float(), torch.tensor(ids, dtype=torch.long), fp, cap
