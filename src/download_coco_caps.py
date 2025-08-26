#!/usr/bin/env python3
"""
download_coco_caps.py

Download a random subset of COCO images and their captions, save images to:
  coco_caps_data/coco_caps/
and write a simple JSON manifest to:
  coco_caps_data/annotations.json

The JSON is a list of records:
[
  {
    "file_name": "000000000123.jpg",
    "image_path": "coco_caps_data/coco_caps/000000000123.jpg",
    "captions": ["...", "...", ...]
  },
  ...
]

Notes
-----
- This script expects an **official COCO captions** file (e.g., captions_train2017.json)
  to load caption annotations via pycocotools.
- It does NOT write a COCO-style JSON; it writes a simple manifest suitable for your loader.
"""

from __future__ import annotations
import os
import io
import json
import random
import argparse
from typing import List, Dict, Any

import requests
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


def download_image_bytes(url: str, timeout: int = 20) -> bytes:
    """Fetch image bytes from a URL with basic error handling."""
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    return r.content


def save_rgb_jpeg(raw: bytes, save_path: str) -> None:
    """Decode image bytes with PIL, convert to RGB, and save as JPEG/PNG by extension."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)  # infers format from extension (.jpg/.png)


def choose_images(coco_caps: COCO, k: int, seed: int | None) -> List[Dict[str, Any]]:
    """Randomly sample k images (no replacement) and return their COCO image dicts."""
    img_ids_all = coco_caps.getImgIds()
    if seed is not None:
        random.seed(seed)
    ids = random.sample(img_ids_all, k=min(k, len(img_ids_all)))
    return coco_caps.loadImgs(ids)


def build_records(coco_caps: COCO, images: List[Dict[str, Any]], img_dir: str) -> List[Dict[str, Any]]:
    """Download images, fetch captions, and build manifest records."""
    records: List[Dict[str, Any]] = []
    for img in tqdm(images, desc="Downloading images + captions"):
        try:
            url = img["coco_url"]
            file_name = img["file_name"]
            save_path = os.path.join(img_dir, file_name)

            # Download + save
            raw = download_image_bytes(url)
            save_rgb_jpeg(raw, save_path)

            # Captions
            ann_ids = coco_caps.getAnnIds(imgIds=img["id"])
            anns = coco_caps.loadAnns(ann_ids)
            captions = [a["caption"] for a in anns if "caption" in a]

            records.append({
                "file_name": file_name,
                "image_path": save_path,
                "captions": captions
            })
        except Exception as e:
            print(f"[skip] {img.get('file_name', 'unknown')}: {e}")
    return records


def atomic_json_dump(obj: Any, out_path: str) -> None:
    """Write JSON atomically: to tmp then rename, to avoid truncated files."""
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)


def main():
    """
    CLI entry point.

    Example:
        python download_coco_caps.py \\
          --ann-file /path/to/captions_train2017.json \\
          --out-dir coco_caps_data \\
          --n 500 --seed 42
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-file", type=str, required=True,
                    help="Path to official COCO captions JSON (e.g., captions_train2017.json)")
    ap.add_argument("--out-dir", type=str, default="coco_caps_data",
                    help="Output root directory (images + annotations.json)")
    ap.add_argument("--n", type=int, default=500,
                    help="Number of images to sample")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible sampling")
    args = ap.parse_args()

    # Prepare paths
    img_dir = os.path.join(args.out_dir, "coco_caps")
    os.makedirs(img_dir, exist_ok=True)
    annotations_path = os.path.join(args.out_dir, "annotations.json")

    # Load COCO captions file (official schema)
    coco_caps = COCO(args.ann_file)

    # Choose images
    images_subset = choose_images(coco_caps, k=args.n, seed=args.seed)

    # Build records: download images + collect captions
    records = build_records(coco_caps, images_subset, img_dir)

    # Save annotations.json atomically
    atomic_json_dump(records, annotations_path)

    print(f"✅ Done.\n   Images: {img_dir}\n   Manifest: {annotations_path}\n   Samples: {len(records)}")


if __name__ == "__main__":
    main()
