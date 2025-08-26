"""
infer_clip.py

This script provides inference utilities for a minimal CLIP model
(Contrastive Language–Image Pretraining). It allows performing:

1. Zero-shot classification:
   Given an image and a set of candidate text prompts, the script ranks the prompts
   according to similarity with the image embedding.

2. Image-to-text retrieval:
   Given an image and a list of candidate captions, the script retrieves the top-k
   most similar captions.

Usage examples:
---------------
Zero-shot classification:
    python infer_clip.py \
        --ckpt runs_clip_scratch/best.pt \
        --image /PATH/TO/IMAGE.jpg \
        --prompts "a photo of a dog,a photo of a cat,a photo of a car"

Image→Text retrieval:
    python infer_clip.py \
        --ckpt runs_clip_scratch/best.pt \
        --image /PATH/TO/IMAGE.jpg \
        --prompts_file /PATH/TO/captions.txt \
        --topk 10
"""

import torch
import argparse
from PIL import Image

from clip_model import CLIP
from preprocess_img import get_transform
from tokenizer import BPETokenizer


def load_model(ckpt_path: str, device: str = "cpu") -> CLIP:
    """
    Load a trained CLIP model checkpoint.

    Args:
        ckpt_path (str): Path to the model checkpoint (.pt file).
        device (str): Device to load the model on ("cpu" or "cuda").

    Returns:
        CLIP: The loaded CLIP model ready for inference.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["model_cfg"]
    model = CLIP(**model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(img_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Load and preprocess a single image for CLIP inference.

    Args:
        img_path (str): Path to the input image.
        device (str): Device to put the tensor on.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, C, H, W).
    """
    img = Image.open(img_path).convert("RGB")
    transform = get_transform()
    return transform(img).unsqueeze(0).to(device)


def encode_texts(texts: list, tokenizer: BPETokenizer, model: CLIP, device: str = "cpu") -> torch.Tensor:
    """
    Tokenize and encode a list of text prompts using the CLIP text encoder.

    Args:
        texts (list): A list of text strings.
        tokenizer (BPETokenizer): Tokenizer instance to convert text into token IDs.
        model (CLIP): Trained CLIP model with a text encoder.
        device (str): Device to put the tensors on.

    Returns:
        torch.Tensor: Normalized embeddings of the text prompts.
    """
    token_ids = [tokenizer.encode(t) for t in texts]
    max_len = max(len(t) for t in token_ids)
    padded = [t + [0] * (max_len - len(t)) for t in token_ids]
    text_tensor = torch.tensor(padded).to(device)
    with torch.no_grad():
        return model.encode_text(text_tensor)


def zero_shot_classification(image: torch.Tensor, prompts: list, tokenizer: BPETokenizer,
                             model: CLIP, device: str = "cpu") -> list:
    """
    Perform zero-shot classification of an image against candidate prompts.

    Args:
        image (torch.Tensor): Preprocessed image tensor of shape (1, C, H, W).
        prompts (list): A list of candidate prompt strings.
        tokenizer (BPETokenizer): Tokenizer for text encoding.
        model (CLIP): Trained CLIP model.
        device (str): Device for computation.

    Returns:
        list: Sorted list of (prompt, similarity score) tuples.
    """
    with torch.no_grad():
        img_emb = model.encode_image(image)
        txt_emb = encode_texts(prompts, tokenizer, model, device)
        sims = (img_emb @ txt_emb.T).squeeze(0)
        ranked = sorted(zip(prompts, sims.tolist()), key=lambda x: x[1], reverse=True)
        return ranked


def main():
    """
    Command-line entry point for running CLIP inference.

    Supports:
    - Zero-shot classification with `--prompts`
    - Image→Text retrieval with `--prompts_file`
    """
    parser = argparse.ArgumentParser(description="CLIP Inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Comma-separated list of prompts for zero-shot classification")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="File containing one prompt per line for retrieval")
    parser.add_argument("--topk", type=int, default=5, help="Number of top prompts to show")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    args = parser.parse_args()

    device = args.device
    model = load_model(args.ckpt, device=device)
    tokenizer = BPETokenizer.load("bpe.json")
    image = preprocess_image(args.image, device=device)

    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",")]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        raise ValueError("Either --prompts or --prompts_file must be provided.")

    results = zero_shot_classification(image, prompts, tokenizer, model, device)

    print("\nTop predictions:")
    for p, score in results[: args.topk]:
        print(f"{p}: {score:.3f}")


if __name__ == "__main__":
    main()
