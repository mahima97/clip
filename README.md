# CLIP From Scratch (PyTorch, No External Libraries)

This project implements a **minimal CLIP model** (Contrastive Language–Image Pretraining) from scratch:
- **Byte-Pair Encoding (BPE)** tokenizer for text
- **Vision Transformer (ViT, tiny)** image encoder
- **Text Transformer** encoder
- **Contrastive loss** to align images and text in a shared embedding space

It is designed for **learning the concepts**, not for SOTA performance.  
With small datasets (e.g., 500 COCO-style images with 5 captions each), you can still observe training dynamics and retrieval behavior.

---

## 📂 Project Structure
