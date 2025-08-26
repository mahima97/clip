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
```
clip_scratch/
├── tokenizer.py        # Simple character-level BPE tokenizer
├── preprocess_img.py   # Image preprocessing (resize, crop, tensor)
├── utils.py            # Loss, normalization, retrieval metrics
├── data.py             # COCO-style dataset loader
├── clip_model.py       # Text encoder, Vision encoder, CLIP model
├── train_clip.py       # Training loop + resume + checkpointing
├── infer_clip.py       # Inference (zero-shot + retrieval)
└── README.md
```

---

## 🛠️ Installation
```bash
git clone https://github.com/<your-username>/clip_scratch.git
cd clip_scratch
pip install torch pillow
```

---

## 📑 Dataset Layout
Expect **COCO-style annotations.json**:
```
/PATH/TO/DATA/
  images/
    <file_name>.jpg        # must match entries in annotations.json
  annotations.json         # COCO captions format
```

Example `annotations.json` structure:
```json
{
  "images": [
    {"id": 1, "file_name": "000001.jpg"},
    {"id": 2, "file_name": "000002.jpg"}
  ],
  "annotations": [
    {"image_id": 1, "caption": "a brown dog running"},
    {"image_id": 1, "caption": "a dog in motion"},
    {"image_id": 2, "caption": "a cat sitting on sofa"}
  ]
}
```

---

## 🚀 Training

### Start training from scratch
```bash
python train_clip.py   --data_root /PATH/TO/DATA   --out_dir runs_clip_scratch   --epochs 20   --batch_size 32   --width 128 --layers 4 --heads 4 --embed_dim 128
```

### Resume automatically from the latest checkpoint
```bash
python train_clip.py --data_root /PATH/TO/DATA --out_dir runs_clip_scratch
```

### Resume from a specific checkpoint
```bash
python train_clip.py   --data_root /PATH/TO/DATA   --resume_from runs_clip_scratch/epoch_12.pt
```

### Disable auto-resume (always start fresh)
```bash
python train_clip.py --data_root /PATH/TO/DATA --no_auto_resume
```

---

## 📝 Features
- **Auto-resume** from the latest checkpoint in `--out_dir`
- **Save per-epoch checkpoints** (`epoch_<n>.pt`)
- **Track and save best model** (`best.pt`) using sum of retrieval recalls
- **All captions per image** option:
  ```bash
  python train_clip.py --all_captions_each_epoch
  ```

---

## 📊 Evaluation
During training, the script reports:
- **Loss**
- **Temperature** (exp(logit_scale))
- **Recall@1, Recall@5, Recall@10** for both Image→Text and Text→Image

Example log:
```
Epoch 3/20 | loss=2.1634 | R@1 (I→T)=0.320, R@5 (I→T)=0.770, ... | temp=42.1 | 45.2s
```

---

## 📦 Checkpoints
- `epoch_<n>.pt`: model + optimizer state at epoch n  
- `best.pt`: best validation retrieval score so far  
- `bpe.json`: saved tokenizer vocabulary/merges  

---

## 🔎 Inference

### Zero-shot classification
```bash
python infer_clip.py   --ckpt runs_clip_scratch/best.pt   --image /PATH/TO/IMAGE.jpg   --prompts "a photo of a dog,a photo of a cat,a photo of a car"
```

### Image→Text retrieval with a caption list
```bash
python infer_clip.py   --ckpt runs_clip_scratch/best.pt   --image /PATH/TO/IMAGE.jpg   --prompts_file /PATH/TO/captions.txt   --topk 10
```

> **Note:** The model architecture flags in `infer_clip.py` (e.g., `--width`, `--layers`) must match training.  
If you trained with the default values, the defaults in the script will work out of the box.

---

## 🔮 Next Steps
- Add more advanced inference (batch mode, evaluation against a full dataset)
- Try swapping VisionTransformer with CNN for sanity checks
- Scale dataset & training params for better results

---

## ⚠️ Notes
- With ~500 images, this is for **conceptual learning only**.  
- Real CLIP was trained on **400M+ pairs** for strong zero-shot performance.  
- Still, you can observe the mechanics of joint embedding learning on small data.
