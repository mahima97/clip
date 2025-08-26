# train_clip.py
"""
Training script for CLIP-from-scratch with:
- Auto-resume from latest checkpoint or a user-specified path
- Best-so-far checkpointing (best.pt)
- Per-epoch checkpoints (epoch_<n>.pt)
"""

import os, re, json, time, random, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from tokenizer import SimpleBPE
from data import CocoClipDataset
from clip_model import CLIP
from utils import clip_contrastive_loss, clip_normalize_inplace, eval_retrieval


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gather_captions(data_root: str, ann: str) -> list:
    """Load all caption strings from COCO-style annotations."""
    ann_path = ann if os.path.isabs(ann) else os.path.join(data_root, ann)
    with open(ann_path, "r") as f:
        data = json.load(f)
    return [a["caption"] for a in data["annotations"] if a.get("caption")]


def evaluate(model, val_loader, device) -> dict:
    """
    Evaluate model on validation split using retrieval Recall@K.
    Returns a dict with R@1/5/10 for I→T and T→I.
    """
    model.eval()
    feats_i, feats_t = [], []
    with torch.no_grad():
        for imgs, ids, _, _ in val_loader:
            imgs, ids = imgs.to(device), ids.to(device)
            clip_normalize_inplace(imgs)
            feats_i.append(model.encode_image(imgs))
            feats_t.append(model.encode_text(ids))
    I = torch.cat(feats_i, 0)
    T = torch.cat(feats_t, 0)
    sim = model.logit_scale.exp() * (I @ T.t())
    return eval_retrieval(sim)


def find_latest_checkpoint(out_dir: str) -> str | None:
    """
    Return the path to the latest epoch_*.pt checkpoint in out_dir, if any.
    """
    if not os.path.isdir(out_dir):
        return None
    patt = re.compile(r"^epoch_(\d+)\.pt$")
    epochs = []
    for fn in os.listdir(out_dir):
        m = patt.match(fn)
        if m:
            epochs.append((int(m.group(1)), os.path.join(out_dir, fn)))
    if not epochs:
        return None
    epochs.sort(key=lambda x: x[0])
    return epochs[-1][1]  # latest


def load_checkpoint_if_any(model, optimizer, resume_path: str | None) -> tuple[int, float]:
    """
    Load model/optimizer from checkpoint if resume_path is provided.
    Also returns:
      - start_epoch (int): next epoch index to run (last_epoch + 1)
      - best_score (float): best validation score so far (from best.pt if found)
    """
    start_epoch = 0
    if resume_path and os.path.isfile(resume_path):
        ck = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        if optimizer is not None and "opt" in ck:
            optimizer.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        print(f"[Resume] Loaded checkpoint: {resume_path} (next epoch = {start_epoch})")
    else:
        if resume_path:
            print(f"[Resume] Path not found, starting fresh: {resume_path}")

    # Try to fetch best score from best.pt if it exists
    best_score = -1.0
    best_path = os.path.join(os.path.dirname(resume_path) if resume_path else ".", "best.pt")
    if os.path.isfile(best_path):
        try:
            b = torch.load(best_path, map_location="cpu")
            # score isn't stored, but we can keep placeholder > -inf to continue tracking;
            # user will overwrite best.pt when validation improves.
            print(f"[Best] Found existing best checkpoint: {best_path}")
        except Exception:
            pass
    return start_epoch, best_score


def main():
    """
    Main:
    - Parse CLI args
    - Train/load tokenizer
    - Build dataset/dataloaders
    - (Auto-)resume if checkpoints exist
    - Train with per-epoch and best-so-far checkpointing
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Path containing images/ and annotations.json")
    ap.add_argument("--ann", type=str, default="annotations.json",
                    help="Annotations JSON file (relative to data_root by default)")
    ap.add_argument("--out_dir", type=str, default="runs_clip_scratch",
                    help="Directory to save outputs/checkpoints")
    ap.add_argument("--image_size", type=int, default=224, help="Square input size")
    ap.add_argument("--patch_size", type=int, default=16, help="ViT patch size")
    ap.add_argument("--ctx_len", type=int, default=77, help="Max text tokens")
    ap.add_argument("--vocab_size", type=int, default=2000, help="BPE vocab size")
    ap.add_argument("--width", type=int, default=256, help="Hidden size")
    ap.add_argument("--layers", type=int, default=6, help="Transformer layers")
    ap.add_argument("--heads", type=int, default=8, help="Attention heads")
    ap.add_argument("--embed_dim", type=int, default=256, help="Joint embedding dim")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size")
    ap.add_argument("--epochs", type=int, default=20, help="Total epochs to train")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--val_split", type=float, default=0.15, help="Validation fraction")
    ap.add_argument("--all_captions_each_epoch", action="store_true",
                    help="Expand each caption -> separate sample each epoch")
    ap.add_argument("--no_auto_resume", action="store_true",
                    help="Disable auto-resume; start from scratch unless --resume_from is set")
    ap.add_argument("--resume_from", type=str, default="",
                    help="Path to a checkpoint to resume from (overrides auto-resume)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # ---------- Tokenizer ----------
    bpe_path = os.path.join(args.out_dir, "bpe.json")
    if os.path.isfile(bpe_path):
        bpe = SimpleBPE.load(bpe_path)
        print(f"[BPE] Loaded tokenizer ({len(bpe.id2tok)} tokens) from {bpe_path}")
    else:
        texts = gather_captions(args.data_root, args.ann)
        bpe = SimpleBPE(vocab_size=args.vocab_size)
        bpe.train(texts)
        bpe.save(bpe_path)
        print(f"[BPE] Trained + saved tokenizer to {bpe_path} (vocab={len(bpe.id2tok)})")

    # ---------- Dataset & loaders ----------
    full = CocoClipDataset(
        data_root=args.data_root, ann_path=args.ann, bpe=bpe,
        ctx_len=args.ctx_len, image_size=args.image_size, train=True,
        all_captions_each_epoch=args.all_captions_each_epoch
    )
    n = len(full)
    n_val = max(1, int(n * args.val_split))
    n_train = n - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=g)
    train_set.dataset.train = True
    val_set.dataset.train = False

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

    # ---------- Model & Optimizer ----------
    model = CLIP(
        vocab_size=len(bpe.id2tok), ctx_len=args.ctx_len,
        img_size=args.image_size, patch_size=args.patch_size,
        width=args.width, layers=args.layers, heads=args.heads, embed_dim=args.embed_dim
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # ---------- Resume logic ----------
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif not args.no_auto_resume:
        # auto: prefer latest epoch_*.pt in out_dir
        maybe = find_latest_checkpoint(args.out_dir)
        if maybe:
            resume_path = maybe

    start_epoch = 0
    best_score = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")

    if resume_path:
        start_epoch, _ = load_checkpoint_if_any(model, opt, resume_path)
        # If a previous best exists, we will keep tracking from there.
        if os.path.isfile(best_path):
            print(f"[Best] Continuing best tracking using existing {best_path}")

    # ---------- Training loop ----------
    total_epochs = args.epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss, steps, t0 = 0.0, 0, time.time()

        for imgs, ids, _, _ in train_loader:
            imgs, ids = imgs.to(device, non_blocking=True), ids.to(device, non_blocking=True)
            clip_normalize_inplace(imgs)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits_i, logits_t = model(imgs, ids)
                loss = clip_contrastive_loss(logits_i, logits_t)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        metrics = evaluate(model, val_loader, device)
        msg = ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
        print(f"Epoch {epoch+1}/{total_epochs} | loss={avg_loss:.4f} | {msg} "
              f"| temp={model.logit_scale.exp().item():.3f} | {(time.time()-t0):.1f}s")

        # Save per-epoch checkpoint
        ck = os.path.join(args.out_dir, f"epoch_{epoch+1}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "opt": opt.state_dict(), "bpe": bpe_path}, ck)

        # Save best-so-far
        score = sum(metrics.values())
        if score > best_score:
            best_score = score
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "opt": opt.state_dict(), "bpe": bpe_path}, best_path)
            print(f"  New best {best_score:.3f} -> {best_path}")


if __name__ == "__main__":
    main()
