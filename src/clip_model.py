# clip_model.py
"""
Neural network architectures for CLIP-from-scratch:
- Transformer blocks (attention, MLP, LayerNorm)
- Text encoder (Transformer)
- Vision encoder (tiny ViT)
- CLIP model wrapper combining both
"""

import math
import torch
import torch.nn as nn
from utils import cosine_normalize


class LayerNorm(nn.Module):
    """
    Custom LayerNorm operating over the last dimension.

    Args:
        d (int): Feature dimension.
        eps (float): Numerical stability epsilon.
    """
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        """
        Normalize input tensor along last dimension.

        Args:
            x (torch.Tensor): Input [..., d].

        Returns:
            torch.Tensor: Normalized output of same shape.
        """
        m = x.mean(-1, keepdim=True)
        v = x.var(-1, keepdim=True, unbiased=False)
        return (x - m) / (v + self.eps).sqrt() * self.weight + self.bias


class MLP(nn.Module):
    """
    Two-layer feed-forward network used inside Transformer.

    Args:
        d (int): Model dimension.
        mlp_ratio (float): Expansion factor for hidden layer.
        dropout (float): Dropout probability.
    """
    def __init__(self, d, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        h = int(d * mlp_ratio)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention layer.

    Args:
        d (int): Model dimension.
        heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x (torch.Tensor): Input [B, N, d].
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            torch.Tensor: Output [B, N, d].
        """
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t): return t.view(B, N, self.h, self.dk).transpose(1, 2)
        q, k, v = map(split, (q, k, v))

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.drop(self.out(out))


class TransformerBlock(nn.Module):
    """
    Transformer encoder block: Pre-LN + Attention + MLP.

    Args:
        d (int): Model dimension.
        heads (int): Number of attention heads.
        mlp_ratio (float): Hidden size expansion in MLP.
        dropout (float): Dropout probability.
    """
    def __init__(self, d, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = LayerNorm(d)
        self.attn = MultiheadSelfAttention(d, heads, dropout)
        self.ln2 = LayerNorm(d)
        self.mlp = MLP(d, mlp_ratio, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TextTransformer(nn.Module):
    """
    Text encoder: Transformer over token embeddings.

    Args:
        vocab_size (int): Size of tokenizer vocabulary.
        ctx_len (int): Max sequence length.
        width (int): Hidden dimension.
        layers (int): Number of Transformer layers.
        heads (int): Number of attention heads.
        emb_dropout (float): Dropout prob. on embeddings.
    """
    def __init__(self, vocab_size, ctx_len=77, width=256, layers=6, heads=8, emb_dropout=0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, width)
        self.pos = nn.Parameter(torch.randn(1, ctx_len, width) * 0.01)
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(layers)])
        self.ln_final = LayerNorm(width)
        self.drop = nn.Dropout(emb_dropout)

    def forward(self, token_ids):
        """
        Args:
            token_ids (torch.LongTensor): [B, L] token indices.

        Returns:
            torch.Tensor: [B, width] sentence embeddings.
        """
        x = self.tok(token_ids) + self.pos[:, :token_ids.size(1), :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attn_mask=None)
        x = self.ln_final(x)
        return x[:, -1, :]  # last token as sentence representation


class VisionTransformerTiny(nn.Module):
    """
    Vision encoder: small ViT.

    Args:
        image_size (int): Input resolution (assumed square).
        patch_size (int): Size of each patch.
        width (int): Hidden dimension.
        layers (int): Number of Transformer layers.
        heads (int): Number of attention heads.
    """
    def __init__(self, image_size=224, patch_size=16, width=256, layers=6, heads=8):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.randn(1, 1 + num_patches, width) * 0.01)
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(layers)])
        self.ln = LayerNorm(width)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, 3, H, W] image batch.

        Returns:
            torch.Tensor: [B, width] image embeddings (CLS token).
        """
        x = self.conv(x)                         # B, D, H/ps, W/ps
        B, D, H, W = x.shape
        x = x.view(B, D, H*W).transpose(1, 2)    # B, N, D
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = x + self.pos[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return x[:, 0, :]                        # CLS token embedding


class CLIP(nn.Module):
    """
    CLIP model combining text + vision encoders.

    Args:
        vocab_size (int): Vocabulary size for text encoder.
        ctx_len (int): Max sequence length for text input.
        img_size (int): Input image resolution.
        patch_size (int): Patch size for ViT.
        width (int): Hidden dimension (shared by both encoders).
        layers (int): Transformer depth.
        heads (int): Attention heads.
        embed_dim (int): Projection dimension for joint space.
    """
    def __init__(self, vocab_size, ctx_len=77, img_size=224, patch_size=16,
                 width=256, layers=6, heads=8, embed_dim=256):
        super().__init__()
        self.text = TextTransformer(vocab_size, ctx_len, width, layers, heads)
        self.vision = VisionTransformerTiny(img_size, patch_size, width, layers, heads)
        self.text_proj = nn.Linear(width, embed_dim, bias=False)
        self.vision_proj = nn.Linear(width, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

    def encode_text(self, ids):
        """
        Encode token IDs to normalized embeddings.

        Args:
            ids (torch.LongTensor): [B, L] token IDs.

        Returns:
            torch.Tensor: [B, embed_dim] normalized text embeddings.
        """
        x = self.text(ids)
        x = self.text_proj(x)
        return cosine_normalize(x)

    def encode_image(self, images):
        """
        Encode images to normalized embeddings.

        Args:
            images (torch.Tensor): [B, 3, H, W].

        Returns:
            torch.Tensor: [B, embed_dim] normalized image embeddings.
        """
        x = self.vision(images)
        x = self.vision_proj(x)
        return cosine_normalize(x)

    def forward(self, images, ids):
        """
        Forward pass computing similarity logits.

        Args:
            images (torch.Tensor): [B, 3, H, W]
            ids (torch.LongTensor): [B, L] token IDs

        Returns:
            tuple:
                - logits_per_image (torch.Tensor): [B, B] sim(i, t)
                - logits_per_text (torch.Tensor): [B, B] sim(t, i)
        """
        i = self.encode_image(images)
        t = self.encode_text(ids)
        s = self.logit_scale.exp()
        logits_i = s * (i @ t.t())
        return logits_i, logits_i.t()
