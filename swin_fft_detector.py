"""
Reference implementation of the paper:
"생성형 AI 합성 이미지 판별을 위한 Swin Transformer 기반 주파수 정보 통합 전략에 관한 연구"
(A Study on Frequency Integration Strategies with Swin Transformer for AI-generated Image Detection)
Journal of the Korea Institute of Electronic Communication Sciences (JKIECS), Vol. 21, No. 3, pp. 1101-1108, 2026.
DOI: http://dx.doi.org/10.13067/JKIECS.2026.21.3.1101
"""

import torch
import torch.nn as nn
import timm


def extract_log_fft_spectrum(rgb_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Section 3.2: 2D FFT Log-Scaled Amplitude Spectrum Extraction.

    Pipeline:
      1. Convert 3-channel RGB image (B, 3, H, W) to 1-channel Grayscale (B, 1, H, W).
      2. Apply 2D Fast Fourier Transform (FFT) and shift zero-frequency component to center.
      3. Compute Amplitude Spectrum (excluding Phase component due to low discriminative validity).
      4. Apply Log-Scale transformation to balance low- and high-frequency components.
      5. Normalize per image and replicate to 3 channels (B, 3, H, W) for Swin-Tiny input compatibility.
    """
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb_tensor.device, dtype=rgb_tensor.dtype)
    gray = (rgb_tensor * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

    fft_complex = torch.fft.fft2(gray, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

    amplitude = torch.abs(fft_shifted)
    log_spectrum = torch.log1p(amplitude + eps)

    b = log_spectrum.shape[0]
    flat = log_spectrum.view(b, -1)
    min_val = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    max_val = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    normalized = (log_spectrum - min_val) / (max_val - min_val + eps)

    return normalized.repeat(1, 3, 1, 1)


class AttentionFusionModule(nn.Module):
    """
    Section 3.3: Attention-based Feature Fusion Module for Spatial (RGB) and Frequency (2D FFT) features.
    Supports:
      - 'concat': Simple feature concatenation
      - 'mlp_attention': Dynamic modality weighting via MLP gating
      - 'self_attention': Multi-head Self-Attention over stacked modality tokens
      - 'cross_attention_rgb_query': Cross-Attention using RGB features as Query (Q) and FFT as Key/Value (K, V)
    """

    def __init__(self, embed_dim: int = 768, fusion_type: str = "mlp_attention", num_heads: int = 8):
        super().__init__()
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim

        if fusion_type == "concat":
            self.out_dim = embed_dim * 2
        elif fusion_type == "mlp_attention":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 2),
                nn.Softmax(dim=-1),
            )
            self.out_dim = embed_dim * 2
        elif fusion_type == "self_attention":
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            self.out_dim = embed_dim * 2
        elif fusion_type == "cross_attention_rgb_query":
            self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            self.out_dim = embed_dim
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, feat_rgb: torch.Tensor, feat_fft: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat([feat_rgb, feat_fft], dim=-1)

        if self.fusion_type == "mlp_attention":
            combined = torch.cat([feat_rgb, feat_fft], dim=-1)
            weights = self.gate(combined)  # (B, 2) -> w_orig, w_fft
            w_rgb = weights[:, 0:1]
            w_fft = weights[:, 1:2]
            return torch.cat([feat_rgb * w_rgb, feat_fft * w_fft], dim=-1)

        if self.fusion_type == "self_attention":
            tokens = torch.stack([feat_rgb, feat_fft], dim=1)  # (B, 2, D)
            attn_out, _ = self.attn(tokens, tokens, tokens)
            tokens = self.norm(tokens + attn_out)
            return tokens.flatten(start_dim=1)

        if self.fusion_type == "cross_attention_rgb_query":
            q = feat_rgb.unsqueeze(1)  # (B, 1, D)
            kv = feat_fft.unsqueeze(1)  # (B, 1, D)
            attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
            return self.norm(q + attn_out).squeeze(1)


class SwinFFTSyntheticDetector(nn.Module):
    """
    Dual-Stream Swin Transformer Tiny with 2D FFT Frequency Integration.
    """

    def __init__(
        self,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        fusion_type: str = "mlp_attention",
        num_classes: int = 1,
    ):
        super().__init__()
        self.rgb_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.fft_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        embed_dim = self.rgb_backbone.num_features  # 768 for Swin-Tiny
        self.fusion = AttentionFusionModule(embed_dim=embed_dim, fusion_type=fusion_type)
        self.classifier = nn.Linear(self.fusion.out_dim, num_classes)

    def forward(self, rgb_img: torch.Tensor, fft_img: torch.Tensor = None) -> torch.Tensor:
        if fft_img is None:
            fft_img = extract_log_fft_spectrum(rgb_img)

        feat_rgb = self.rgb_backbone(rgb_img)
        feat_fft = self.fft_backbone(fft_img)
        fused = self.fusion(feat_rgb, feat_fft)
        logits = self.classifier(fused)
        return logits
