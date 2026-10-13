"""BS-RoFormer 추론 전용 구현 (학습 코드 제외).

구조와 파라미터 이름은 lucidrains/BS-RoFormer (MIT) → ZFTurbo/Music-Source-Separation-Training (MIT)
→ nomadkaraoke/python-audio-separator (MIT) 계열 구현과 동일하게 맞춰, 해당 계열에서 학습된
체크포인트(예: BS-RoFormer SW 6-stem by jarredou)를 그대로 불러올 수 있습니다.
외부 의존성(beartype, rotary_embedding_torch 등) 없이 torch 와 einops 만 사용합니다.
"""
from __future__ import annotations

import math
import threading
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import nn

# BS-RoFormer SW 설정 (BS-Roformer-SW.yaml 의 model / audio / inference 항목)
SW_CONFIG = dict(
    dim=256, depth=12, stereo=True, num_stems=6, time_transformer_depth=1, freq_transformer_depth=1,
    linear_transformer_depth=0,
    freqs_per_bands=(2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129),
    dim_head=64, heads=8, stft_n_fft=2048, stft_hop_length=512, stft_win_length=2048, mask_estimator_depth=2,
    mlp_expansion_factor=4,
)
SW_INSTRUMENTS = ("bass", "drums", "other", "vocals", "guitar", "piano")
SW_CHUNK = 588800        # 13.35초 @ 44.1kHz
SW_SAMPLE_RATE = 44100


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class Rotary(nn.Module):
    """rotary_embedding_torch.RotaryEmbedding(dim) 과 동일한 계산 (인터리브 방식)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.freqs = nn.Parameter(1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)),
                                  requires_grad=False)
        self._cache: dict = {}

    def rotate(self, t: torch.Tensor) -> torch.Tensor:
        n = t.shape[-2]
        key = (n, t.device, t.dtype)
        cs = self._cache.get(key)
        if cs is None:
            pos = torch.arange(n, device=t.device, dtype=self.freqs.dtype)
            f = torch.einsum("i,j->ij", pos, self.freqs.to(t.device))
            f = torch.repeat_interleave(f, 2, dim=-1)
            cs = (f.cos().to(t.dtype), f.sin().to(t.dtype))
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = cs
        cos, sin = cs
        rot = cos.shape[-1]
        t_rot, t_pass = t[..., :rot], t[..., rot:]
        x = rearrange(t_rot, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(-1)
        half = rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")
        out = t_rot * cos + half * sin
        return torch.cat((out, t_pass), dim=-1) if t_pass.shape[-1] else out


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, rotary: Rotary):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.rotary_embed = rotary
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q = self.rotary_embed.rotate(q)
        k = self.rotary_embed.rotate(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out * rearrange(self.to_gates(x), "b n h -> b h n 1").sigmoid()
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = int(dim * mult)
        # 인덱스(0,1,4)를 원본 구현(Dropout 포함)과 맞추기 위해 Identity 자리 유지
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, inner), nn.GELU(), nn.Identity(),
                                 nn.Linear(inner, dim), nn.Identity())

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, rotary):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads, dim_head, rotary), FeedForward(dim)])
                                     for _ in range(depth)])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs: Sequence[int]):
        super().__init__()
        self.dim_inputs = tuple(dim_inputs)
        self.to_features = nn.ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dim_inputs])

    def forward(self, x):
        parts = x.split(self.dim_inputs, dim=-1)
        return torch.stack([f(p) for p, f in zip(parts, self.to_features)], dim=-2)


def _mlp(dim_in, dim_out, dim_hidden, depth):
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)
    layers: list = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    def __init__(self, dim, dim_inputs: Sequence[int], depth, mlp_expansion_factor=4):
        super().__init__()
        hidden = dim * mlp_expansion_factor
        self.to_freqs = nn.ModuleList([nn.Sequential(_mlp(dim, d * 2, hidden, depth), nn.GLU(dim=-1))
                                       for d in dim_inputs])

    def forward(self, x):
        return torch.cat([mlp(b) for b, mlp in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)


class BSRoformer(nn.Module):
    def __init__(self, dim, *, depth, stereo=True, num_stems=1, time_transformer_depth=1,
                 freq_transformer_depth=1, linear_transformer_depth=0, freqs_per_bands=(), dim_head=64,
                 heads=8, stft_n_fft=2048, stft_hop_length=512, stft_win_length=2048, mask_estimator_depth=2,
                 mlp_expansion_factor=4):
        super().__init__()
        if linear_transformer_depth:
            raise ValueError("linear transformer 는 지원하지 않습니다")
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        time_rot = Rotary(dim_head)
        freq_rot = Rotary(dim_head)
        self.layers = nn.ModuleList([
            nn.ModuleList([Transformer(dim, time_transformer_depth, heads, dim_head, time_rot),
                           Transformer(dim, freq_transformer_depth, heads, dim_head, freq_rot)])
            for _ in range(depth)])
        self.final_norm = RMSNorm(dim)
        self.n_fft, self.hop, self.win = stft_n_fft, stft_hop_length, stft_win_length
        n_freqs = stft_n_fft // 2 + 1
        if sum(freqs_per_bands) != n_freqs:
            raise ValueError("freqs_per_bands 합계가 STFT 주파수 수와 다릅니다")
        dims = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim, dims)
        self.mask_estimators = nn.ModuleList([MaskEstimator(dim, dims, mask_estimator_depth, mlp_expansion_factor)
                                              for _ in range(num_stems)])

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (b, ch, t) → (b, stems, ch, t)"""
        device = audio.device
        # 복소수 연산(STFT)은 MPS 에서 불안정할 수 있어 CPU 에서 수행
        cplx_dev = torch.device("cpu") if device.type == "mps" else device
        b, s, t = audio.shape
        window = torch.hann_window(self.win, device=cplx_dev)
        spec = torch.stft(audio.reshape(b * s, t).to(cplx_dev), n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.win, window=window, return_complex=True)
        spec = torch.view_as_real(spec)                                  # (b*s, f, frames, 2)
        spec = rearrange(spec, "(b s) f n c -> b (f s) n c", b=b, s=s)
        x = rearrange(spec, "b f n c -> b n (f c)").to(device)
        x = self.band_split(x)
        for time_tf, freq_tf in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_tf(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_tf(x)
            (x,) = unpack(x, ps, "* f d")
        x = self.final_norm(x)
        mask = torch.stack([m(x) for m in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2).to(cplx_dev).float()
        spec_c = torch.view_as_complex(rearrange(spec, "b f t c -> b 1 f t c").contiguous())
        out = spec_c * torch.view_as_complex(mask.contiguous())
        out = rearrange(out, "b n (f s) t -> (b n s) f t", s=self.audio_channels)
        rec = torch.istft(out, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win, window=window,
                          length=t)
        return rearrange(rec, "(b n s) t -> b n s t", b=b, s=self.audio_channels, n=self.num_stems).to(device)


def load_sw(path) -> BSRoformer:
    model = BSRoformer(**SW_CONFIG)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class Cancelled(Exception):
    pass


def _window(size: int, fade: int) -> torch.Tensor:
    w = torch.ones(size)
    ramp = torch.linspace(0, 1, fade)
    w[:fade] = ramp
    w[-fade:] = ramp.flip(0)
    return w


@torch.no_grad()
def demix(model: BSRoformer, mix: np.ndarray, device: str = "cpu", num_overlap: int = 2,
          chunk: int = SW_CHUNK, batch: int = 1,
          progress: Optional[Callable[[float], None]] = None,
          cancel: Optional[threading.Event] = None) -> np.ndarray:
    """mix: (n, 2) float32 → (stems, n, 2). ZFTurbo 방식의 겹침 + 페이드 창 가중 평균."""
    model.to(device)
    x = torch.from_numpy(np.ascontiguousarray(np.asarray(mix, dtype=np.float32).T))  # (2, n)
    n = x.shape[1]
    step = chunk // num_overlap
    border = chunk - step
    fade = chunk // 10
    pad_mode = "reflect" if n > 2 * border else "constant"
    x = F.pad(x[None], (border, border), mode=pad_mode)[0]
    total = x.shape[1]
    out = torch.zeros(model.num_stems, 2, total)
    counter = torch.zeros(total)
    win = _window(chunk, fade)
    starts = list(range(0, total, step))
    done = 0
    for i in range(0, len(starts), batch):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        group = starts[i:i + batch]
        parts = []
        lengths = []
        for st in group:
            seg = x[:, st:st + chunk]
            lengths.append(seg.shape[1])
            if seg.shape[1] < chunk:
                seg = F.pad(seg, (0, chunk - seg.shape[1]))
            parts.append(seg)
        y = model(torch.stack(parts).to(device)).float().cpu()                       # (b, stems, 2, chunk)
        for j, st in enumerate(group):
            ln = lengths[j]
            w = win.clone()
            if st == 0:
                w[:fade] = 1
            if st + chunk >= total:
                w[-fade:] = 1
            out[..., st:st + ln] += y[j, ..., :ln] * w[:ln]
            counter[st:st + ln] += w[:ln]
        done += len(group)
        if progress:
            progress(done / len(starts))
    out = out / counter.clamp_min(1e-8)
    out = out[..., border:border + n]
    return out.permute(0, 2, 1).contiguous().numpy()


def estimate_steps(n_samples: int, num_overlap: int = 2, chunk: int = SW_CHUNK) -> int:
    step = chunk // num_overlap
    total = n_samples + 2 * (chunk - step)
    return math.ceil(total / step)
