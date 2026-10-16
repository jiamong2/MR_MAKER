"""RMVPE 보컬 음정(f0) 추적 — 추론 전용 구현.

RMVPE: "A Robust Model for Vocal Pitch Estimation in Polyphonic Music" (Wei et al., Interspeech 2023).
구조와 파라미터 이름은 RVC-Project/Retrieval-based-Voice-Conversion-WebUI 의 infer/lib/rmvpe.py (MIT)와
같게 맞춰 공개 가중치(rmvpe.pt)를 그대로 불러온다. librosa · scipy 없이 torch 와 numpy 만 사용.

입력은 16kHz 모노, 10ms(160샘플) 간격으로 f0(Hz, 0 = 무성)와 신뢰도(0~1)를 낸다.
긴 곡은 32초 단위로 나누고 앞뒤 1.6초를 겹쳐 계산해 메모리를 일정하게 유지한다.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SAMPLE_RATE = 16000
HOP = 160                    # 10ms
FPS = SAMPLE_RATE // HOP     # 100 프레임/초
N_MELS = 128
N_FFT = 1024
FMIN, FMAX = 30.0, 8000.0
THRESHOLD = 0.03             # RVC 기본값 (최대 신뢰도가 이 이하이면 무성)
SEGMENT = 3200               # 한 번에 계산하는 프레임 수 (32초, 32의 배수)
CONTEXT = 160                # 앞뒤로 함께 넣는 프레임 수 (1.6초, 32의 배수)
CENTS = 20 * np.arange(360) + 1997.3794084376191


class Cancelled(Exception):
    pass


# ------------------------------------------------------------------ 모델 ---

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else None

    def forward(self, x):
        return self.conv(x) + (x if self.shortcut is None else self.shortcut(x))


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)]
                                  + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])
        self.pool = nn.AvgPool2d(kernel_size=kernel_size) if kernel_size is not None else None

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x if self.pool is None else (x, self.pool(x))


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            in_channels, out_channels, in_size = out_channels, out_channels * 2, in_size // 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            skips.append(t)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)]
                                    + [ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
                                       for _ in range(n_inters - 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride, (1, 1), output_padding=out_padding,
                               bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)]
                                   + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])

    def forward(self, x, skip):
        x = torch.cat((self.conv1(x), skip), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            self.layers.append(ResDecoderBlock(in_channels, in_channels // 2, stride, n_blocks, momentum))
            in_channels //= 2

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers,
                                         n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class E2E(nn.Module):
    """RMVPE 본체: 로그 멜 스펙트로그램 (b, 128, t) → 프레임별 음정 확률 (b, t, 360)."""

    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5, inter_layers=4, in_channels=1,
                 en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, 360), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        x = self.cnn(self.unet(mel.transpose(-1, -2).unsqueeze(1))).transpose(1, 2).flatten(-2)
        return self.fc(x)


def load(path) -> E2E:
    model = E2E(4, 1, (2, 2))
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ------------------------------------------------------- 멜 스펙트로그램 ---

def mel_filters(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS, fmin: float = FMIN,
                fmax: float = FMAX) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=True) 와 같은 값 (Slaney 면적 정규화)."""
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    to_mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)  # noqa: E731
    mel_f = 700.0 * (10.0 ** (np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2) / 2595.0) - 1.0)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    weights = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


_MEL_BASIS: dict = {}


def log_mel(audio16: torch.Tensor) -> torch.Tensor:
    """(b, n) 16kHz → (b, 128, frames) 로그 멜. 프레임 i 의 중심 = i × 10ms."""
    dev = audio16.device
    basis = _MEL_BASIS.get(dev)
    if basis is None:
        basis = _MEL_BASIS[dev] = torch.from_numpy(mel_filters()).to(dev)
    spec = torch.stft(audio16, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=torch.hann_window(N_FFT, device=dev), center=True, return_complex=True)
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2))
    return torch.log(torch.clamp(torch.matmul(basis, mag), min=1e-5))


# ------------------------------------------------------------------ 추론 ---

def decode(salience: np.ndarray, threshold: float = THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """(t, 360) 확률 → f0(Hz, 무성 = 0), 최대 확률. 최댓값 주변 9칸 가중 평균 (RVC 와 같은 방식)."""
    sal = np.asarray(salience, dtype=np.float32)
    center = np.argmax(sal, axis=1)
    padded = np.pad(sal, ((0, 0), (4, 4)))
    cents_map = np.pad(CENTS, (4, 4))
    idx = center[:, None] + np.arange(9)[None, :]
    w = np.take_along_axis(padded, idx, axis=1)
    cents = np.sum(w * cents_map[idx], axis=1) / np.sum(w, axis=1)
    conf = sal.max(axis=1)
    f0 = 10.0 * 2.0 ** (cents / 1200.0)
    f0[conf <= threshold] = 0.0
    return f0.astype(np.float32), conf.astype(np.float32)


@torch.no_grad()
def salience_16k(model: E2E, audio16: np.ndarray, device: str = "cpu",
                 progress: Optional[Callable[[float], None]] = None,
                 cancel: Optional[threading.Event] = None) -> np.ndarray:
    """16kHz 모노 → (프레임, 360) 확률. 짧으면 한 번에(RVC 와 동일), 길면 겹쳐 나눠 계산."""
    model.to(device)
    a = np.ascontiguousarray(audio16, dtype=np.float32)
    n = 1 + len(a) // HOP
    if len(a) < N_FFT:                                           # 아주 짧은 입력 (반사 패딩이 안 되는 길이)
        a = np.pad(a, (0, N_FFT - len(a)))
    mel = log_mel(torch.from_numpy(a)[None].to(device))[..., :n]   # (1, 128, t)

    def run(m):
        k = m.shape[-1]
        pad = 32 * ((k - 1) // 32 + 1) - k
        if pad:
            m = F.pad(m, (0, pad), mode="constant")
        return model(m)[:, :k].float().cpu()

    if n <= SEGMENT + 2 * CONTEXT:
        out = run(mel)[0].numpy()
        if progress:
            progress(1.0)
        return out
    parts = []
    starts = list(range(0, n, SEGMENT))
    for i, s in enumerate(starts):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        a, b = max(0, s - CONTEXT), min(n, s + SEGMENT + CONTEXT)
        y = run(mel[..., a:b])[0]
        parts.append(y[s - a:s - a + min(SEGMENT, n - s)])
        if progress:
            progress((i + 1) / len(starts))
    return torch.cat(parts).numpy()


def track(model: E2E, audio: np.ndarray, rate: int, device: str = "cpu",
          progress: Optional[Callable[[float], None]] = None,
          cancel: Optional[threading.Event] = None) -> tuple[np.ndarray, np.ndarray]:
    """(n,) 또는 (n, ch) 오디오 → (f0 Hz, 신뢰도), 10ms 간격. 프레임 i 의 중심 = i / 100 초."""
    from .audio_io import resample

    x = np.asarray(audio, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    a16 = resample(mono[:, None], rate, SAMPLE_RATE)[:, 0] if rate != SAMPLE_RATE else mono
    return decode(salience_16k(model, a16, device, progress, cancel))
