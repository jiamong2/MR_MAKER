"""멀티트랙 실시간 재생.

Mixer  : 순수 numpy 믹싱 로직 (뮤트/솔로/볼륨, 클릭 방지 게인 램프) — 오디오 장치 없이 테스트 가능
Player : sounddevice(PortAudio) 출력 스트림. Windows/macOS 휠에 PortAudio 가 포함되어 있음
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .audio_io import SAMPLE_RATE
from .dsp import audible_flags

log = logging.getLogger(__name__)


@dataclass
class Track:
    key: str
    data: np.ndarray          # (n, 2) float32, memmap 가능
    gain: float = 1.0         # 0.0 ~ 1.5
    mute: bool = False
    solo: bool = False
    reference: bool = False   # 원본(내보내기/MR 제외)


class Mixer:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._lock = threading.RLock()
        self.tracks: list[Track] = []
        self.length = 0
        self.position = 0
        self.master = 1.0
        self._prev_gains: Optional[np.ndarray] = None

    # -- 구성 --------------------------------------------------------------
    def set_tracks(self, tracks: list[Track]) -> None:
        with self._lock:
            self.tracks = list(tracks)
            self.length = min((len(t.data) for t in tracks), default=0)
            self.position = min(self.position, self.length)
            self._prev_gains = None

    def clear(self) -> None:
        self.set_tracks([])
        self.position = 0

    def track(self, key: str) -> Optional[Track]:
        for t in self.tracks:
            if t.key == key:
                return t
        return None

    def seek(self, sample: int) -> None:
        with self._lock:
            self.position = int(max(0, min(self.length, sample)))

    def seek_seconds(self, sec: float) -> None:
        self.seek(int(sec * self.sample_rate))

    @property
    def seconds(self) -> float:
        return self.position / self.sample_rate

    @property
    def duration(self) -> float:
        return self.length / self.sample_rate

    def target_gains(self) -> np.ndarray:
        flags = audible_flags([t.mute for t in self.tracks], [t.solo for t in self.tracks])
        return np.array([t.gain * self.master * (1.0 if f else 0.0)
                         for t, f in zip(self.tracks, flags)], dtype=np.float32)

    # -- 렌더 --------------------------------------------------------------
    def render(self, frames: int) -> tuple[np.ndarray, bool]:
        """frames 만큼 믹싱한 (frames, 2) 블록과 끝 도달 여부."""
        out = np.zeros((frames, 2), dtype=np.float32)
        with self._lock:
            if not self.tracks or self.position >= self.length:
                return out, True
            start = self.position
            end = min(self.length, start + frames)
            n = end - start
            target = self.target_gains()
            prev = self._prev_gains if self._prev_gains is not None and \
                len(self._prev_gains) == len(target) else target
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            for i, t in enumerate(self.tracks):
                g0, g1 = float(prev[i]), float(target[i])
                if g0 == 0.0 and g1 == 0.0:
                    continue
                block = np.asarray(t.data[start:end], dtype=np.float32)
                if g0 == g1:
                    out[:n] += block * g1
                else:
                    out[:n] += block * (g0 + (g1 - g0) * ramp)
            self._prev_gains = target
            self.position = end
            ended = end >= self.length
        np.clip(out, -1.0, 1.0, out=out)
        return out, ended


class Player:
    """sounddevice 기반 재생기. 장치 오류는 last_error 에 기록하고 예외를 던지지 않음."""

    def __init__(self, mixer: Mixer):
        self.mixer = mixer
        self._stream = None
        self.playing = False
        self.last_error: Optional[str] = None
        self._latency = 0.0

    def available(self) -> bool:
        try:
            import sounddevice  # noqa: F401
            return True
        except Exception as exc:
            self.last_error = f"오디오 출력 모듈을 불러올 수 없습니다: {exc}"
            return False

    def _callback(self, outdata, frames, time_info, status):  # PortAudio 스레드
        block, ended = self.mixer.render(frames)
        outdata[:] = block
        if ended:
            import sounddevice as sd

            self.playing = False
            raise sd.CallbackStop()

    def play(self) -> bool:
        if self.playing:
            return True
        if self.mixer.length == 0:
            return False
        if self.mixer.position >= self.mixer.length:
            self.mixer.seek(0)
        try:
            import sounddevice as sd

            self._close()
            self._stream = sd.OutputStream(
                samplerate=self.mixer.sample_rate, channels=2, dtype="float32",
                callback=self._callback, finished_callback=self._finished,
            )
            self._latency = float(self._stream.latency or 0.0)
            self.playing = True
            self._stream.start()
            self.last_error = None
            return True
        except Exception as exc:
            log.exception("오디오 재생 실패")
            self.playing = False
            self.last_error = f"오디오 장치를 열 수 없습니다. 출력 장치(스피커/헤드폰)를 확인해 주세요.\n({exc})"
            self._close()
            return False

    def _finished(self) -> None:
        self.playing = False

    def pause(self) -> None:
        self.playing = False
        self._close()

    def stop(self) -> None:
        self.pause()
        self.mixer.seek(0)

    def toggle(self) -> bool:
        if self.playing:
            self.pause()
            return True
        return self.play()

    def display_seconds(self) -> float:
        """출력 지연을 고려한 화면 표시 위치."""
        s = self.mixer.seconds
        if self.playing:
            s -= self._latency
        return max(0.0, s)

    def _close(self) -> None:
        st, self._stream = self._stream, None
        if st is not None:
            try:
                st.abort(ignore_errors=True)
                st.close(ignore_errors=True)
            except Exception:
                pass

    def shutdown(self) -> None:
        self.pause()
