from __future__ import annotations
from io import BytesIO
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_uploaded_audio(uploaded_file) -> dict:
    data, rate = sf.read(uploaded_file)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data)) or 1.0
    data = data / peak
    return {"signal": data, "rate": int(rate)}


def naive_downsample(signal: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    step = max(int(round(original_rate / target_rate)), 1)
    reduced = signal[::step]
    stretched_idx = np.linspace(0, len(reduced) - 1, len(signal))
    return np.interp(stretched_idx, np.arange(len(reduced)), reduced)


def proper_resample(signal: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    gcd = np.gcd(original_rate, target_rate)
    up = target_rate // gcd
    down = original_rate // gcd
    reduced = resample_poly(signal, up, down)
    stretched_idx = np.linspace(0, len(reduced) - 1, len(signal))
    return np.interp(stretched_idx, np.arange(len(reduced)), reduced)


def audio_bytes_from_signal(signal: np.ndarray, rate: int) -> bytes:
    clipped = np.clip(signal, -1.0, 1.0).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, clipped, rate, format="WAV")
    buffer.seek(0)
    return buffer.read()
