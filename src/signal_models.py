from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt


def generate_continuous_signal(
    frequency: float,
    amplitude: float,
    phase: float,
    duration: float,
    display_rate: int,
    signal_type: str = "sine",
    second_frequency: float | None = None,
    second_amplitude: float = 0.0,
    noise_level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate continuous signal at high resolution (display_rate).
    
    Args:
        frequency: Primary frequency in Hz
        amplitude: Signal amplitude
        phase: Phase in radians
        duration: Duration in seconds
        display_rate: Sample rate for "continuous" representation (Hz)
        signal_type: "sine", "chirp", or "multi_tone"
        second_frequency: Secondary frequency for multi_tone
        second_amplitude: Secondary amplitude
        noise_level: Gaussian noise standard deviation
    
    Returns:
        (time_array, signal_array)
    """
    t = np.arange(0.0, duration, 1.0 / display_rate)
    
    if signal_type == "chirp":
        # Chirp from frequency to 2*frequency
        signal = amplitude * np.sin(2 * np.pi * (frequency * t + (frequency / (2 * duration)) * t**2) + phase)
    elif signal_type == "multi_tone":
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        if second_frequency and second_amplitude > 0:
            signal += second_amplitude * np.sin(2 * np.pi * second_frequency * t + 0.25 * phase)
    else:  # "sine"
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    # Add noise if specified
    if noise_level > 0:
        signal += np.random.normal(0, noise_level, len(signal))
    
    return t, signal


def sample_signal(
    continuous_time: np.ndarray,
    continuous_signal: np.ndarray,
    sampling_rate: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    ts = 1.0 / sampling_rate
    sample_times = np.arange(0.0, duration, ts)
    sample_values = np.interp(sample_times, continuous_time, continuous_signal)
    return sample_times, sample_values


def reconstruct_signal(
    sample_times: np.ndarray,
    sample_values: np.ndarray,
    evaluation_time: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    if len(sample_times) < 2:
        return np.full_like(evaluation_time, sample_values[0] if len(sample_values) else 0.0)

    if method == "zero_order_hold":
        idx = np.searchsorted(sample_times, evaluation_time, side="right") - 1
        idx = np.clip(idx, 0, len(sample_values) - 1)
        return sample_values[idx]

    return np.interp(evaluation_time, sample_times, sample_values)


def alias_frequency(signal_frequency: float, sampling_rate: float) -> float:
    if sampling_rate <= 0:
        return float("nan")
    folded = abs(((signal_frequency + sampling_rate / 2) % sampling_rate) - sampling_rate / 2)
    return float(folded)


def apply_anti_aliasing_filter(signal: np.ndarray, sample_rate: float, cutoff_rate: float) -> np.ndarray:
    """Apply Butterworth low-pass filter before downsampling.
    
    Args:
        signal: Input signal
        sample_rate: Sample rate of input signal (Hz)
        cutoff_rate: Target sample rate - filter cutoff set to cutoff_rate/2 (Hz)
    
    Returns:
        Filtered signal
    """
    if cutoff_rate >= sample_rate * 0.99:
        return signal.copy()
    
    nyquist = sample_rate / 2
    normalized_cutoff = (cutoff_rate / 2) / nyquist
    normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)
    
    sos = butter(4, normalized_cutoff, output='sos')
    return sosfilt(sos, signal)


def naive_decimate_synthetic(signal: np.ndarray, decimation_factor: int) -> np.ndarray:
    """Naive downsampling: just take every Nth sample (no filtering).
    
    This is what causes aliasing!
    """
    if decimation_factor <= 1:
        return signal.copy()
    return signal[::decimation_factor]


def filtered_decimate_synthetic(
    signal: np.ndarray,
    original_rate: float,
    target_rate: float,
) -> np.ndarray:
    """Downsampling: anti-alias filter + decimation.
    
    This is the correct way to avoid aliasing.
    """
    decimation_factor = int(round(original_rate / target_rate))
    if decimation_factor <= 1:
        return signal.copy()
    
    filtered = apply_anti_aliasing_filter(signal, original_rate, target_rate)
    return filtered[::decimation_factor]
