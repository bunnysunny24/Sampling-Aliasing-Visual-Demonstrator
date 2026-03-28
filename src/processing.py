from __future__ import annotations
import numpy as np
from src.signal_models import (
    generate_continuous_signal,
    sample_signal,
    reconstruct_signal,
    alias_frequency,
    apply_anti_aliasing_filter,
    naive_decimate_synthetic,
    filtered_decimate_synthetic,
)
from src.audio_utils import naive_downsample, proper_resample


def compute_fft(signal: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(len(signal)) if len(signal) > 8 else np.ones(len(signal))
    spectrum = np.fft.rfft(signal * window)
    freq = np.fft.rfftfreq(len(signal), d=1.0 / rate)
    magnitude = np.abs(spectrum)
    return freq, magnitude


def run_signal_simulation(settings: dict) -> dict:
    """Full technical pipeline for synthetic signal processing.
    
    Layers:
    1. Signal Generation: Continuous signal at high resolution
    2. Sampling: Sample at target rate
    3. Processing: Naive vs. anti-aliased downsampling modes
    4. Computation: FFT, alias frequency, metrics
    """
    # === SIGNAL GENERATION LAYER ===
    t, x = generate_continuous_signal(
        frequency=settings["signal_frequency"],
        amplitude=settings["amplitude"],
        phase=settings["phase"],
        duration=settings["duration"],
        display_rate=settings["display_rate"],
        signal_type=settings.get("signal_type", "sine"),
        second_frequency=settings.get("second_frequency") if settings.get("add_second_tone") else None,
        second_amplitude=settings.get("second_amplitude", 0.0),
        noise_level=settings.get("noise_level", 0.0),
    )
    
    # === SAMPLING LAYER ===
    sample_t, sample_x = sample_signal(t, x, settings["sampling_rate"], settings["duration"])
    reconstructed = reconstruct_signal(sample_t, sample_x, t, method=settings.get("reconstruction_method", "linear"))

    # === PROCESSING LAYER: COMPARATIVE MODES ===
    # Mode 1: Naive downsampling (causes aliasing)
    decimation_factor = max(int(round(settings["display_rate"] / settings["sampling_rate"])), 1)
    naive_downsampled = naive_decimate_synthetic(x, decimation_factor)
    
    # Mode 2: Filtered downsampling (anti-aliased)
    filtered_downsampled = filtered_decimate_synthetic(x, float(settings["display_rate"]), float(settings["sampling_rate"]))

    # === COMPUTATION LAYER ===
    signal_max = settings["signal_frequency"]
    if settings.get("add_second_tone") and settings.get("second_frequency"):
        signal_max = max(signal_max, settings["second_frequency"])
    
    nyquist_required = 2 * signal_max
    sampling_rate = settings["sampling_rate"]
    
    # Determine sampling status
    if sampling_rate > nyquist_required * 1.05:
        status = "Oversampling / Safe"
    elif np.isclose(sampling_rate, nyquist_required, atol=0.5):
        status = "Nyquist Limit"
    else:
        status = "Undersampling / Aliasing"

    # === VISUALIZATION LAYER: Frequency domain ===
    original_freq, original_mag = compute_fft(x, settings["display_rate"])
    recon_freq, recon_mag = compute_fft(reconstructed, settings["display_rate"])
    naive_freq, naive_mag = compute_fft(naive_downsampled, settings["display_rate"])
    filtered_freq, filtered_mag = compute_fft(filtered_downsampled, settings["display_rate"])

    # === EXPLANATION LAYER ===
    alias_freq = alias_frequency(settings["signal_frequency"], sampling_rate)
    explanation = _generate_explanation(
        sampling_rate=sampling_rate,
        nyquist_required=nyquist_required,
        signal_freqs=[settings["signal_frequency"]] + 
                     ([settings["second_frequency"]] if settings.get("add_second_tone") and settings.get("second_frequency") else []),
        alias_freq=alias_freq,
        status=status,
    )

    return {
        "mode": "signal",
        "continuous_time": t,
        "continuous_signal": x,
        "sample_times": sample_t,
        "sample_values": sample_x,
        "reconstructed_signal": reconstructed,
        "naive_downsampled": naive_downsampled,
        "filtered_downsampled": filtered_downsampled,
        "signal_frequency": settings["signal_frequency"],
        "second_frequency": settings.get("second_frequency"),
        "apparent_alias_frequency": alias_freq,
        "sampling_rate": sampling_rate,
        "nyquist_required": nyquist_required,
        "status": status,
        "original_freq": original_freq,
        "original_mag": original_mag,
        "recon_freq": recon_freq,
        "recon_mag": recon_mag,
        "naive_freq": naive_freq,
        "naive_mag": naive_mag,
        "filtered_freq": filtered_freq,
        "filtered_mag": filtered_mag,
        "display_rate": settings["display_rate"],
        "settings": settings,
        "explanation": explanation,
    }


def _generate_explanation(
    sampling_rate: float,
    nyquist_required: float,
    signal_freqs: list[float],
    alias_freq: float,
    status: str,
) -> str:
    """Generate dynamic explanation of what's happening."""
    freq_str = ", ".join(f"{f:.1f} Hz" for f in signal_freqs)
    
    lines = [
        f"**Signal frequencies:** {freq_str}",
        f"**Nyquist requirement:** {nyquist_required:.2f} Hz (2 × max frequency)",
        f"**Sampling rate:** {sampling_rate:.2f} Hz",
        f"**Nyquist limit:** {sampling_rate / 2:.2f} Hz",
        "",
    ]
    
    if status == "Oversampling / Safe":
        lines.append(
            "✅ **Status: OVERSAMPLING** — Sampling rate exceeds Nyquist requirement. "
            "All signal information is preserved. No aliasing."
        )
    elif status == "Nyquist Limit":
        lines.append(
            "⚠️ **Status: NYQUIST LIMIT** — Sampling rate exactly meets Nyquist requirement. "
            "Theoretically recoverable, but practically risky."
        )
    else:
        lines.append(
            f"❌ **Status: UNDERSAMPLING/ALIASING** — Sampling rate is below Nyquist! "
            f"The signal at {signal_freqs[0]:.1f} Hz will alias to {alias_freq:.2f} Hz."
        )
        lines.append(
            f"To prevent aliasing, apply an anti-aliasing filter (low-pass at {sampling_rate / 2:.1f} Hz) before sampling."
        )
    
    return "\n".join(lines)


def estimate_dominant_frequency(signal: np.ndarray, rate: int) -> float:
    freq, mag = compute_fft(signal, rate)
    if len(mag) < 2:
        return 0.0
    idx = np.argmax(mag[1:]) + 1
    return float(freq[idx])


def run_audio_simulation(audio_data: dict, settings: dict) -> dict:
    """Audio processing pipeline.
    
    Computes BOTH naive and filtered downsampling to demonstrate
    how anti-aliasing filter prevents high-frequency distortion.
    """
    signal = audio_data["signal"]
    original_rate = audio_data["rate"]
    target_rate = settings["target_audio_rate"]
    
    # === PROCESSING LAYER: COMPARATIVE MODES ===
    # Both naive and filtered are always computed for comparison
    naive_processed = naive_downsample(signal, original_rate, target_rate)
    filtered_processed = proper_resample(signal, original_rate, target_rate)
    
    # Use selected mode for main display
    if settings["audio_mode"] == "Filtered Downsampling":
        processed = filtered_processed
        mode_label = "Filtered"
    else:
        processed = naive_processed
        mode_label = "Naive"

    # === COMPUTATION LAYER ===
    # Analyze dominant frequency in first 2 seconds
    sample_idx = min(len(signal), original_rate * 2)
    dominant = estimate_dominant_frequency(signal[:sample_idx], original_rate)
    nyquist_limit = target_rate / 2
    
    # Status based on frequency content vs Nyquist limit
    if dominant > nyquist_limit:
        if settings["audio_mode"] == "Naive Downsampling":
            status = f"❌ Aliasing Risk: {dominant:.1f} Hz > Nyquist {nyquist_limit:.1f} Hz"
        else:
            status = f"✅ Safe: Anti-aliased before resampling"
    else:
        status = "✅ Safe: Signal within Nyquist limit"

    # === VISUALIZATION LAYER ===
    fft_window = min(len(signal), original_rate * 4)
    original_freq, original_mag = compute_fft(signal[:fft_window], original_rate)
    processed_freq, processed_mag = compute_fft(processed[:fft_window], original_rate)
    naive_freq, naive_mag = compute_fft(naive_processed[:fft_window], original_rate)
    filtered_freq, filtered_mag = compute_fft(filtered_processed[:fft_window], original_rate)

    # === EXPLANATION LAYER ===
    explanation = _generate_audio_explanation(
        dominant_freq=dominant,
        original_rate=original_rate,
        target_rate=target_rate,
        aliasing_risk=dominant > nyquist_limit,
    )

    return {
        "mode": "audio",
        "original_signal": signal,
        "processed_signal": processed,
        "naive_processed": naive_processed,
        "filtered_processed": filtered_processed,
        "original_rate": original_rate,
        "target_rate": target_rate,
        "nyquist_required": dominant * 2,
        "sampling_rate": target_rate,
        "status": status,
        "dominant_frequency": dominant,
        "original_freq": original_freq,
        "original_mag": original_mag,
        "processed_freq": processed_freq,
        "processed_mag": processed_mag,
        "naive_freq": naive_freq,
        "naive_mag": naive_mag,
        "filtered_freq": filtered_freq,
        "filtered_mag": filtered_mag,
        "apparent_alias_frequency": alias_frequency(dominant, target_rate),
        "settings": settings,
        "explanation": explanation,
        "mode_label": mode_label,
    }


def _generate_audio_explanation(
    dominant_freq: float,
    original_rate: int,
    target_rate: int,
    aliasing_risk: bool,
) -> str:
    """Explain what's happening with audio processing."""
    lines = [
        f"**Original sample rate:** {original_rate} Hz",
        f"**Target sample rate:** {target_rate} Hz",
        f"**Nyquist limit:** {target_rate / 2:.1f} Hz",
        f"**Dominant frequency detected:** {dominant_freq:.1f} Hz",
        "",
    ]
    
    if aliasing_risk:
        lines.append(
            f"⚠️ **ALIASING RISK:** The audio contains energy at {dominant_freq:.1f} Hz, "
            f"which exceeds the Nyquist limit of {target_rate / 2:.1f} Hz.\n\n"
            f"**Naive downsampling** will alias this to {alias_frequency(dominant_freq, target_rate):.1f} Hz.\n\n"
            f"**Anti-aliased downsampling** applies a low-pass filter at {target_rate / 2:.1f} Hz "
            f"before decimation, removing frequencies above the Nyquist limit."
        )
    else:
        lines.append(
            f"✅ **SAFE:** All significant frequencies are below the Nyquist limit. "
            f"Both naive and filtered downsampling will produce acceptable results."
        )
    
    return "\n".join(lines)
