# 🌊 Sampling & Aliasing Visual Demonstrator
### PS18 — IBM Hackathon | Interactive DSP Education Tool

> *"See exactly why digital systems misrepresent signals — and how to fix it."*

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-green)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.13+-orange)](https://scipy.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.24+-purple)](https://plotly.com)

---

## 📌 Table of Contents
1. [Problem Statement](#problem-statement)
2. [What This Tool Does](#what-this-tool-does)
3. [Theory](#theory)
4. [System Architecture](#system-architecture)
5. [Features](#features)
6. [Demo Cases](#demo-cases)
7. [Installation & Running](#installation--running)
8. [Project Structure](#project-structure)
9. [Real-World Applications](#real-world-applications)
10. [Key Equations](#key-equations)
11. [Future Work](#future-work)

---

## Problem Statement

**PS18:** *Build a visual simulator to demonstrate under-sampling, Nyquist sampling, and aliasing effects.*

Digital computers cannot directly store continuous signals. They must **sample** the signal at discrete points in time. If sampling is done too slowly — below the **Nyquist rate** — the stored data no longer accurately represents the original signal. This phenomenon is called **aliasing**, and it causes real failures in audio, video, medical, and communications systems.

**This simulator makes aliasing visible, interactive, and undeniable.**

---

## What This Tool Does

| View | What You See |
|------|-------------|
| **Time Domain** | Original continuous wave + sampled dots (with stem lines) |
| **Reconstruction** | Original vs reconstructed signal side-by-side |
| **Frequency Domain (FFT)** | Spectral peaks with Nyquist limit marked |
| **⚖️ Comparative Analysis** | Naive (aliased) vs Anti-Aliased downsampling — side-by-side FFT |
| **Audio Preview** | Listen to original vs reconstructed signal |
| **Audio Upload Mode** | Upload WAV file, compare naive vs filtered downsampling |

---

## Theory

### 1. Continuous Signal

A real-world signal is a smooth function of time:

```
x(t) = A · sin(2πft + φ)
```

| Symbol | Meaning |
|--------|---------|
| A | Amplitude |
| f | Frequency (Hz) |
| t | Time (seconds) |
| φ | Phase (radians) |

This signal exists at every instant in time. A computer must approximate it using a **finite set of measurements**.

---

### 2. Sampling

Instead of every point, we measure only at fixed intervals:

```
t_n = n × T_s
x[n] = x(n · T_s)
```

| Symbol | Meaning |
|--------|---------|
| T_s | Sampling period |
| f_s = 1/T_s | Sampling frequency (Hz) |
| n | Sample index |

---

### 3. Nyquist-Shannon Sampling Theorem

> **To perfectly reconstruct a bandlimited signal with maximum frequency f_max, sample at f_s ≥ 2·f_max.**

```
f_s ≥ 2 · f_max     ← Nyquist Criterion
```

- **Nyquist Rate:** `f_Nyquist = 2·f_max` — the minimum safe sampling rate
- **Nyquist Frequency:** `f_N = f_s/2` — the highest representable frequency at rate f_s

**Real examples:**
- CD Audio: 20 kHz max → 44,100 Hz sampling (2.2× Nyquist)
- Phone calls: 3.4 kHz max → 8,000 Hz sampling (2.35× Nyquist)
- ECG medical: minimum 250 Hz to capture QRS complex

---

### 4. Aliasing

When `f_s < 2·f`, sampled values from two different frequencies become **indistinguishable**.

**Why this happens (mathematical proof):**

Consider `x(t) = sin(2πf·t)` sampled at rate `f_s`:
```
x[n] = sin(2πf·nT_s)
```

Now consider `y(t) = sin(2π(f - f_s)·t)` sampled at the same rate:
```
y[n] = sin(2π(f - f_s)·nT_s)
     = sin(2πf·nT_s - 2π·n)
     = sin(2πf·nT_s)   ← identical to x[n]!
```

The sampler cannot distinguish them. The signal appears at the **wrong frequency**.

**Alias Frequency Formula:**
```
f_alias = |f − k · f_s|
```
Choose integer k so that f_alias ∈ [0, f_s/2].

**Example:** f = 8 Hz, f_s = 10 Hz → f_alias = |8 − 10| = **2 Hz**

---

### 5. Anti-Aliasing Filter

The solution: apply a **Low-Pass Filter** before sampling to remove all content above f_s/2.

```
x(t) → [Low-Pass Filter at f_s/2] → x_filtered(t) → [Sample at f_s] → x[n]
```

This simulator uses a **4th-order Butterworth filter** — maximally flat, no passband ripple:

```
|H(jω)|² = 1 / (1 + (ω/ω_c)^(2N))
```

With N=4, cutoff ω_c = 2π·(f_s/2). Implemented via `scipy.signal.butter()`.

---

### 6. Signal Reconstruction

**Linear Interpolation** (used in this simulator):
```
x̂(t) ≈ np.interp(t, sample_times, sample_values)
```

**Ideal (Whittaker-Shannon):**
```
x̂(t) = Σ x[n] · sinc(t/T_s − n)
```
Perfect recovery when Nyquist is satisfied.

**Zero-Order Hold:** hold each sample value until the next — simulates basic DAC behavior.

---

### 7. The FFT (Frequency Domain View)

The Fast Fourier Transform reveals which frequencies are present:

```python
window = np.hanning(len(signal))
spectrum = np.fft.rfft(signal * window)
freqs = np.fft.rfftfreq(len(signal), d=1.0/rate)
magnitude = np.abs(spectrum)
```

- **Clean signal:** single spike at true frequency
- **Aliased signal:** spike appears at *alias* frequency (wrong!)
- **Anti-aliased:** spike is suppressed/removed entirely

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INPUT LAYER                       │
│  Sliders: f, f_s, amplitude, phase, noise, type     │
│  Presets: Oversampling / Nyquist / Aliasing         │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│           SIGNAL GENERATION LAYER                   │
│  generate_continuous_signal()                       │
│  Types: sine | multi_tone | chirp                   │
│  Resolution: 2000 Hz display rate                   │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│               SAMPLING LAYER                        │
│  sample_signal() → x[n] = x(nT_s)                 │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│       PROCESSING LAYER (COMPARATIVE)                │
│  ├─ naive_decimate_synthetic() — NO filter          │
│  └─ filtered_decimate_synthetic() — LPF + decimate  │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│             COMPUTATION LAYER                       │
│  compute_fft() · alias_frequency() · Nyquist check  │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│           VISUALIZATION LAYER                       │
│  build_time_domain_figure()                         │
│  build_reconstruction_figure()                      │
│  build_frequency_figure()                           │
│  build_naive_vs_filtered_figure()  ← KEY CHART      │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│            EXPLANATION LAYER                        │
│  Dynamic Nyquist status · alias freq · fix message  │
└─────────────────────────────────────────────────────┘
```

---

## Features

### Synthetic Signal Mode
| Feature | Details |
|---------|---------|
| Signal Types | Sine, Multi-tone, Chirp (frequency sweep) |
| Presets | Oversampling Demo, Nyquist Demo, Aliasing Demo |
| Controls | Frequency, Sampling Rate, Amplitude, Phase, Duration, Noise |
| Live Status Badge | 🟢 Oversampling / 🟡 Nyquist Limit / 🔴 Aliasing — updates as slider moves |
| 4 Charts | Time Domain, Reconstruction, FFT, Naive vs Anti-Aliased |
| Audio Preview | Listen to original vs reconstructed signal |
| Alias Prediction | Exact alias frequency computed and displayed |

### Audio Upload Mode
| Feature | Details |
|---------|---------|
| File Support | WAV files |
| Processing | Naive downsampling vs Polyphase (anti-aliased) resampling |
| Visualization | Waveforms + spectrum overlay + comparative FFT |
| Audio Playback | Original vs processed side-by-side listening |
| Auto-Analysis | Dominant frequency detected, Nyquist risk assessed |

---

## Demo Cases

### Case 1 — Oversampling ✅
```
Signal frequency:   5 Hz
Sampling frequency: 50 Hz (10× Nyquist rate)
f_s / f_Nyquist:    50 / 10 = 5× oversample
```
**Result:** Perfect reconstruction. Sampled points sit exactly on the curve.
FFT shows a clean spike at 5 Hz. Both naive and filtered outputs are identical.

---

### Case 2 — Nyquist Limit ⚠️
```
Signal frequency:   5 Hz
Sampling frequency: 10 Hz (exactly 2× signal frequency)
f_s / f_Nyquist:    10 / 10 = 1.0 (right at the limit)
```
**Result:** Theoretically recoverable, but phase-sensitive. Sampling exactly at 10 Hz when signal is 5 Hz can land all samples at zero-crossings, giving no amplitude information.

---

### Case 3 — Aliasing ❌
```
Signal frequency:   8 Hz
Sampling frequency: 10 Hz (below Nyquist)
f_alias:            |8 - 10| = 2 Hz
```
**Result:**
- Naive FFT: spike appears at **2 Hz** (the alias) — completely wrong
- Anti-Aliased FFT: spike **removed** — content above 5 Hz filtered before sampling
- The 8 Hz signal is gone entirely in the anti-aliased version — correctly so

---

## Installation & Running

### Option 1: Direct Run
```bash
cd d:\Bunny\IBM_hackathon\sampling_aliasing_demo
pip install -r requirements.txt
python -m streamlit run app.py
```

### Option 2: With virtual environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m streamlit run app.py
```

Open browser at: **http://localhost:8501**

### requirements.txt
```
streamlit>=1.40.0
numpy>=1.26.0
scipy>=1.13.0
plotly>=5.24.0
soundfile>=0.12.1
```

---

## Project Structure

```
sampling_aliasing_demo/
│
├── app.py                    # Main Streamlit app — orchestrates all layers
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
└── src/
    ├── config.py             # APP_TITLE, THEMES constants
    │
    ├── ui.py                 # Sidebar controls + live Nyquist badge + metrics panel
    │                         # → render_sidebar(), render_header(), render_status_panel()
    │
    ├── signal_models.py      # Core signal math
    │                         # → generate_continuous_signal()  [sine/chirp/multi_tone]
    │                         # → sample_signal()               [discrete sampling]
    │                         # → reconstruct_signal()          [linear / ZOH]
    │                         # → alias_frequency()             [formula: |f - k·fs|]
    │                         # → apply_anti_aliasing_filter()  [4th-order Butterworth]
    │                         # → naive_decimate_synthetic()    [no filter — shows aliasing]
    │                         # → filtered_decimate_synthetic() [LPF + decimation]
    │
    ├── processing.py         # Signal processing pipeline
    │                         # → compute_fft()                 [Hanning window + rfft]
    │                         # → run_signal_simulation()       [full synthetic pipeline]
    │                         # → run_audio_simulation()        [audio pipeline]
    │                         # → _generate_explanation()       [dynamic text]
    │
    ├── audio_utils.py        # Audio I/O utilities
    │                         # → read_uploaded_audio()         [WAV → float32 array]
    │                         # → naive_downsample()            [stride decimation]
    │                         # → proper_resample()             [polyphase resampling]
    │                         # → audio_bytes_from_signal()     [float32 → WAV bytes]
    │
    └── plots.py              # All Plotly visualizations
                              # → build_time_domain_figure()
                              # → build_reconstruction_figure()
                              # → build_frequency_figure()
                              # → build_naive_vs_filtered_figure()   ← KEY comparative chart
                              # → build_audio_naive_vs_filtered_figure()
                              # → build_audio_waveform_figure()
                              # → build_spectrum_overlay_figure()
```

---

## Real-World Applications

| Domain | Problem | Solution |
|--------|---------|---------|
| 🎵 Audio | High-pitch aliases to wrong low pitch | Anti-alias filter before ADC (e.g., CD uses LPF at 20 kHz) |
| 🎬 Video/Film | Wagon wheel spins backward (temporal aliasing) | Frame rate chosen relative to motion speed |
| 🏥 Medical ECG | QRS complexes missed or distorted | Min 250 Hz sampling for clinical ECG |
| 📡 Radar | Fast targets appear at wrong Doppler speed | f_s ≥ 2 × max Doppler shift |
| 🏭 IoT Sensors | Vibration events completely missed | Sampling matched to max event frequency |
| 📸 Cameras | Moiré patterns on fabric / screens | Optical anti-aliasing filter on sensor |
| 📻 Radio / SDR | Wrong frequency decoded | ADC with brick-wall anti-alias filter |

---

## Key Equations

| Formula | Description |
|---------|-------------|
| `x(t) = A·sin(2πft + φ)` | Continuous sinusoidal signal |
| `x[n] = x(nT_s)` | Sampled signal at discrete times |
| `f_s ≥ 2·f_max` | Nyquist-Shannon criterion |
| `f_N = f_s / 2` | Nyquist frequency (max representable) |
| `f_alias = \|f − k·f_s\|` | Alias frequency (choose k for [0, f_N] range) |
| `\|H\|² = 1/(1 + (ω/ω_c)^2N)` | Butterworth filter magnitude response |
| `x̂(t) = Σ x[n]·sinc(t/T_s − n)` | Ideal Whittaker-Shannon reconstruction |
| `X[k] = Σ x[n]·e^(−j2πkn/N)` | Discrete Fourier Transform (DFT) |

---

## Future Work

- **2D Spatial Aliasing** — Moiré patterns in image downsampling
- **Live Microphone Input** — Real-time sampling from microphone
- **3D Surface Plots** — Frequency vs sampling rate vs alias frequency surface
- **Quantization Noise** — Show bit-depth effects alongside sampling
- **SNR Analysis** — Signal-to-Noise Ratio as a function of sampling rate
- **PDF Report Export** — Auto-generate analysis report with all charts
- **Multi-Signal Mode** — Multiple aliasing tones shown simultaneously
- **Spectrogram View** — Time-frequency heatmap for chirp aliasing visualization

---

## Conclusion

This simulator makes the **Nyquist-Shannon theorem tangible**. By moving a single slider, users can watch a signal transition from perfect reconstruction → Nyquist limit → full aliasing — with live FFT confirmation of exactly what frequency the alias appears at.

The comparative Naive vs Anti-Aliased chart is the core contribution: it proves that **aliasing is not random noise — it's a predictable, preventable mathematical phenomenon** that appears precisely where the theory says it will.

---

*Built for PS18 — IBM Hackathon | Python · NumPy · SciPy · Plotly · Streamlit*
