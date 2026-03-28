from __future__ import annotations
import streamlit as st
from src.config import APP_TITLE, THEMES


def render_header() -> None:
    st.markdown(
        """
        <div style="padding: 1rem 0 0.5rem 0;">
            <h1 style="margin-bottom: 0.1rem;">🌊 Sampling & Aliasing Visual Demonstrator</h1>
            <p style="color: gray; margin-top: 0.1rem;">
                Interactive simulation of oversampling, Nyquist-rate sampling, undersampling, and aliasing —
                PS18 | IBM Hackathon
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_sidebar() -> tuple[str, dict]:
    st.sidebar.header("🎛️ Controls")
    mode = st.sidebar.radio("Mode", ["Synthetic Signal", "Audio Upload"])
    theme = st.sidebar.selectbox("Graph Theme", THEMES, index=0)

    common = {"theme": theme}

    if mode == "Synthetic Signal":
        # --- SIGNAL GENERATION CONTROL ---
        st.sidebar.subheader("📡 Signal Generation")
        signal_type = st.sidebar.selectbox("Signal Type", ["sine", "multi_tone", "chirp"])

        preset = st.sidebar.selectbox(
            "Preset", ["Custom", "Oversampling Demo", "Nyquist Demo", "Aliasing Demo"]
        )
        defaults = {
            "Custom":           (5.0, 20.0),
            "Oversampling Demo": (5.0, 50.0),
            "Nyquist Demo":     (5.0, 10.0),
            "Aliasing Demo":    (8.0, 10.0),
        }
        default_signal, default_sampling = defaults[preset]

        signal_frequency = st.sidebar.slider(
            "Signal Frequency (Hz)", 1.0, 30.0, float(default_signal), 0.5
        )
        sampling_rate = st.sidebar.slider(
            "Sampling Frequency (Hz)", 2.0, 80.0, float(default_sampling), 0.5
        )

        # Inline Nyquist status indicator
        nyquist_needed = 2 * signal_frequency
        if sampling_rate >= nyquist_needed * 1.05:
            st.sidebar.success(f"✅ Oversampling — fs = {sampling_rate} Hz ≥ {nyquist_needed} Hz")
        elif abs(sampling_rate - nyquist_needed) <= 0.6:
            st.sidebar.warning(f"⚠️ At Nyquist Limit — fs = {sampling_rate} Hz ≈ {nyquist_needed} Hz")
        else:
            st.sidebar.error(
                f"❌ ALIASING! fs = {sampling_rate} Hz < {nyquist_needed} Hz required"
            )

        amplitude    = st.sidebar.slider("Amplitude", 0.2, 2.0, 1.0, 0.1)
        phase        = st.sidebar.slider("Phase (radians)", 0.0, 6.28, 0.0, 0.1)
        duration     = st.sidebar.slider("Duration (seconds)", 1.0, 5.0, 2.0, 0.5)
        display_rate = st.sidebar.slider("Continuous Display Rate (Hz)", 200, 5000, 2000, 100)
        noise_level  = st.sidebar.slider("Noise Level (σ)", 0.0, 0.3, 0.0, 0.01)

        # --- RECONSTRUCTION CONTROL ---
        st.sidebar.subheader("🔄 Reconstruction")
        reconstruction_method = st.sidebar.selectbox(
            "Interpolation Method", ["linear", "zero_order_hold"]
        )

        # --- MULTI-TONE CONTROL ---
        st.sidebar.subheader("🎵 Multi-Tone Control")
        add_second_tone   = st.sidebar.checkbox("Add second tone", value=False)
        second_frequency  = st.sidebar.slider(
            "Second Tone Frequency (Hz)", 1.0, 40.0, 12.0, 0.5,
            disabled=not add_second_tone
        )
        second_amplitude  = st.sidebar.slider(
            "Second Tone Amplitude", 0.1, 1.0, 0.4, 0.1,
            disabled=not add_second_tone
        )

        settings = {
            **common,
            "signal_type":           signal_type,
            "signal_frequency":      signal_frequency,
            "sampling_rate":         sampling_rate,
            "amplitude":             amplitude,
            "phase":                 phase,
            "duration":              duration,
            "display_rate":          display_rate,
            "noise_level":           noise_level,
            "reconstruction_method": reconstruction_method,
            "add_second_tone":       add_second_tone,
            "second_frequency":      second_frequency,
            "second_amplitude":      second_amplitude,
        }
        return mode, settings

    # --- AUDIO UPLOAD CONTROLS ---
    st.sidebar.subheader("🎙️ Audio Processing")
    target_audio_rate = st.sidebar.selectbox(
        "Target Sample Rate (Hz)",
        [48000, 44100, 22050, 16000, 12000, 8000, 4000],
        index=3
    )
    audio_mode = st.sidebar.radio(
        "Processing Mode", ["Naive Downsampling", "Filtered Downsampling"]
    )

    settings = {
        **common,
        "target_audio_rate": target_audio_rate,
        "audio_mode":        audio_mode,
    }
    return mode, settings


def render_status_panel(result: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)

    sr = result["sampling_rate"]
    sr_str = f"{sr:.2f} Hz" if isinstance(sr, float) else f"{sr} Hz"

    with col1:
        st.metric("📡 Sampling Rate", sr_str)
    with col2:
        st.metric("📐 Nyquist Requirement", f"{result['nyquist_required']:.2f} Hz")
    with col3:
        status = result["status"]
        # Status may contain emoji prefixes already from audio mode
        st.metric("🔍 Status", status)
    with col4:
        alias = result["apparent_alias_frequency"]
        st.metric(
            "🔀 Alias Frequency",
            f"{alias:.2f} Hz" if not (alias != alias) else "N/A"  # NaN check
        )
