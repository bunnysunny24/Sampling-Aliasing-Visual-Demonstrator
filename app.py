import streamlit as st
from src.ui import render_sidebar, render_header, render_status_panel
from src.processing import run_signal_simulation, run_audio_simulation
from src.plots import (
    build_time_domain_figure,
    build_reconstruction_figure,
    build_frequency_figure,
    build_naive_vs_filtered_figure,
    build_audio_waveform_figure,
    build_spectrum_overlay_figure,
    build_audio_naive_vs_filtered_figure,
)
from src.audio_utils import audio_bytes_from_signal, read_uploaded_audio
from src.config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🌊")

render_header()
mode, settings = render_sidebar()

# ══════════════════════════════════════════════════════
#  SYNTHETIC SIGNAL MODE
# ══════════════════════════════════════════════════════
if mode == "Synthetic Signal":
    result = run_signal_simulation(settings)

    # === STATUS METRICS ===
    render_status_panel(result)

    # === EXPLANATION LAYER ===
    with st.expander("📊 Methodology & Analysis", expanded=True):
        st.markdown(result["explanation"])

    st.divider()

    # === VIEW 1 & 2: Time-domain (side by side) ===
    st.subheader("📈 Time-Domain Views")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            build_time_domain_figure(result, theme=settings["theme"]),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            build_reconstruction_figure(result, theme=settings["theme"]),
            use_container_width=True
        )

    st.divider()

    # === VIEW 3: Frequency domain ===
    st.subheader("🎵 Frequency-Domain View (FFT)")
    st.plotly_chart(
        build_frequency_figure(result, theme=settings["theme"]),
        use_container_width=True
    )

    st.divider()

    # === VIEW 4: THE KEY CHART — Naive vs Anti-Aliased ===
    st.subheader("⚖️ Comparative Analysis: Naive vs Anti-Aliased Downsampling")
    st.caption(
        "**Left:** Naive decimation — takes every N-th sample with NO filtering. "
        "High-frequency components fold back (alias) into the low-frequency range. &nbsp;|&nbsp; "
        "**Right:** Anti-aliased decimation — applies Butterworth low-pass filter at Nyquist limit "
        "before decimation. Frequencies above f_s/2 are cleanly removed."
    )
    st.plotly_chart(
        build_naive_vs_filtered_figure(result, theme=settings["theme"]),
        use_container_width=True
    )

    st.divider()

    # === AUDIO PREVIEW ===
    st.subheader("🎧 Audio Preview: Original vs Reconstructed")
    p1, p2 = st.columns(2)
    with p1:
        st.caption("Original continuous signal")
        st.audio(
            audio_bytes_from_signal(result["continuous_signal"], result["display_rate"]),
            format="audio/wav"
        )
    with p2:
        st.caption("Sampled and reconstructed signal")
        st.audio(
            audio_bytes_from_signal(result["reconstructed_signal"], result["display_rate"]),
            format="audio/wav"
        )

# ══════════════════════════════════════════════════════
#  AUDIO UPLOAD MODE
# ══════════════════════════════════════════════════════
else:
    st.subheader("🎙️ Audio Upload: Real-Signal Experiment")
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if not uploaded:
        st.info(
            "📂 Upload a WAV file to compare original audio with downsampled versions "
            "using **naive** and **anti-aliased** processing."
        )
        st.stop()

    audio_data = read_uploaded_audio(uploaded)
    result = run_audio_simulation(audio_data, settings)

    # === STATUS METRICS ===
    render_status_panel(result)

    # === EXPLANATION LAYER ===
    with st.expander("📊 Methodology & Analysis", expanded=True):
        st.markdown(result["explanation"])

    st.divider()

    # === WAVEFORMS ===
    st.subheader("📈 Waveform Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            build_audio_waveform_figure(
                result["original_signal"], result["original_rate"],
                f"Original Audio ({result['original_rate']} Hz)", theme=settings["theme"]
            ),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            build_audio_waveform_figure(
                result["processed_signal"], result["original_rate"],
                f"Processed Audio — {result['mode_label']}", theme=settings["theme"]
            ),
            use_container_width=True
        )

    st.divider()

    # === SPECTRUM OVERLAY ===
    st.subheader("🎵 Spectrum: Original vs Selected Mode")
    st.plotly_chart(
        build_spectrum_overlay_figure(result, theme=settings["theme"]),
        use_container_width=True
    )

    st.divider()

    # === KEY CHART: naive vs filtered for audio ===
    st.subheader("⚖️ Comparative Analysis: Naive vs Anti-Aliased (Audio)")
    st.caption(
        "See how aliasing artifacts appear in the naive downsampled spectrum vs the clean filtered version."
    )
    st.plotly_chart(
        build_audio_naive_vs_filtered_figure(result, theme=settings["theme"]),
        use_container_width=True
    )

    st.divider()

    # === AUDIO PLAYBACK ===
    st.subheader("🎧 Listen & Compare")
    p1, p2 = st.columns(2)
    with p1:
        st.caption(f"Original audio ({result['original_rate']} Hz)")
        st.audio(
            audio_bytes_from_signal(result["original_signal"], result["original_rate"]),
            format="audio/wav"
        )
    with p2:
        st.caption(
            f"Processed audio ({result['target_rate']} Hz target, "
            f"replayed at {result['original_rate']} Hz)"
        )
        st.audio(
            audio_bytes_from_signal(result["processed_signal"], result["original_rate"]),
            format="audio/wav"
        )
