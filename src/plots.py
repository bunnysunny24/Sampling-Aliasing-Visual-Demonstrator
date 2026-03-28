from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "continuous": "#2563eb",
    "sampled": "#ef4444",
    "reconstructed": "#10b981",
    "spectrum_original": "#7c3aed",
    "spectrum_processed": "#f59e0b",
    "naive": "#ef4444",
    "filtered": "#10b981",
    "nyquist": "#dc2626",
}


def _base_figure(theme: str, title: str, x_title: str, y_title: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=theme.lower().replace(" ", "_"),
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    return fig


def build_time_domain_figure(result: dict, theme: str) -> go.Figure:
    fig = _base_figure(theme, "📈 Continuous Signal vs Sampled Points", "Time (s)", "Amplitude")
    fig.add_trace(go.Scatter(
        x=result["continuous_time"], y=result["continuous_signal"],
        mode="lines", name="Continuous Signal",
        line=dict(color=COLORS["continuous"], width=3)
    ))
    fig.add_trace(go.Scatter(
        x=result["sample_times"], y=result["sample_values"],
        mode="markers", name="Sampled Points",
        marker=dict(color=COLORS["sampled"], size=10, symbol="circle",
                    line=dict(color="white", width=1.5))
    ))
    # Vertical stem lines from x-axis to sample points
    for st, sv in zip(result["sample_times"][::max(1, len(result["sample_times"])//40)],
                      result["sample_values"][::max(1, len(result["sample_values"])//40)]):
        fig.add_shape(type="line", x0=st, x1=st, y0=0, y1=sv,
                      line=dict(color=COLORS["sampled"], width=1, dash="dot"))
    return fig


def build_reconstruction_figure(result: dict, theme: str) -> go.Figure:
    fig = _base_figure(theme, "🔄 Original vs Reconstructed Signal", "Time (s)", "Amplitude")
    fig.add_trace(go.Scatter(
        x=result["continuous_time"], y=result["continuous_signal"],
        mode="lines", name="Original Signal",
        line=dict(color=COLORS["continuous"], width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=result["continuous_time"], y=result["reconstructed_signal"],
        mode="lines", name="Reconstructed Signal",
        line=dict(color=COLORS["reconstructed"], width=3)
    ))
    return fig


def build_frequency_figure(result: dict, theme: str) -> go.Figure:
    fig = _base_figure(theme, "🎵 Frequency-Domain Analysis (FFT)", "Frequency (Hz)", "Magnitude", height=380)
    fig.add_trace(go.Scatter(
        x=result["original_freq"], y=result["original_mag"],
        mode="lines", name="Original Spectrum",
        line=dict(color=COLORS["spectrum_original"], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=result["recon_freq"], y=result["recon_mag"],
        mode="lines", name="Reconstructed Spectrum",
        line=dict(color=COLORS["spectrum_processed"], width=2)
    ))
    nyq = result["sampling_rate"] / 2
    fig.add_vline(
        x=nyq, line_dash="dash", line_color=COLORS["nyquist"], line_width=2,
        annotation_text=f"Nyquist Limit ({nyq:.1f} Hz)",
        annotation_position="top right",
        annotation_font_color=COLORS["nyquist"]
    )
    x_max = max(result["sampling_rate"] * 2, result["signal_frequency"] * 3, 20)
    fig.update_xaxes(range=[0, x_max])
    return fig


def build_naive_vs_filtered_figure(result: dict, theme: str) -> go.Figure:
    """The KEY missing chart: Naive (aliased) vs Anti-Aliased spectrum comparison."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "❌ Naive Downsampling (Aliasing Occurs)",
            "✅ Anti-Aliased Downsampling (Correct)"
        ),
        shared_yaxes=False
    )

    nyq = result["sampling_rate"] / 2
    x_max = max(result["sampling_rate"] * 2, result["signal_frequency"] * 3, 20)

    # --- LEFT: Naive (causes aliasing) ---
    fig.add_trace(go.Scatter(
        x=result["original_freq"], y=result["original_mag"],
        mode="lines", name="Original",
        line=dict(color=COLORS["continuous"], width=1.5, dash="dot"),
        showlegend=True
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=result["naive_freq"], y=result["naive_mag"],
        mode="lines", name="Naive Output",
        line=dict(color=COLORS["naive"], width=2.5),
        showlegend=True
    ), row=1, col=1)
    fig.add_vline(
        x=nyq, line_dash="dash", line_color=COLORS["nyquist"], line_width=2,
        annotation_text=f"Nyquist ({nyq:.1f} Hz)", row=1, col=1
    )

    # --- RIGHT: Filtered (anti-aliased) ---
    fig.add_trace(go.Scatter(
        x=result["original_freq"], y=result["original_mag"],
        mode="lines", name="Original",
        line=dict(color=COLORS["continuous"], width=1.5, dash="dot"),
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=result["filtered_freq"], y=result["filtered_mag"],
        mode="lines", name="Filtered Output",
        line=dict(color=COLORS["filtered"], width=2.5),
        showlegend=True
    ), row=1, col=2)
    fig.add_vline(
        x=nyq, line_dash="dash", line_color=COLORS["nyquist"], line_width=2,
        annotation_text=f"Nyquist ({nyq:.1f} Hz)", row=1, col=2
    )

    fig.update_xaxes(range=[0, x_max], title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Magnitude")
    fig.update_layout(
        template=theme.lower().replace(" ", "_"),
        title=dict(
            text="⚖️ Comparative Analysis: Naive vs Anti-Aliased Downsampling",
            font=dict(size=16)
        ),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=90, b=50),
    )
    return fig


def build_audio_naive_vs_filtered_figure(result: dict, theme: str) -> go.Figure:
    """Side-by-side for audio mode: naive vs filtered spectra."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "❌ Naive Downsampling",
            "✅ Anti-Aliased Downsampling"
        ),
        shared_yaxes=False,
    )
    x_max = min(result["original_rate"] / 2, 12000)
    nyq = result["target_rate"] / 2

    fig.add_trace(go.Scatter(
        x=result["original_freq"], y=result["original_mag"],
        mode="lines", name="Original",
        line=dict(color=COLORS["continuous"], width=1.5, dash="dot"), showlegend=True
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=result["naive_freq"], y=result["naive_mag"],
        mode="lines", name="Naive",
        line=dict(color=COLORS["naive"], width=2.5), showlegend=True
    ), row=1, col=1)
    fig.add_vline(x=nyq, line_dash="dash", line_color=COLORS["nyquist"],
                  annotation_text=f"Nyquist ({nyq:.0f} Hz)", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=result["original_freq"], y=result["original_mag"],
        mode="lines", name="Original",
        line=dict(color=COLORS["continuous"], width=1.5, dash="dot"), showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=result["filtered_freq"], y=result["filtered_mag"],
        mode="lines", name="Anti-Aliased",
        line=dict(color=COLORS["filtered"], width=2.5), showlegend=True
    ), row=1, col=2)
    fig.add_vline(x=nyq, line_dash="dash", line_color=COLORS["nyquist"],
                  annotation_text=f"Nyquist ({nyq:.0f} Hz)", row=1, col=2)

    fig.update_xaxes(range=[0, x_max], title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Magnitude")
    fig.update_layout(
        template=theme.lower().replace(" ", "_"),
        title=dict(text="⚖️ Audio Naive vs Anti-Aliased Spectrum Comparison", font=dict(size=16)),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=90, b=50),
    )
    return fig


def build_audio_waveform_figure(signal: np.ndarray, rate: int, title: str, theme: str) -> go.Figure:
    preview_len = min(len(signal), rate * 2)
    t = np.arange(preview_len) / rate
    fig = _base_figure(theme, title, "Time (s)", "Amplitude")
    fig.add_trace(go.Scatter(
        x=t, y=signal[:preview_len],
        mode="lines", line=dict(color=COLORS["continuous"], width=2), name=title
    ))
    return fig


def build_spectrum_overlay_figure(result: dict, theme: str) -> go.Figure:
    fig = _base_figure(theme, "🎵 Original vs Processed Spectrum", "Frequency (Hz)", "Magnitude")
    fig.add_trace(go.Scatter(
        x=result["original_freq"], y=result["original_mag"],
        mode="lines", name="Original",
        line=dict(color=COLORS["spectrum_original"], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=result["processed_freq"], y=result["processed_mag"],
        mode="lines", name="Processed",
        line=dict(color=COLORS["spectrum_processed"], width=2)
    ))
    nyq = result["target_rate"] / 2
    fig.add_vline(
        x=nyq, line_dash="dash", line_color=COLORS["nyquist"], line_width=2,
        annotation_text=f"Target Nyquist ({nyq:.0f} Hz)",
        annotation_font_color=COLORS["nyquist"]
    )
    fig.update_xaxes(range=[0, min(result["original_rate"] / 2, 12000)])
    return fig
