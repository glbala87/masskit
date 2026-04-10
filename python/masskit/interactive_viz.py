"""
Interactive visualization for LC-MS data using Plotly.

Provides zoomable, pannable, interactive plots for spectra,
chromatograms, heatmaps, and feature maps.

Requires: plotly (pip install plotly)
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from .spectrum import Spectrum, SpectrumType
from .chromatogram import Chromatogram
from .peak import Peak, PeakList
from .feature import Feature, FeatureMap
from .experiment import MSExperiment


def _check_plotly():
    """Check if plotly is installed."""
    try:
        import plotly
        return True
    except ImportError:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install with: pip install plotly"
        )


def plot_spectrum_interactive(
    spectrum: Spectrum,
    mz_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    color: str = "#1f77b4",
    height: int = 400,
    show: bool = True,
) -> Any:
    """
    Interactive spectrum plot with zoom, pan, and hover tooltips.

    Args:
        spectrum: Spectrum to plot
        mz_range: Optional (min, max) m/z range
        title: Plot title
        color: Trace color
        height: Plot height in pixels
        show: Show plot immediately

    Returns:
        Plotly Figure object
    """
    _check_plotly()
    import plotly.graph_objects as go

    mz = spectrum.mz
    intensity = spectrum.intensity

    if mz_range is not None:
        mask = (mz >= mz_range[0]) & (mz <= mz_range[1])
        mz = mz[mask]
        intensity = intensity[mask]

    if title is None:
        title = f"MS{spectrum.ms_level} Spectrum @ RT {spectrum.rt:.2f}s"

    fig = go.Figure()

    if spectrum.spectrum_type == SpectrumType.CENTROID:
        # Stick plot for centroid data
        for m, i in zip(mz, intensity):
            fig.add_trace(go.Scatter(
                x=[m, m, None],
                y=[0, i, None],
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False,
                hoverinfo="text",
                text=[f"m/z: {m:.4f}<br>Intensity: {i:.0f}"] * 3,
            ))
    else:
        fig.add_trace(go.Scatter(
            x=mz, y=intensity,
            mode="lines",
            line=dict(color=color, width=1),
            fill="tozeroy",
            fillcolor=f"rgba(31,119,180,0.1)",
            name="Spectrum",
            hovertemplate="m/z: %{x:.4f}<br>Intensity: %{y:.0f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="m/z",
        yaxis_title="Intensity",
        height=height,
        template="plotly_white",
        hovermode="closest",
        xaxis=dict(range=mz_range),
        yaxis=dict(rangemode="tozero"),
    )

    if show:
        fig.show()
    return fig


def plot_chromatogram_interactive(
    chromatogram: Chromatogram,
    rt_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    color: str = "#1f77b4",
    height: int = 400,
    show: bool = True,
) -> Any:
    """
    Interactive chromatogram plot.

    Args:
        chromatogram: Chromatogram to plot
        rt_range: Optional (min, max) RT range
        title: Plot title
        color: Trace color
        height: Plot height in pixels
        show: Show plot immediately

    Returns:
        Plotly Figure object
    """
    _check_plotly()
    import plotly.graph_objects as go

    rt = chromatogram.rt
    intensity = chromatogram.intensity

    if rt_range is not None:
        mask = (rt >= rt_range[0]) & (rt <= rt_range[1])
        rt = rt[mask]
        intensity = intensity[mask]

    if title is None:
        type_name = chromatogram.chrom_type.name
        if chromatogram.target_mz > 0:
            title = f"{type_name} @ m/z {chromatogram.target_mz:.4f}"
        else:
            title = type_name

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rt, y=intensity,
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba(31,119,180,0.15)",
        name="Chromatogram",
        hovertemplate="RT: %{x:.2f}s<br>Intensity: %{y:.0f}<extra></extra>",
    ))

    # Mark apex
    if len(intensity) > 0:
        apex_idx = np.argmax(intensity)
        fig.add_trace(go.Scatter(
            x=[rt[apex_idx]], y=[intensity[apex_idx]],
            mode="markers+text",
            marker=dict(color="red", size=8),
            text=[f"RT: {rt[apex_idx]:.2f}s"],
            textposition="top center",
            showlegend=False,
            hovertemplate=f"Apex<br>RT: {rt[apex_idx]:.2f}s<br>Int: {intensity[apex_idx]:.0f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Retention Time (s)",
        yaxis_title="Intensity",
        height=height,
        template="plotly_white",
        hovermode="closest",
        yaxis=dict(rangemode="tozero"),
    )

    if show:
        fig.show()
    return fig


def plot_heatmap_interactive(
    experiment: MSExperiment,
    mz_range: Optional[Tuple[float, float]] = None,
    rt_range: Optional[Tuple[float, float]] = None,
    mz_bins: int = 500,
    rt_bins: int = 200,
    log_scale: bool = True,
    colorscale: str = "Viridis",
    ms_level: int = 1,
    title: Optional[str] = None,
    height: int = 600,
    show: bool = True,
) -> Any:
    """
    Interactive 2D heatmap (RT vs m/z) with zoom and hover.

    Args:
        experiment: MSExperiment
        mz_range: (min, max) m/z range
        rt_range: (min, max) RT range
        mz_bins: Number of m/z bins
        rt_bins: Number of RT bins
        log_scale: Use log intensity scale
        colorscale: Plotly colorscale name
        ms_level: MS level to plot
        title: Plot title
        height: Plot height
        show: Show plot immediately

    Returns:
        Plotly Figure object
    """
    _check_plotly()
    import plotly.graph_objects as go

    spectra = [s for s in experiment.spectra if s.ms_level == ms_level] if ms_level > 0 else experiment.spectra

    if not spectra:
        fig = go.Figure()
        fig.update_layout(title="No spectra to display")
        if show:
            fig.show()
        return fig

    if mz_range is None:
        all_mz_mins = [np.min(s.mz) for s in spectra if len(s.mz) > 0]
        all_mz_maxs = [np.max(s.mz) for s in spectra if len(s.mz) > 0]
        mz_range = (min(all_mz_mins), max(all_mz_maxs)) if all_mz_mins else (0, 1000)

    if rt_range is None:
        rts = [s.rt for s in spectra]
        rt_range = (min(rts), max(rts))

    heatmap = np.zeros((rt_bins, mz_bins))

    for spec in spectra:
        if not (rt_range[0] <= spec.rt <= rt_range[1]):
            continue

        rt_idx = int((spec.rt - rt_range[0]) / (rt_range[1] - rt_range[0]) * rt_bins)
        rt_idx = min(rt_idx, rt_bins - 1)

        mask = (spec.mz >= mz_range[0]) & (spec.mz <= mz_range[1])
        if not np.any(mask):
            continue

        mz_vals = spec.mz[mask]
        int_vals = spec.intensity[mask]

        mz_indices = ((mz_vals - mz_range[0]) / (mz_range[1] - mz_range[0]) * mz_bins).astype(int)
        mz_indices = np.clip(mz_indices, 0, mz_bins - 1)

        for mi, iv in zip(mz_indices, int_vals):
            heatmap[rt_idx, mi] = max(heatmap[rt_idx, mi], iv)

    if log_scale:
        heatmap = np.log1p(heatmap)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap,
        x=np.linspace(mz_range[0], mz_range[1], mz_bins),
        y=np.linspace(rt_range[0], rt_range[1], rt_bins),
        colorscale=colorscale,
        colorbar=dict(title="log(Int+1)" if log_scale else "Intensity"),
        hovertemplate="m/z: %{x:.4f}<br>RT: %{y:.2f}s<br>Intensity: %{z:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=title or f"LC-MS Heatmap (MS{ms_level})",
        xaxis_title="m/z",
        yaxis_title="Retention Time (s)",
        height=height,
        template="plotly_white",
    )

    if show:
        fig.show()
    return fig


def plot_mirror_interactive(
    spectrum1: Spectrum,
    spectrum2: Spectrum,
    labels: Tuple[str, str] = ("Query", "Reference"),
    mz_range: Optional[Tuple[float, float]] = None,
    score: Optional[float] = None,
    height: int = 500,
    show: bool = True,
) -> Any:
    """
    Interactive mirror plot for spectral comparison.

    Args:
        spectrum1: Top spectrum (positive y)
        spectrum2: Bottom spectrum (negative y)
        labels: Labels for the spectra
        mz_range: m/z range
        score: Optional similarity score to display
        height: Plot height
        show: Show plot immediately

    Returns:
        Plotly Figure object
    """
    _check_plotly()
    import plotly.graph_objects as go

    mz1, int1 = spectrum1.mz.copy(), spectrum1.intensity.copy()
    mz2, int2 = spectrum2.mz.copy(), spectrum2.intensity.copy()

    if mz_range:
        mask1 = (mz1 >= mz_range[0]) & (mz1 <= mz_range[1])
        mask2 = (mz2 >= mz_range[0]) & (mz2 <= mz_range[1])
        mz1, int1 = mz1[mask1], int1[mask1]
        mz2, int2 = mz2[mask2], int2[mask2]

    # Normalize to 100%
    if len(int1) > 0:
        int1 = int1 / np.max(int1) * 100
    if len(int2) > 0:
        int2 = int2 / np.max(int2) * 100

    fig = go.Figure()

    # Top spectrum
    for m, i in zip(mz1, int1):
        fig.add_trace(go.Scatter(
            x=[m, m, None], y=[0, i, None],
            mode="lines", line=dict(color="blue", width=1),
            showlegend=False,
            hoverinfo="text",
            text=[f"{labels[0]}<br>m/z: {m:.4f}<br>Int: {i:.1f}%"] * 3,
        ))

    # Bottom spectrum (inverted)
    for m, i in zip(mz2, int2):
        fig.add_trace(go.Scatter(
            x=[m, m, None], y=[0, -i, None],
            mode="lines", line=dict(color="red", width=1),
            showlegend=False,
            hoverinfo="text",
            text=[f"{labels[1]}<br>m/z: {m:.4f}<br>Int: {i:.1f}%"] * 3,
        ))

    # Center line
    fig.add_hline(y=0, line_width=1, line_color="black")

    title = f"Mirror Plot: {labels[0]} vs {labels[1]}"
    if score is not None:
        title += f" (Score: {score:.4f})"

    fig.update_layout(
        title=title,
        xaxis_title="m/z",
        yaxis_title="Relative Intensity (%)",
        height=height,
        template="plotly_white",
        xaxis=dict(range=mz_range),
    )

    if show:
        fig.show()
    return fig


def plot_feature_map_interactive(
    features: FeatureMap,
    color_by: str = "intensity",
    title: Optional[str] = None,
    height: int = 600,
    show: bool = True,
) -> Any:
    """
    Interactive 2D feature map (RT vs m/z).

    Args:
        features: FeatureMap to plot
        color_by: Color by 'intensity', 'charge', or 'quality'
        title: Plot title
        height: Plot height
        show: Show plot immediately

    Returns:
        Plotly Figure object
    """
    _check_plotly()
    import plotly.graph_objects as go

    if len(features) == 0:
        fig = go.Figure()
        fig.update_layout(title="No features to display")
        if show:
            fig.show()
        return fig

    mzs = [f.mz for f in features]
    rts = [f.rt for f in features]
    intensities = [f.intensity for f in features]

    if color_by == "charge":
        colors = [f.charge for f in features]
        colorbar_title = "Charge"
    elif color_by == "quality":
        colors = [f.quality for f in features]
        colorbar_title = "Quality"
    else:
        colors = np.log10(np.array(intensities) + 1)
        colorbar_title = "log10(Intensity)"

    hover_text = [
        f"m/z: {f.mz:.4f}<br>"
        f"RT: {f.rt:.2f}s<br>"
        f"Intensity: {f.intensity:.0f}<br>"
        f"Charge: {f.charge}<br>"
        f"Quality: {f.quality:.2f}"
        for f in features
    ]

    fig = go.Figure(data=go.Scatter(
        x=rts, y=mzs,
        mode="markers",
        marker=dict(
            size=6,
            color=colors,
            colorscale="Viridis",
            colorbar=dict(title=colorbar_title),
            opacity=0.7,
        ),
        text=hover_text,
        hoverinfo="text",
    ))

    fig.update_layout(
        title=title or f"Feature Map ({len(features)} features)",
        xaxis_title="Retention Time (s)",
        yaxis_title="m/z",
        height=height,
        template="plotly_white",
    )

    if show:
        fig.show()
    return fig


def plot_tic_interactive(
    experiment: MSExperiment,
    ms_level: int = 1,
    title: Optional[str] = None,
    height: int = 400,
    show: bool = True,
) -> Any:
    """Interactive TIC plot."""
    tic = experiment.generate_tic(level=ms_level)
    return plot_chromatogram_interactive(
        tic,
        title=title or f"Total Ion Chromatogram (MS{ms_level})",
        height=height,
        show=show,
    )


def plot_multi_xic_interactive(
    experiment: MSExperiment,
    target_mzs: List[float],
    tolerance: float = 0.01,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    height: int = 400,
    show: bool = True,
) -> Any:
    """
    Interactive overlay of multiple XICs.

    Args:
        experiment: MSExperiment
        target_mzs: List of target m/z values
        tolerance: m/z tolerance
        labels: Optional labels for each XIC
        title: Plot title
        height: Plot height
        show: Show plot immediately

    Returns:
        Plotly Figure object
    """
    _check_plotly()
    import plotly.graph_objects as go

    fig = go.Figure()

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    for i, mz in enumerate(target_mzs):
        xic = experiment.generate_xic(mz, tolerance)
        label = labels[i] if labels and i < len(labels) else f"m/z {mz:.4f}"
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=xic.rt, y=xic.intensity,
            mode="lines",
            line=dict(color=color, width=1.5),
            name=label,
            hovertemplate=f"{label}<br>RT: %{{x:.2f}}s<br>Int: %{{y:.0f}}<extra></extra>",
        ))

    fig.update_layout(
        title=title or "Extracted Ion Chromatograms",
        xaxis_title="Retention Time (s)",
        yaxis_title="Intensity",
        height=height,
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(rangemode="tozero"),
    )

    if show:
        fig.show()
    return fig


def create_dashboard(
    experiment: MSExperiment,
    port: int = 8050,
) -> None:
    """
    Launch an interactive Dash dashboard for exploring LC-MS data.

    Requires: dash (pip install dash)

    Args:
        experiment: MSExperiment to explore
        port: Port number for the web server
    """
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
    except ImportError:
        raise ImportError(
            "dash is required for the dashboard. "
            "Install with: pip install dash"
        )

    import plotly.graph_objects as go

    app = dash.Dash(__name__)

    ms1_spectra = [s for s in experiment.spectra if s.ms_level == 1]
    rt_min = min(s.rt for s in ms1_spectra) if ms1_spectra else 0
    rt_max = max(s.rt for s in ms1_spectra) if ms1_spectra else 100

    app.layout = html.Div([
        html.H1("LCMS Toolkit - Data Explorer"),

        html.Div([
            html.Div([
                html.H3("TIC"),
                dcc.Graph(id="tic-plot"),
            ], style={"width": "100%"}),

            html.Div([
                html.Div([
                    html.H3("Spectrum"),
                    html.Label("Select spectrum by RT:"),
                    dcc.Slider(
                        id="rt-slider",
                        min=rt_min, max=rt_max,
                        step=(rt_max - rt_min) / 100 if rt_max > rt_min else 1,
                        value=rt_min,
                        marks={int(rt_min): f"{rt_min:.0f}s",
                               int(rt_max): f"{rt_max:.0f}s"},
                    ),
                    dcc.Graph(id="spectrum-plot"),
                ], style={"width": "48%", "display": "inline-block"}),

                html.Div([
                    html.H3("XIC"),
                    html.Label("Target m/z:"),
                    dcc.Input(id="xic-mz", type="number", value=500, step=0.01),
                    html.Label(" Tolerance:"),
                    dcc.Input(id="xic-tol", type="number", value=0.01, step=0.001),
                    dcc.Graph(id="xic-plot"),
                ], style={"width": "48%", "display": "inline-block", "float": "right"}),
            ]),
        ]),

        html.Div([
            html.P(f"File: {experiment.num_spectra} spectra | "
                   f"RT: {rt_min:.1f} - {rt_max:.1f}s"),
        ], style={"color": "gray", "fontSize": "12px"}),
    ])

    @app.callback(Output("tic-plot", "figure"), Input("rt-slider", "value"))
    def update_tic(_):
        tic = experiment.generate_tic(level=1)
        fig = go.Figure(go.Scatter(
            x=tic.rt, y=tic.intensity,
            mode="lines", fill="tozeroy",
            line=dict(color="#1f77b4", width=1),
        ))
        fig.update_layout(
            xaxis_title="RT (s)", yaxis_title="Intensity",
            height=250, margin=dict(t=10, b=40),
            template="plotly_white",
        )
        return fig

    @app.callback(Output("spectrum-plot", "figure"), Input("rt-slider", "value"))
    def update_spectrum(rt_value):
        # Find closest spectrum
        best_spec = min(ms1_spectra, key=lambda s: abs(s.rt - rt_value)) if ms1_spectra else None
        if best_spec is None:
            return go.Figure()

        fig = go.Figure(go.Scatter(
            x=best_spec.mz, y=best_spec.intensity,
            mode="lines", line=dict(color="#1f77b4", width=0.5),
            hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.0f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"MS1 @ RT {best_spec.rt:.2f}s",
            xaxis_title="m/z", yaxis_title="Intensity",
            height=350, template="plotly_white",
        )
        return fig

    @app.callback(
        Output("xic-plot", "figure"),
        [Input("xic-mz", "value"), Input("xic-tol", "value")]
    )
    def update_xic(mz_value, tol_value):
        if mz_value is None or tol_value is None:
            return go.Figure()

        xic = experiment.generate_xic(float(mz_value), float(tol_value))
        fig = go.Figure(go.Scatter(
            x=xic.rt, y=xic.intensity,
            mode="lines", fill="tozeroy",
            line=dict(color="#ff7f0e", width=1.5),
        ))
        fig.update_layout(
            title=f"XIC @ m/z {mz_value:.4f}",
            xaxis_title="RT (s)", yaxis_title="Intensity",
            height=350, template="plotly_white",
        )
        return fig

    print(f"Starting LCMS dashboard at http://localhost:{port}")
    app.run(port=port, debug=False)
