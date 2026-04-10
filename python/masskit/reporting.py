"""
Report generation module for LC-MS analysis results.

Generates HTML and PDF reports with embedded plots, tables,
and summary statistics from LC-MS experiments.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import datetime
import base64
import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """A section within a report."""
    title: str = ""
    content: str = ""  # HTML content
    figures: List[str] = field(default_factory=list)  # base64 encoded images
    tables: List[Dict[str, Any]] = field(default_factory=list)
    order: int = 0


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "LC-MS Analysis Report"
    author: str = ""
    date: str = ""
    logo_path: Optional[str] = None
    include_raw_data: bool = False
    include_qc: bool = True
    include_statistics: bool = True
    color_scheme: str = "default"
    page_size: str = "A4"

    def __post_init__(self):
        if not self.date:
            self.date = datetime.date.today().isoformat()


class ReportBuilder:
    """
    Build HTML/PDF reports from LC-MS analysis results.

    Example:
        >>> builder = ReportBuilder(ReportConfig(title="My Analysis"))
        >>> builder.add_summary(experiment)
        >>> builder.add_qc_section(qc_metrics)
        >>> builder.add_statistics_section(pca_result)
        >>> builder.save_html("report.html")
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.sections: List[ReportSection] = []
        self._order_counter = 0

    def add_section(
        self,
        title: str,
        content: str = "",
        figures: Optional[List[str]] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add a custom section to the report."""
        self._order_counter += 1
        self.sections.append(ReportSection(
            title=title,
            content=content,
            figures=figures or [],
            tables=tables or [],
            order=self._order_counter,
        ))

    def add_summary(
        self,
        n_spectra: int = 0,
        n_ms1: int = 0,
        n_ms2: int = 0,
        n_features: int = 0,
        n_identified: int = 0,
        rt_range: Optional[tuple] = None,
        mz_range: Optional[tuple] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add experiment summary section."""
        rows = [
            ("Total Spectra", str(n_spectra)),
            ("MS1 Spectra", str(n_ms1)),
            ("MS2 Spectra", str(n_ms2)),
            ("Features Detected", str(n_features)),
            ("Identifications", str(n_identified)),
        ]
        if rt_range:
            rows.append(("RT Range", f"{rt_range[0]:.1f} - {rt_range[1]:.1f} s"))
        if mz_range:
            rows.append(("m/z Range", f"{mz_range[0]:.1f} - {mz_range[1]:.1f}"))
        if extra:
            for key, val in extra.items():
                rows.append((key, str(val)))

        table = {"headers": ["Parameter", "Value"], "rows": rows}
        self.add_section("Experiment Summary", tables=[table])

    def add_qc_section(self, qc_metrics: Any) -> None:
        """Add QC metrics section from a QCMetrics object."""
        content = ""
        tables = []

        if hasattr(qc_metrics, "tic_cv"):
            rows = [
                ("TIC CV", f"{qc_metrics.tic_cv:.3f}"),
                ("MS1 Count", str(getattr(qc_metrics, "ms1_count", "N/A"))),
                ("MS2 Count", str(getattr(qc_metrics, "ms2_count", "N/A"))),
                ("Median Peak Width", f"{getattr(qc_metrics, 'median_peak_width', 0):.2f} s"),
                ("Peak Capacity", str(getattr(qc_metrics, "peak_capacity", "N/A"))),
                ("Dynamic Range", f"{getattr(qc_metrics, 'dynamic_range', 0):.1f}"),
            ]
            tables.append({"headers": ["Metric", "Value"], "rows": rows})

            if qc_metrics.tic_cv < 0.2:
                content = '<p class="status-good">QC Status: PASS - TIC CV within acceptable range</p>'
            else:
                content = '<p class="status-warn">QC Status: WARNING - TIC CV exceeds threshold</p>'

        self.add_section("Quality Control", content=content, tables=tables)

    def add_statistics_section(
        self,
        pca_result: Optional[Any] = None,
        anova_results: Optional[List[Any]] = None,
        volcano_data: Optional[tuple] = None,
    ) -> None:
        """Add statistical analysis section."""
        content_parts = []

        if pca_result is not None and hasattr(pca_result, "explained_variance_ratio"):
            evr = pca_result.explained_variance_ratio
            rows = []
            for i, ratio in enumerate(evr):
                rows.append((f"PC{i+1}", f"{ratio:.1%}", f"{sum(evr[:i+1]):.1%}"))
            self.add_section(
                "PCA Results",
                tables=[{"headers": ["Component", "Variance Explained", "Cumulative"], "rows": rows}],
            )

        if anova_results:
            sig = [r for r in anova_results if getattr(r, "significant", False)]
            content_parts.append(
                f"<p>ANOVA: {len(sig)} of {len(anova_results)} features significant (adjusted p < 0.05)</p>"
            )

        if content_parts:
            self.add_section("Statistical Analysis", content="\n".join(content_parts))

    def add_identification_section(
        self,
        n_psms: int = 0,
        n_peptides: int = 0,
        n_proteins: int = 0,
        fdr_threshold: float = 0.01,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add identification results section."""
        rows = [
            ("PSMs (at FDR)", f"{n_psms} (< {fdr_threshold:.0%})"),
            ("Unique Peptides", str(n_peptides)),
            ("Protein Groups", str(n_proteins)),
        ]
        if search_params:
            for key, val in search_params.items():
                rows.append((key, str(val)))

        self.add_section(
            "Identification Results",
            tables=[{"headers": ["Parameter", "Value"], "rows": rows}],
        )

    def add_quantification_section(
        self,
        n_features: int = 0,
        n_samples: int = 0,
        missing_rate: float = 0.0,
        normalization: str = "none",
        differential_features: int = 0,
    ) -> None:
        """Add quantification results section."""
        rows = [
            ("Quantified Features", str(n_features)),
            ("Samples", str(n_samples)),
            ("Missing Value Rate", f"{missing_rate:.1%}"),
            ("Normalization", normalization),
            ("Differential Features", str(differential_features)),
        ]
        self.add_section(
            "Quantification Results",
            tables=[{"headers": ["Parameter", "Value"], "rows": rows}],
        )

    def add_figure_from_matplotlib(self, fig: Any, title: str = "Figure") -> None:
        """Add a matplotlib figure as base64-encoded image."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        self._order_counter += 1
        self.sections.append(ReportSection(
            title=title,
            figures=[img_base64],
            order=self._order_counter,
        ))

    def _generate_css(self) -> str:
        """Generate CSS styles for the report."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }
        .report-header {
            border-bottom: 3px solid #2c5282;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }
        .report-header h1 {
            color: #2c5282;
            margin-bottom: 5px;
        }
        .report-header .meta {
            color: #666;
            font-size: 0.9em;
        }
        .section {
            margin-bottom: 30px;
            page-break-inside: avoid;
        }
        .section h2 {
            color: #2c5282;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #e2e8f0;
            padding: 8px 12px;
            text-align: left;
        }
        th {
            background-color: #edf2f7;
            font-weight: 600;
            color: #2d3748;
        }
        tr:nth-child(even) {
            background-color: #f7fafc;
        }
        .figure-container {
            text-align: center;
            margin: 15px 0;
        }
        .figure-container img {
            max-width: 100%;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
        }
        .status-good {
            color: #276749;
            background-color: #c6f6d5;
            padding: 8px 12px;
            border-radius: 4px;
        }
        .status-warn {
            color: #975a16;
            background-color: #fefcbf;
            padding: 8px 12px;
            border-radius: 4px;
        }
        .status-fail {
            color: #9b2c2c;
            background-color: #fed7d7;
            padding: 8px 12px;
            border-radius: 4px;
        }
        .toc {
            background-color: #f7fafc;
            padding: 15px 25px;
            border-radius: 4px;
            margin-bottom: 30px;
        }
        .toc h3 { margin-top: 0; }
        .toc ul { list-style-type: none; padding-left: 10px; }
        .toc a { text-decoration: none; color: #2c5282; }
        .toc a:hover { text-decoration: underline; }
        @media print {
            body { max-width: 100%; }
            .section { page-break-inside: avoid; }
        }
        """

    def _render_table(self, table_data: Dict[str, Any]) -> str:
        """Render a table dict to HTML."""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        html = "<table>\n<thead><tr>"
        for h in headers:
            html += f"<th>{h}</th>"
        html += "</tr></thead>\n<tbody>\n"
        for row in rows:
            html += "<tr>"
            for cell in row:
                html += f"<td>{cell}</td>"
            html += "</tr>\n"
        html += "</tbody>\n</table>"
        return html

    def _render_section(self, section: ReportSection) -> str:
        """Render a section to HTML."""
        anchor = section.title.lower().replace(" ", "-").replace("/", "-")
        html = f'<div class="section" id="{anchor}">\n'
        html += f"<h2>{section.title}</h2>\n"

        if section.content:
            html += section.content + "\n"

        for table in section.tables:
            html += self._render_table(table) + "\n"

        for fig_b64 in section.figures:
            html += '<div class="figure-container">'
            html += f'<img src="data:image/png;base64,{fig_b64}" />'
            html += "</div>\n"

        html += "</div>\n"
        return html

    def build_html(self) -> str:
        """Build the complete HTML report."""
        sorted_sections = sorted(self.sections, key=lambda s: s.order)

        html = "<!DOCTYPE html>\n<html>\n<head>\n"
        html += f"<title>{self.config.title}</title>\n"
        html += f"<style>{self._generate_css()}</style>\n"
        html += "</head>\n<body>\n"

        # Header
        html += '<div class="report-header">\n'
        html += f"<h1>{self.config.title}</h1>\n"
        meta_parts = []
        if self.config.author:
            meta_parts.append(f"Author: {self.config.author}")
        meta_parts.append(f"Date: {self.config.date}")
        meta_parts.append("Generated by MassKit")
        html += f'<p class="meta">{" | ".join(meta_parts)}</p>\n'
        html += "</div>\n"

        # Table of contents
        if len(sorted_sections) > 1:
            html += '<div class="toc">\n<h3>Contents</h3>\n<ul>\n'
            for section in sorted_sections:
                anchor = section.title.lower().replace(" ", "-").replace("/", "-")
                html += f'<li><a href="#{anchor}">{section.title}</a></li>\n'
            html += "</ul>\n</div>\n"

        # Sections
        for section in sorted_sections:
            html += self._render_section(section)

        # Footer
        html += '<hr><p style="color: #999; font-size: 0.8em; text-align: center;">'
        html += f"Report generated by MassKit v1.0.0 on {self.config.date}</p>\n"
        html += "</body>\n</html>"

        return html

    def save_html(self, filepath: str) -> None:
        """Save report as HTML file."""
        logger.info("Generating HTML report: %s", filepath)
        html = self.build_html()
        Path(filepath).write_text(html, encoding="utf-8")
        logger.info("HTML report saved (%d sections, %d bytes)", len(self.sections), len(html))

    def save_pdf(self, filepath: str) -> None:
        """
        Save report as PDF file.

        Requires weasyprint or pdfkit to be installed.
        Falls back to HTML if PDF generation fails.
        """
        html = self.build_html()

        try:
            from weasyprint import HTML
            HTML(string=html).write_pdf(filepath)
            return
        except ImportError:
            pass

        try:
            import pdfkit
            pdfkit.from_string(html, filepath)
            return
        except ImportError:
            pass

        # Fallback: save as HTML with .pdf note
        html_path = filepath.replace(".pdf", ".html")
        Path(html_path).write_text(html, encoding="utf-8")
        raise ImportError(
            f"PDF generation requires 'weasyprint' or 'pdfkit'. "
            f"HTML saved to {html_path} instead."
        )


def generate_analysis_report(
    config: Optional[ReportConfig] = None,
    experiment_summary: Optional[Dict[str, Any]] = None,
    qc_metrics: Optional[Any] = None,
    pca_result: Optional[Any] = None,
    anova_results: Optional[List[Any]] = None,
    identification_summary: Optional[Dict[str, Any]] = None,
    quantification_summary: Optional[Dict[str, Any]] = None,
    output_path: str = "report.html",
) -> str:
    """
    Generate a complete analysis report.

    Convenience function that builds a report from common analysis outputs.

    Args:
        config: Report configuration
        experiment_summary: Dict with keys like 'n_spectra', 'n_ms1', etc.
        qc_metrics: QCMetrics object
        pca_result: PCAResult object
        anova_results: List of ANOVAResult objects
        identification_summary: Dict with 'n_psms', 'n_peptides', 'n_proteins'
        quantification_summary: Dict with 'n_features', 'n_samples', etc.
        output_path: Output file path (.html or .pdf)

    Returns:
        Path to generated report
    """
    builder = ReportBuilder(config)

    if experiment_summary:
        builder.add_summary(**experiment_summary)

    if qc_metrics is not None:
        builder.add_qc_section(qc_metrics)

    if pca_result is not None or anova_results is not None:
        builder.add_statistics_section(pca_result=pca_result, anova_results=anova_results)

    if identification_summary:
        builder.add_identification_section(**identification_summary)

    if quantification_summary:
        builder.add_quantification_section(**quantification_summary)

    if output_path.endswith(".pdf"):
        try:
            builder.save_pdf(output_path)
        except ImportError:
            output_path = output_path.replace(".pdf", ".html")
            builder.save_html(output_path)
    else:
        builder.save_html(output_path)

    return output_path
