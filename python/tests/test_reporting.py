"""Tests for the reporting module."""

import pytest
import tempfile
import os

from masskit.reporting import (
    ReportBuilder,
    ReportConfig,
    ReportSection,
    generate_analysis_report,
)


class TestReportConfig:
    def test_defaults(self):
        config = ReportConfig()
        assert config.title == "LC-MS Analysis Report"
        assert config.date != ""
        assert config.page_size == "A4"

    def test_custom(self):
        config = ReportConfig(title="My Report", author="Test User")
        assert config.title == "My Report"
        assert config.author == "Test User"


class TestReportBuilder:
    def test_empty_report(self):
        builder = ReportBuilder()
        html = builder.build_html()
        assert "<!DOCTYPE html>" in html
        assert "LC-MS Analysis Report" in html

    def test_add_section(self):
        builder = ReportBuilder()
        builder.add_section("Test Section", content="<p>Hello</p>")
        html = builder.build_html()
        assert "Test Section" in html
        assert "Hello" in html

    def test_add_summary(self):
        builder = ReportBuilder()
        builder.add_summary(n_spectra=1000, n_ms1=500, n_ms2=500)
        html = builder.build_html()
        assert "1000" in html
        assert "Experiment Summary" in html

    def test_add_identification_section(self):
        builder = ReportBuilder()
        builder.add_identification_section(n_psms=500, n_peptides=200, n_proteins=50)
        html = builder.build_html()
        assert "500" in html
        assert "Identification" in html

    def test_add_quantification_section(self):
        builder = ReportBuilder()
        builder.add_quantification_section(n_features=1000, n_samples=6)
        html = builder.build_html()
        assert "1000" in html

    def test_table_of_contents(self):
        builder = ReportBuilder()
        builder.add_section("Section A")
        builder.add_section("Section B")
        html = builder.build_html()
        assert "Contents" in html
        assert "Section A" in html
        assert "Section B" in html

    def test_save_html(self):
        builder = ReportBuilder()
        builder.add_section("Test", content="<p>test</p>")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            try:
                builder.save_html(f.name)
                assert os.path.exists(f.name)
                content = open(f.name).read()
                assert "<!DOCTYPE html>" in content
            finally:
                os.unlink(f.name)

    def test_css_generation(self):
        builder = ReportBuilder()
        css = builder._generate_css()
        assert "font-family" in css
        assert "status-good" in css


class TestGenerateReport:
    def test_basic_report(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            try:
                path = generate_analysis_report(
                    experiment_summary={"n_spectra": 100},
                    output_path=f.name,
                )
                assert os.path.exists(path)
            finally:
                os.unlink(f.name)
