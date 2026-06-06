"""Reporting: HTML/PDF race reports, model cards and living documentation."""
from .report import build_race_report
from .docs import (
    data_dictionary_markdown, feature_glossary_markdown, FEATURE_GLOSSARY,
)
__all__ = [
    "build_race_report", "data_dictionary_markdown",
    "feature_glossary_markdown", "FEATURE_GLOSSARY",
]
