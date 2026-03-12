"""Unit tests for the exporter module."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.post_processing.exporter import export


SAMPLE = [
    {"text": "Hello World", "source": "img_001.png"},
    {"text": "Foo Bar", "source": "img_002.png"},
]


def test_export_json(tmp_path):
    out = tmp_path / "results.json"
    export(SAMPLE, fmt="json", out_path=str(out))
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    assert loaded[0]["text"] == "Hello World"


def test_export_csv(tmp_path):
    out = tmp_path / "results.csv"
    export(SAMPLE, fmt="csv", out_path=str(out))
    assert out.exists()
    df = pd.read_csv(out)
    assert "text" in df.columns
    assert len(df) == 2


def test_export_markdown(tmp_path):
    out = tmp_path / "results.md"
    export(SAMPLE, fmt="markdown", out_path=str(out))
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    # Should have a top-level heading and the text content
    assert "#" in content
    assert "Hello World" in content


def test_export_xlsx(tmp_path):
    out = tmp_path / "results.xlsx"
    export(SAMPLE, fmt="xlsx", out_path=str(out))
    assert out.exists()
    df = pd.read_excel(out, engine="openpyxl")
    assert "text" in df.columns
    assert len(df) == 2


def test_export_unknown_format_raises(tmp_path):
    out = tmp_path / "results.xyz"
    with pytest.raises(ValueError, match="(?i)unsupported|unknown|format"):
        export(SAMPLE, fmt="xyz", out_path=str(out))


def test_export_empty_list_json(tmp_path):
    out = tmp_path / "empty.json"
    export([], fmt="json", out_path=str(out))
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == []


def test_export_creates_parent_dir(tmp_path):
    nested = tmp_path / "sub" / "dir" / "out.json"
    export(SAMPLE, fmt="json", out_path=str(nested))
    assert nested.exists()
