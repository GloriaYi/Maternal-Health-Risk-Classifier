import os
import pandas as pd
import pytest
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from src.plot_corr_heatmap import (
    compute_feature_correlations,
    plot_correlation_heatmap
)

# ============================================================
# Tests for compute_feature_correlations
# ============================================================

# Case 1: Simple expected cases
def test_compute_corr_simple():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1, 2, 3],
        "c": [3, 2, 1],
    })

    corr = compute_feature_correlations(df, ["a", "b", "c"])

    assert isinstance(corr, pd.DataFrame)
    assert list(corr.columns) == ["a", "b", "c"]
    assert corr.loc["a", "b"] == 1.0
    assert corr.loc["a", "c"] == -1.0

# Case 1: Subset of columns
# passing in df and a subset of columns as feature_cols
def test_compute_corr_subset_cols():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
    })

    corr = compute_feature_correlations(df, ["a", "b"])
    assert list(corr.columns) == ["a", "b"]

# Case 2: Error cases
# requested column does not exist.
def test_compute_corr_missing_column():
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(KeyError):
        compute_feature_correlations(df, ["a", "missing"])

# Case 3: Error cases
# df contains non-numeric data for selected feature.
def test_compute_corr_non_numeric():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })

    with pytest.raises(ValueError):
        compute_feature_correlations(df, ["a", "b"])


# ============================================================
# Tests for plot_correlation_heatmap
# ============================================================

# Case 1: Simple expected case
def test_plot_heatmap_simple(tmp_path):
    corr = pd.DataFrame(
        [[1.0, 0.5],
         [0.5, 1.0]],
        columns=["a", "b"],
        index=["a", "b"],
    )

    save_path = tmp_path / "heatmap.png"
    plot_correlation_heatmap(corr, save_path)

    assert save_path.exists()
    assert os.path.getsize(save_path) > 0

# Case 2: Error cases
# corr_matrix is not a DataFrame
def test_plot_heatmap_invalid_corr_type():
    with pytest.raises(TypeError):
        plot_correlation_heatmap([1, 2, 3], "heatmap.png")

# Case 3: Error cases
# save_path is not valid
def test_plot_heatmap_invalid_path():
    """Error: invalid save_path should raise an exception."""
    corr = pd.DataFrame([[1, 0.1], [0.1, 1]])

    invalid_paths = [
        "", 
        None, 
        "/this/path/does/not/exist/heatmap.png",
        "heatmap.txt"
    ]

    for p in invalid_paths:
        with pytest.raises(Exception):
            plot_correlation_heatmap(corr, p)
