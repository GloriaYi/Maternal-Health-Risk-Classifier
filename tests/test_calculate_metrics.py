import pandas as pd
import pytest

from src.calc_metrics import compute_classification_metrics

# Case 1: simple expected case
def test_metrics_perfect_prediction():
    y_true = ["low", "mid", "high"]
    y_pred = ["low", "mid", "high"]

    result = compute_classification_metrics(y_true, y_pred, beta=2)

    assert result["accuracy"] == 1.0
    assert result["recall_weighted"] == 1.0
    assert result["f_beta_2_weighted"] == 1.0

# Case 2: Edge case
# All predictions are the same class, testing weighted metrics
def test_metrics_single_class():
    y_true = ["low", "low", "low"]
    y_pred = ["low", "low", "low"]

    result = compute_classification_metrics(y_true, y_pred, beta=1)

    assert result["accuracy"] == 1.0
    assert result["recall_weighted"] == 1.0
    assert result["f_beta_1_weighted"] == 1.0
    
# Case 3: Different beta values
def test_metrics_custom_beta():
    y_true = ["low", "mid", "high"]
    y_pred = ["low", "mid", "low"]

    f1 = compute_classification_metrics(y_true, y_pred, beta=1)["f_beta_1_weighted"]
    f2 = compute_classification_metrics(y_true, y_pred, beta=2)["f_beta_2_weighted"]

    assert f1 != f2 


# Case 4: error case
# true and predicted labels of different lengths
def test_metrics_length_mismatch():
    y_true = ["low", "mid"]
    y_pred = ["low", "mid", "high"]

    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, beta=2)
        
# Case 5: error case
# invalid beta value (e.g., negative)
def test_metrics_invalid_beta():
    y_true = ["low", "mid"]
    y_pred = ["low", "mid"]

    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, beta=-1)

# Case 6: Weighted vs macro averaging should differ for imbalanced classes
def test_metrics_weighted_vs_macro():
    # highly imbalanced dataset
    y_true = ["low", "low", "low", "mid", "high"]
    y_pred = ["low", "low", "mid", "mid", "low"] 

    result_weighted = compute_classification_metrics(
        y_true, y_pred, beta=1, average="weighted"
    )
    result_macro = compute_classification_metrics(
        y_true, y_pred, beta=1, average="macro"
    )

    # accuracy is always identical
    assert result_weighted["accuracy"] == result_macro["accuracy"]

    # recall and f1 scores should differ when distribution is imbalanced
    assert result_macro["recall_macro"] != result_weighted["recall_weighted"]
    assert result_macro["f_beta_1_macro"] != result_weighted["f_beta_1_weighted"]
