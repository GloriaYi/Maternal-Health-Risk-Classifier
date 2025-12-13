import pytest
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

from src.roc_auc import compute_multiclass_auc


# Case 1: Simple expected case
def test_auc_simple_case():
    y_true = ["low", "mid", "high"]
    classes = ["low", "mid", "high"]

    # perfect predictions means AUC = 1 for all classes
    y_score = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1], 
    ])

    auc_dict = compute_multiclass_auc(y_true, y_score, classes)

    assert isinstance(auc_dict, dict)
    assert len(auc_dict) == 3
    assert all(abs(v - 1.0) < 1e-8 for v in auc_dict.values())



# Case 2: computed for different random scores
# AUC should not be exactly 1 or 0 but between 0 and 1
def test_auc_random_scores():
    y_true = ["low", "low", "mid", "high", "high"]
    classes = ["low", "mid", "high"]

    # generate random prediction scores
    np.random.seed(123)
    y_score = np.random.rand(len(y_true), 3)

    auc_dict = compute_multiclass_auc(y_true, y_score, classes)

    assert isinstance(auc_dict, dict)
    assert len(auc_dict) == 3
    # AUC should be between 0 and 1
    for auc in auc_dict.values():
        assert 0 <= auc <= 1


# Case 3: only one class present in y_true
# Some classes may have undefined ROC and AUC since either FPR or TPR cannot be computed
# Should raise ValueError to avoid misleading output
def test_auc_missing_class_error():
    y_true = ["low", "low", "low"]  # only one class exists
    classes = ["low", "mid", "high"]
    y_score = np.array([
        [0.9, 0.05, 0.05],
        [0.8, 0.15, 0.05],
        [0.85, 0.1, 0.05],
    ])

    with pytest.raises(ValueError):
        compute_multiclass_auc(y_true, y_score, classes)


# Case 4: Error – y_score wrong shape
def test_auc_wrong_score_shape():
    y_true = ["low", "mid", "high"]
    classes = ["low", "mid", "high"]

    # y_score shoule be (n_samples, n_classes), we have 3 classes but only provide 2 columns
    y_score = np.array([
        [0.2, 0.8],
        [0.6, 0.4],
        [0.1, 0.9],
    ])

    with pytest.raises(ValueError):
        compute_multiclass_auc(y_true, y_score, classes)


# Case 5: Error – y_score not numeric
def test_auc_non_numeric_scores():
    y_true = ["low", "mid", "high"]
    classes = ["low", "mid", "high"]

    y_score = np.array([
        ["a", "b", "c"],
        ["d", "e", "f"],
        ["g", "h", "i"],
    ])

    with pytest.raises(ValueError):
        compute_multiclass_auc(y_true, y_score, classes)
