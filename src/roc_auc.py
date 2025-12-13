import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

def compute_multiclass_auc(y_true, y_score, classes):
    """
    Computes one-vs-rest ROC AUC for each class.
    
    Returns
    -------
    dict
        key = class name, value = AUC
    """
    
    y_true_bin = label_binarize(y_true, classes=classes)
    # check that y_true contains the right number of classes
    if y_score.shape != y_true_bin.shape:
        raise ValueError(
            f"y_score must have shape {y_true_bin.shape}, "
            f"but got {y_score.shape}"
        )
    # check that y_score contains numeric values
    if not np.issubdtype(y_score.dtype, np.number):
        raise ValueError("y_score must contain numeric values.")
    
    # check that all classes are present in y_true
    present_classes = set(y_true)
    missing = [cls for cls in classes if cls not in present_classes]
    if len(missing) > 0:
        raise ValueError(
            f"Classes {missing} do not appear in y_true, cannot compute AUC."
        )
    
    auc_dict = {}

    for i, cls in enumerate(classes):
        auc_dict[cls] = roc_auc_score(y_true_bin[:, i], y_score[:, i])

    return auc_dict
