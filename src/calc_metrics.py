from sklearn.metrics import accuracy_score, recall_score, fbeta_score

def compute_classification_metrics(y_true, y_pred, beta, average="weighted"):
    """
    Compute accuracy, weighted recall, and weighted F-beta score.

    Parameters
    ----------
    y_true : list or array-like
        True labels.
    y_pred : list or array-like
        Predicted labels.
    beta : float
        Beta value for F-beta score.
    average : str, default="weighted"
        Averaging method for multi-class classification.

    Returns
    -------
    dict
        Dictionary with keys: accuracy, recall_<avg>, f_beta_<beta>_<avg>
    """
    
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=average)
    fbeta = fbeta_score(y_true, y_pred, beta=beta, average=average)

    return {
        "accuracy": accuracy,
        f"recall_{average}": recall,
        f"f_beta_{beta}_{average}": fbeta
    }
