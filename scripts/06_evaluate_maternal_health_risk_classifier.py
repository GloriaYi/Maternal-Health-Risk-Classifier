# evaluate_maternal_health_risk_classifier.py
# author: Gloria Yi
# date: 2025-11-28
# code reference: adapted from https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/evaluate_breast_cancer_predictor.py

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import click
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
from sklearn.metrics import fbeta_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.calc_metrics import compute_classification_metrics
from src.roc_auc import compute_multiclass_auc


@click.command()
@click.option(
    "--processed-test-data",
    type=str,
    required=True,
    help="Path to processed maternal health test data (CSV).",
)
@click.option(
    "--columns-to-drop",
    type=str,
    default=None,
    help="Optional path to CSV listing columns to drop (column name: feats_to_drop).",
)
@click.option(
    "--pipeline-from",
    type=str,
    required=True,
    help="Path to file containing the fitted pipeline object (pickle).",
)
@click.option(
    "--plot-to",
    type=str,
    required=True,
    help="Path to directory where the result plots will be written to.",
)
@click.option(
    "--results-to",
    type=str,
    required=True,
    help="Path to directory where evaluation results will be written.",
)
@click.option(
    "--seed",
    type=int,
    default=123,
    show_default=True,
    help="Random seed.",
)
def main(processed_test_data, columns_to_drop, pipeline_from, plot_to, results_to, seed):
    """
    Evaluate the maternal health risk classification model on the processed
    test dataset and save evaluation metrics, confusion matrix, and ROC curves.
    This script loads a fitted pipeline, optionally removes specified features,
    computes multiple performance metrics, and generates diagnostic plots.

    Parameters
    ----------
    processed_test_data : str
        Path to the processed maternal health test data CSV file.
        The dataset must contain the target column ``RiskLevel``.

    columns_to_drop : str or None
        Optional path to a CSV file that lists columns to drop from the test
        dataset. This CSV must contain a column named ``feats_to_drop``.
        If ``None`` (default), no columns are dropped.

    pipeline_from : str
        Path to a pickle file containing the fitted scikit-learn pipeline
        used for prediction. The pipeline must implement either
        ``decision_function`` or ``predict_proba`` for ROC computation.

    results_to : str
        Directory where all evaluation results will be saved. The function
        creates the directory if it does not already exist.

        Saved outputs include:
            - ``test_scores.csv``: accuracy, weighted recall, and weighted F2.
            - ``confusion_matrix.csv``: crosstab of true vs predicted labels.
            - ``confusion_matrix.png``: visual confusion matrix plot.
            - ``roc_curves.png``: One-vs-Rest ROC curve plot.
            - ``auc_scores.csv``: AUC values per class.

    seed : int, optional
        Random seed used for reproducibility (default: 123).

    Returns
    -------
    None
        The function is executed for its side effects: writing evaluation files
        and generating plots in the directory specified by ``results_to``.
    """
    np.random.seed(seed)
    set_config(transform_output="pandas")
    os.makedirs(results_to, exist_ok=True)

    test_df = pd.read_csv(processed_test_data)

    if columns_to_drop:
        to_drop = pd.read_csv(columns_to_drop)["feats_to_drop"].tolist()
        test_df = test_df.drop(columns=to_drop)

    with open(pipeline_from, "rb") as f:
        mh_fit = pickle.load(f)


    X_test = test_df.drop(columns=["RiskLevel"])
    y_test = test_df["RiskLevel"]

    y_pred = mh_fit.predict(X_test)

    # compute metrics(accuracy, weighted recall, weighted F2)
    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        beta=2,
        average="weighted"
    )

    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(
        os.path.join(results_to, "test_scores.csv"),
        index=False,
    )
    
    # confusion matrix
    confusion_matrix = pd.crosstab(
        y_test,
        y_pred,
        rownames=["true_risk_level"],
        colnames=["predicted_risk_level"],
    )
    confusion_matrix.to_csv(
        os.path.join(results_to, "confusion_matrix.csv")
    )
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )
    ax.set_title("Confusion Matrix – Maternal Health Risk Classifier")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_to, "confusion_matrix.png"), dpi=300)
    plt.close(fig)


    # model must have decision_function or predict_proba
    try:
        y_score = mh_fit.decision_function(X_test)
    except Exception:
        y_score = mh_fit.predict_proba(X_test)

    classes = mh_fit.classes_

    auc_dict = compute_multiclass_auc(
        y_true=y_test,
        y_score=y_score,
        classes=classes
    )

    # Save AUC table
    pd.DataFrame([auc_dict]).to_csv(
        os.path.join(results_to, "auc_scores.csv"),
        index=False
    )

    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(classes):
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(
            y_true=(y_test == cls).astype(int),
            y_pred=y_score[:, i],
            name=f"{cls} (AUC={auc_dict[cls]:.3f})",
            ax=ax
        )

    plt.title("One-vs-Rest ROC Curves – Maternal Health Risk Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "roc_curves.png"))
    plt.close()

if __name__ == "__main__":
    main()
