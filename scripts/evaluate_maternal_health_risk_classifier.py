# src/evaluate_maternal_health_classifier.py
# author: Gloria Yi
# date: 2025-11-28
# code reference: adapted from https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/evaluate_breast_cancer_predictor.py

import os
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
    "--results-to",
    type=str,
    required=True,
    help="Path to directory where evaluation results will be written.",
)
@click.option(
    "--seed",
    type=int,
    default=522,
    show_default=True,
    help="Random seed.",
)
def main(processed_test_data, columns_to_drop, pipeline_from, results_to, seed):
    """Evaluate the maternal health risk classifier on the test data and save metrics."""
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

    # accuracy
    accuracy = mh_fit.score(X_test, y_test)

    y_pred = mh_fit.predict(X_test)
    
    # calculate recall
    recall_weighted = recall_score(
    y_test,
    y_pred,
    average="weighted"
    )

    # compute F2 weighted
    f2_weighted = fbeta_score(
        y_test,
        y_pred,
        beta=2,
        average="weighted",
    )

    test_scores = pd.DataFrame(
        {
            "accuracy": [accuracy],
            "recall_weighted": [recall_weighted],
            "F2_weighted": [f2_weighted],
        }
    )
    test_scores.to_csv(
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
    fig.savefig(os.path.join(results_to, "confusion_matrix.png"), dpi=300)
    plt.close(fig)


    # model must have decision_function or predict_proba
    try:
        y_score = mh_fit.decision_function(X_test)
    except AttributeError:
        y_score = mh_fit.predict_proba(X_test)

    classes = mh_fit.classes_
    y_test_bin = label_binarize(y_test, classes=classes)

    # create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    auc_results = {}

    for i, class_name in enumerate(classes):
        auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
        auc_results[class_name] = auc
        
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i],
            y_score[:, i],
            name=f"{class_name} (AUC={auc:.3f})",
            ax=ax
        )

    plt.title("One-vs-Rest ROC Curves for Maternal Health Risk Classification")
    plt.tight_layout()
    plt.savefig(os.path.join(results_to, "roc_curves.png"))
    plt.close()

    # save AUCs to CSV
    auc_df = pd.DataFrame(auc_results, index=["AUC"])
    auc_df.to_csv(os.path.join(results_to, "auc_scores.csv"))


if __name__ == "__main__":
    main()
