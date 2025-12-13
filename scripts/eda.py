# eda.py
# author: Gloria Yi
# date: 2025-12-05
# code reference: https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/eda.py

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import click
import altair as alt
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import io
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from src.plot_corr_heatmap import (
    compute_feature_correlations,
    plot_correlation_heatmap
)


FEATURE_COLS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]


@click.command()
@click.option(
    "--processed-training-data",
    type=str,
    required=True,
    help="Path to processed maternal health training data (CSV).",
)
@click.option(
    "--plot-to",
    type=str,
    required=True,
    help="Path to directory where the EDA plots will be written to.",
)
@click.option(
    "--tables-to",
    type=str,
    required=True,
    help="Directory to save EDA summary tables (describe, info)."
)
def main(processed_training_data, plot_to, tables_to):
    """
    Generate exploratory data analysis (EDA) outputs for the maternal health
    training dataset. The function reads the processed training CSV, computes
    summary statistics, creates diagnostic tables, and produces several plots
    describing correlations and feature distributions. It also performs 
    data-validation checks using deepchecks.

    Parameters
    ----------
    processed_training_data : str
        Path to the processed maternal health training data (CSV file).
        The dataset should contain the numeric feature columns specified in 
        `FEATURE_COLS` and the target column, ``RiskLevel``.

    plot_to : str
        Directory where all EDA plot files will be saved. The function creates
        the directory if it does not exist. Plots include:
        
        - ``correlation_heatmap.png``: Pearson correlation heatmap among numeric features.
        - ``feature_densities_by_risklevel.png``: KDE distributions of features by risk level.

    tables_to : str
        Directory where summary tables will be saved. The function creates it 
        if it does not exist. Tables include:
        
        - ``train_describe.csv``: Statistical summary using ``DataFrame.describe``.
        - ``train_info.txt``: Output of ``DataFrame.info`` including data types and non-null counts.

    Returns
    -------
    None
        This function is executed for its side effects: producing files
        (EDA tables, figures) and raising errors if deepchecks correlation 
        validations fail.

    Raises
    ------
    ValueError
        If the feature–label correlation exceeds the acceptable threshold 
        defined in ``FeatureLabelCorrelation`` checks.
    
    ValueError
        If the feature–feature correlation violates the maximum allowed
        number of correlated feature pairs as defined in 
        ``FeatureFeatureCorrelation`` checks.
    """
    # make sure the path exists, if not, create it
    os.makedirs(plot_to, exist_ok=True)
    os.makedirs(tables_to, exist_ok=True)
    # read in data
    train_df = pd.read_csv(processed_training_data)

    # Summary tables
    # 1. Describe table
    describe_df = train_df.describe(include="all").transpose()
    describe_df.to_csv(os.path.join(tables_to, "train_describe.csv"))

    # 2. Info table
    buf = io.StringIO()
    train_df.info(buf=buf)
    info_str = buf.getvalue()

    with open(os.path.join(tables_to, "train_info.txt"), "w") as f:
        f.write(info_str)

    # plot correlation heatmap with function
    corr_matrix = compute_feature_correlations(train_df, FEATURE_COLS)
    heatmap_path = os.path.join(plot_to, "correlation_heatmap.png")
    plot_correlation_heatmap(corr_matrix, heatmap_path)


    # visualize feature distributions by risk level
    num_features = len(FEATURE_COLS)
    cols = 3
    rows = math.ceil(num_features / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    
    for idx, feature in enumerate(FEATURE_COLS):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
    
        sns.kdeplot(
            data=train_df,
            x=feature,
            hue="RiskLevel",
            fill=True,
            alpha=0.4,
            ax=ax
        )
        ax.set_title(f"{feature} distribution by RiskLevel")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "feature_densities_by_risklevel.png"), dpi=300)
    plt.close()

    # Data validation - correlation checks

    mh_train_ds = Dataset(
        train_df,
        label="RiskLevel",
        cat_features=[]
    )

    # feature-label correlation check
    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    feat_lab_result = check_feat_lab_corr.run(dataset=mh_train_ds)

    # feature-feature correlation check
    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
        threshold=0.92,
        n_pairs=0
    )
    feat_feat_result = check_feat_feat_corr.run(dataset=mh_train_ds)
    if not feat_lab_result.passed_conditions():
        raise ValueError("Feature-Label correlation exceeds the maximum acceptable threshold.")
    
    if not feat_feat_result.passed_conditions():
        raise ValueError("Feature-feature correlation exceeds the maximum acceptable threshold.")


if __name__ == "__main__":
    main()
