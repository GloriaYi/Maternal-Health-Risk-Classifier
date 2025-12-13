# fit_maternal_health_risk_classifier.py
# Fits and tunes an SVC maternal health risk classifier using cross-validation
# Validates training data for anomalous correlations, 
# Saves both the trained pipeline and model-selection plot.

import click
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from scipy.stats import loguniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(training_data, preprocessor, pipeline_to, plot_to, seed):
    """
    Train and tune an SVC-based maternal health risk classification pipeline.

    This function loads the training dataset, constructs a preprocessing and SVC
    classification pipeline, and performs hyperparameter tuning using randomized
    search with 10-fold cross-validation. The search optimizes the weighted recall
    score across a range of 'C' and 'gamma' values sampled from log-uniform
    distributions. The best-performing model is saved as a serialized pipeline.

    Additionally, the function generates and saves a heatmap showing the top
    10 hyperparameter combinations ranked by cross-validated performance.

    Parameters
    ----------
    training_data : str
        Path to the CSV file containing the training dataset with features and
        the 'RiskLevel' target column.

    preprocessor : str
        (Unused in current implementation.) Path to a serialized preprocessing
        object. The function instead constructs a new StandardScaler-based
        preprocessing pipeline internally.

    pipeline_to : str
        Directory where the trained SVC pipeline will be saved as a pickle file.

    plot_to : str
        Directory where the hyperparameter tuning heatmap will be saved.

    seed : int
        Random seed for reproducibility of data-related operations.
    """
    np.random.seed(seed)
    train_df = pd.read_csv(training_data)
    preprocessor = pickle.load(open(preprocessor, 'rb'))
    svc = make_pipeline(preprocessor, SVC(random_state=seed))
    param_grid = {
    "svc__C": loguniform(1e-2, 1e3),
    "svc__gamma": loguniform(1e-4, 1e1)
    }

    random_search = RandomizedSearchCV(
        svc,                                    
        param_distributions=param_grid, 
        n_iter=100, 
        n_jobs=-1,
        return_train_score=True,
        cv=10,
        scoring='recall_weighted',
        random_state=123
        )

    maternal_risk_fit = random_search.fit(
        train_df.drop(columns=['RiskLevel']), 
        train_df["RiskLevel"]
        )

    with open(os.path.join(pipeline_to, "maternal_risk_classifier.pickle"), 'wb') as f:
        pickle.dump(random_search, f)

    result_grid = pd.DataFrame(random_search.cv_results_)
    result_grid = result_grid[
        [
            "mean_test_score",
            "param_svc__gamma",
            "param_svc__C",
            "mean_fit_time",
            "rank_test_score",
        ]
    ].set_index("rank_test_score").sort_index().T.iloc[:, :10]

    plt.figure(figsize=(12, 5))
    sns.heatmap(result_grid, 
                annot=True,           
                fmt='.3f',           
                cmap='viridis',        
                cbar_kws={'label': 'Value'},
                linewidths=0.5,
                linecolor='gray')

    plt.title('Top 10 SVC Hyperparameter Combinations', fontsize=14, pad=20)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Parameter/Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "svc_hyperparameter_tuning.png"), dpi=300)

if __name__ == '__main__':
    main()