import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_feature_correlations(df, feature_cols):
    """
    Computes a numeric correlation matrix for selected features.
    
    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of str
    
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    corr_matrix = df[feature_cols].corr()
    return corr_matrix

def plot_correlation_heatmap(corr_matrix, save_path):
    """
    Plot and save a correlation heatmap given a correlation matrix.
    
    Parameters
    ----------
    corr_matrix : a correlation matrix
    save_path : str
        File path (including filename) where the heatmap image will be saved.

    Returns
    -------
    None
        This function is executed for its side effects: producing a file.
    """
    # raise error if corr_matrix is not a DataFrame
    if not isinstance(corr_matrix, pd.DataFrame):
        raise TypeError("corr_matrix must be a pandas DataFrame")
    
    # raise error if save_path is not valid
    if not save_path or not isinstance(save_path, (str, os.PathLike)):
        raise ValueError("save_path must be a valid file path string.")
    save_path = str(save_path)

    
    # Must end with a valid image extension
    if not (save_path.endswith(".png") or save_path.endswith(".jpg") or save_path.endswith(".jpeg")):
        raise ValueError("save_path must end with .png, .jpg, or .jpeg")
    
    # Directory must exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory does not exist: {save_dir}")
    
    # generate long-form correlation data for plotting
    corr_long = corr_matrix.reset_index().melt(id_vars="index")
    corr_long.columns = ["Feature 1", "Feature 2", "Correlation"]
    # visualize correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

