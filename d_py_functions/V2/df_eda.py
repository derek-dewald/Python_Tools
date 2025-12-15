##File Description: This File is for Functions related to completing EDA. Including Setting up standard processes. Individual Image or DF manipulation which might be utilized in this function, but exist for broader purposes should be stored in more generic categories.

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

#from Visualization import Heatmap,plot_histograms,plot_scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import pandas as pd
import numpy as np
import math

from df_processing import TransposePivotTable
from feature_engineering import BracketColumn


def analyze_distribution(df):
    """
    Analyzes skewness, kurtosis, and visualizes the distribution of all numeric columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    None (displays plots and summary in an X by 3 format)
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    num_vars = len(numeric_cols)
    
    fig, axes = plt.subplots(num_vars, 3, figsize=(18, 6 * num_vars))
    
    if num_vars == 1:
        axes = [axes]  # Ensure iterable for single variable
    
    for i, col in enumerate(numeric_cols):
        target_data = df[col].dropna()
        
        # Compute skewness & kurtosis
        skewness = skew(target_data)
        kurt = kurtosis(target_data)
        
        if abs(skewness) > 1:
            skewness_comment = "Highly Skewed, Consider Transformation"
        elif skewness > 0:
            skewness_comment = "Right Skewed"
        elif skewness < 0:
            skewness_comment = "Left Skewed"
        else:
            skewness_comment = "Symmetric"

        # Correct Kurtosis Comments
        if kurt >= 2.75 and kurt <= 3.25:
            kurt_comment = "Mesokurtic (Normally Distributed), No Action Necessary."
        elif kurt > 3.25:
            kurt_comment = "Leptokurtic (High Kurtosis), More Extreme Values"
        elif kurt < 2.75:
            kurt_comment = "Platykurtic (Low Kurtosis), Less Extreme Values"
        
        summary_text = f"Skewness: {skewness:.2f}\n{skewness_comment}\nKurtosis: {kurt:.2f}\n{kurt_comment}"
        
        # Plot Histogram
        sns.histplot(target_data, kde=True, bins=30, color="blue", ax=axes[i][0])
        axes[i][0].set_title(f"Histogram of {col} (Skew: {skewness:.2f})")
        
        # Plot Boxplot
        sns.boxplot(x=target_data, color="red", ax=axes[i][1])
        axes[i][1].set_title(f"Boxplot of {col}")
        
        # Display Text
        axes[i][2].text(0.5, 0.5, summary_text, fontsize=12, va='center', ha='center', bbox=dict(facecolor='white', alpha=0.8))
        axes[i][2].axis("off")
        
    plt.tight_layout()
    plt.show()