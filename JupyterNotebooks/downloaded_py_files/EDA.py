from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

def plot_correlation_matrix(df):
    """
    Generates and displays a correlation matrix heatmap for a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 8))  # Set figure size
    correlation_matrix = df.corr()  # Compute correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def plot_histograms(df, bins=30):
    """
    Plots histograms for each column in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    bins (int): Number of bins for the histograms (default=30).
    """
    num_columns = len(df.columns)
    num_rows = math.ceil(num_columns / 4)  # Determine rows for 4 columns
    
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows))
    axes = axes.flatten()  # Flatten in case of fewer than 4 columns

    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=bins, color="skyblue", edgecolor="black")
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_scatter_matrix(df, target_col='Target'):
    """
    Plots scatter plots of each feature against the target variable.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The name of the target variable.
    """
    num_features = df.drop(columns=[target_col]).shape[1]
    num_rows = (num_features // 4) + 1
    
    fig, axes = plt.subplots(num_rows, 4, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        if col != target_col:
            sns.scatterplot(x=df[col], y=df[target_col], ax=axes[i])
            axes[i].set_title(f"{col} vs. {target_col}")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def calculate_vif(df):
    """
    Computes Variance Inflation Factor (VIF) for each numerical feature.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing numerical predictors.
    
    Returns:
    pd.DataFrame: VIF values for each feature.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    condition = [vif_data['VIF']<=1,
                 vif_data['VIF']<=5,
                 vif_data['VIF']<=10,
                 vif_data['VIF']>10]
    
    values = ['No Multicollinearity',
              'Low to Moderate Multicollinearity',
              'High Multicollinearity',
              'Severe Multicollinearity'
             ]
        
    vif_data['Assessement'] = np.select(condition,values,default="NA")
    vif_data['Action'] = np.where(vif_data['VIF']>5,'Review','Pass')
    
    
    return vif_data

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