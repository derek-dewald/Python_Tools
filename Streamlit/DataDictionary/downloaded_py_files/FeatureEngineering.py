from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

def generate_polynomial_features(df, target_col='Target', degree=2, include_bias=False):
    """
    Generates polynomial features for all numerical columns except the target.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The name of the target column to exclude from transformation.
    degree (int): The polynomial degree (default = 2).
    include_bias (bool): Whether to include a bias column (default = False).

    Returns:
    pd.DataFrame: A new DataFrame with named polynomial features.
    """
    
    # Select numeric columns excluding the target variable
    numeric_features = df.drop(columns=[target_col]).select_dtypes(include=['number'])
    
    # Initialize PolynomialFeatures transformer
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    
    # Fit and transform the numerical features
    X_poly = poly.fit_transform(numeric_features)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(numeric_features.columns)
    
    # Create new DataFrame with transformed features
    df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
    
    # Add back the target column
    df_poly[target_col] = df[target_col]
    
    return df_poly
