import pandas as pd
import numpy as np
from scipy.stats import norm


def SampleDataFrame(df, 
                    conf=.95, 
                    me=0.05,
                    mv=0.5):
    """
    Returns a random sample from a DataFrame based on confidence level and margin of error.

    Parameters:
        df (pd.DataFrame): The dataset to sample from.
        conf(float): Desired Confidence Percentage Level (e.g., 90, 95, 99).
        me (float): Margin of Error, (default is 5%).
        mv (float): Maximum Variability (Expected Level of Default)

    Returns:
        pd.DataFrame: A random sample of the required size.
    """
    
    if not 0 <= mv <= 1:
        raise ValueError("mv (failure rate) must be between 0 and 1.")

    N = len(df)
    if N == 0:
        raise ValueError("DataFrame is empty")

    # Calculate the Z-score based on the confidence level
    z = norm.ppf(1 - (1 - conf) / 2)
    print(f"Z-score: {z}")  # Debug Z-score

    # Calculate the initial sample size (without finite population correction)
    n0 = (z**2 * mv * (1 - mv)) / (me**2)
    print(f"Initial sample size (n0): {n0}")  # Debug n0

    # Apply finite population correction if the population is smaller than 100,000
    if N >= 10000:  # For large populations, skip the correction
        n = int(n0)
    else:
        n = int((n0 * N) / (n0 + N - 1))

    print(f"Sample size with FPC: {n}")  # Debug final sample size

    return df.sample(n=n, random_state=42)