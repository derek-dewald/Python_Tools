import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler



def apply_scaling(X_train, X_test, scaler=None):
    """
    Applies optional scaling to training and test datasets.
    
    Parameters:
        X_train (np.array or pd.DataFrame): Training features.
        X_test (np.array or pd.DataFrame): Test features.
        scaler (str or None): Type of scaling to apply. 
                              Options: 'standard', 'normalization', or None.
    
    Returns:
        Scaled X_train and X_test.
    """
    scalers = {
        'standard': StandardScaler(),
        'normal': MinMaxScaler()
    }

    if scaler in scalers:
        scaler_instance = scalers[scaler]
        X_train = scaler_instance.fit_transform(X_train)
        X_test = scaler_instance.transform(X_test)

    return X_train, X_test
