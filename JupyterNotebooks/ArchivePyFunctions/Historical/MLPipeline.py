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

def GenerateSKLearnModelList(model_type_list=["classifier", "regressor", "cluster", "transformer",None],
                             export=0):
    
    '''
    Function to Generate a list of All ML Models contained with SKlearn and to Bring Visability into how they are 
    labelled, specifically to understand which models will be called when Classifier, Regressor, Cluster or Transformer
    is applied
    
    Parameters:
        model_type_list (List): List of values which can be input into all_estimators
        export: (int): Binary Choice, 1 Returns CSV, 0 Returns Nothing.
    
    Returns:
        Dataframe
    
    '''
    
    final_df = pd.DataFrame()
    
    for model_type in model_type_list:
        if model_type==None:
            temp_df = pd.DataFrame(all_estimators(type_filter=None), columns=['Model Name', 'Estimator Class'])
            temp_df['ModelType'] = "All"
            final_df = pd.concat([final_df,temp_df])
        else:
            temp_df = pd.DataFrame(all_estimators(type_filter=model_type), columns=['Model Name', 'Estimator Class'])
            temp_df['ModelType'] = model_type
            final_df = pd.concat([final_df,temp_df])
            
    final_df = final_df.drop_duplicates(['Model Name','Estimator Class']).reset_index(drop=True)
        
    # Extract full module path and class name
    final_df['Full Class Path'] = final_df['Estimator Class'].astype(str)
    
    # Create a unique identifier (Primary Key) using module path + model name
    final_df['Primary Key'] = final_df['Full Class Path'].str.replace("<class '", "").str.replace("'>", "")

    # Split the class path into separate columns
    split_columns = final_df['Primary Key'].str.split('.', expand=True)
    split_columns.columns = [f'Part_{i+1}' for i in range(split_columns.shape[1])]

    # Concatenate the original DataFrame with the split columns
    final_df = pd.concat([final_df, split_columns], axis=1).drop('Part_1', axis=1)

    if export == 0:
        return final_df
    else:
        final_df.to_csv('SKLearnModels.csv', index=False)
        return final_df
        