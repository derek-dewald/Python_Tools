from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import all_estimators

import numpy as np
import pandas as pd

def ClassificationMetrics(df,
                          prediction='PREDICTION',
                          actual='ACTUAL',
                          new_column_name='RESULT'):
    '''
    Function
        
    
    Parameters
        
    Returns
            
    '''
    results_dict = {}
    
    condition = [(df[prediction]==1)&(df[actual]==df[prediction]),
                 (df[prediction]==0)&(df[actual]==df[prediction]),
                 (df[prediction]==1)&(df[actual]!=df[prediction]),
                 (df[prediction]==0)&(df[actual]!=df[prediction])]
    
    values = ['True Positives','True Negatives','False Positives (I)','False Negatives (II)']
    
    df[new_column_name] = np.select(condition,values)
    
    results_dict['True Positives'] =       len(df[df['RESULT']=='True Positives'])
    results_dict['True Negatives'] =       len(df[df['RESULT']=='True Negatives'])
    results_dict['False Positives (I)'] =  len(df[df['RESULT']=='False Positives (I)'])
    results_dict['False Negatives (II)'] = len(df[df['RESULT']=='False Negatives (II)'])
    
    
    results_dict['Total Records'] = len(df)
    results_dict['Correct Predictions'] =   len(df[df['RESULT'].isin(['True Negatives','True Positives'])])
    results_dict['Incorrect Predictions'] = len(df[df['RESULT'].isin(['False Negatives (II)','False Positives (I)'])])
    results_dict['Actual Positives'] = len(df[df['RESULT'].isin(['False Negatives (II)','True Positives'])])
    results_dict['Actual Negatives'] = len(df[df['RESULT'].isin(['False Positives (I)','True Negatives'])])
    
    try:
        results_dict['Recall'] = results_dict['True Positives']/(results_dict['True Positives']+results_dict['False Negatives (II)'])     # How Many Positives were Actually Predicted
        results_dict['Precision']    = results_dict['True Positives']/(results_dict['True Positives']+results_dict['False Positives (I)']) # How Many Predictions were Correct
        results_dict['F1'] = 2*(results_dict['Precision']*results_dict['Recall'])/(results_dict['Precision']+results_dict['Recall'])  # Harmonic Mean 
        
    except:
        results_dict['Recall'] = 0
        results_dict['Precision'] = 0
        results_dict['F1'] = 0
        
    results_dict['Accuracy']  = results_dict['Correct Predictions']/results_dict['Total Records']     # How Correct your Model was overall
    results_dict['AUC']       = roc_auc_score(df[actual],df[prediction])
    
    return df,pd.DataFrame([results_dict.values()],columns=results_dict.keys())


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

    return X_train, X_test, scaler_instance


def SKLearnModelList(regressor_type=None):
    '''
    Function to generate a list of all scikit-learn estimators with type classification.
    
    Imports List from all_estimators (sklearn.utils) 
    Imports D-Data Dashboard and Merges Column Dataset Size, to help filter for model processes.

    Parameter:
        Regressor_type: returns a list of relevant regressors if a specific regressor type is Picked. Types include: 
    
    Returns:
        df_final (pd.DataFrame): DataFrame of all estimators with class path, type labels, and module split.
    '''
    
    # 1. Get all estimators
    estimators = all_estimators(type_filter=None)
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(estimators, columns=['Model Name', 'Estimator Class'])

    # Merge In Dataset Size to prevent running of REALLY slow models.
    temp = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv')[['Word','Dataset Size']].rename(columns={'Word':'Model Name'})
    df = df.merge(temp,on='Model Name',how='left')
    
    # 3. Extract full class path
    df['Full Class Path'] = df['Estimator Class'].astype(str).str.replace("<class '", "", regex=False).str.replace("'>", "", regex=False)
    
    # 4. Extract class type via mixin inspection
    def get_estimator_types(cls):
        types = []
        try:
            if issubclass(cls, ClassifierMixin):
                types.append("classifier")
            if issubclass(cls, RegressorMixin):
                types.append("regressor")
            if issubclass(cls, ClusterMixin):
                types.append("cluster")
            if issubclass(cls, TransformerMixin):
                types.append("transformer")
        except:
            types.append("unknown")
        return ', '.join(types) if types else 'unknown'
    
    df['Estimator Type'] = df['Estimator Class'].apply(get_estimator_types)
    
    # 5. Split module path
    path_split = df['Full Class Path'].str.split('.', expand=True)
    path_split.columns = [f'Part_{i+1}' for i in range(path_split.shape[1])]
    
    # 6. Combine and return
    df_final = pd.concat([df, path_split], axis=1)

    if regressor_type==None:
        return df_final
    else:
        return df_final[df_final['Estimator Type'].str.contains(regressor_type)]