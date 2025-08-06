from sklearn.metrics import roc_auc_score,mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from DFProcessing import ConvertDicttoDF

import numpy as np
import pandas as pd
import datetime
import time
import sys


def ClassificationMetrics(df,
                          prediction='PREDICTION',
                          actual='ACTUAL',
                          AUC_Score=0,
                          new_column_name='RESULT'):
    """
    Function to generate summary statitics related to a ML Model Run.
    Removed Classification of TP,FN, FP, TN from df as when running many models quantum of data is high ,also it doesn't make a ton of sense in Multivariante classifications.
    
    Parameters:
        df (pDataFrame): DataFrame with predictions and actual values.
        prediction (str): Name of prediction column.
        actual (str): Name of actual/true label column.
        AUC_Score (float): Optional AUC.
        new_column_name (str): Name of column to store TP/TN/FP/FN tags.

    Returns:
        metrics_summary_df (DataFrame)
    """

    classes = sorted(df[actual].unique())
    metrics_list = []
    df = df.copy()

    for cls in classes:
        # Create One-vs-Rest binary views
        is_actual = df[actual] == cls
        is_pred   = df[prediction] == cls

        # Aggregate metrics per class
        TP = ( is_pred &  is_actual).sum()
        FP = ( is_pred & ~is_actual).sum()
        FN = (~is_pred &  is_actual).sum()
        TN = (~is_pred & ~is_actual).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy  = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

        metrics_list.append({
            "Class": cls,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
            "AUC": AUC_Score
        })

    result_df = pd.DataFrame(metrics_list)
    
    result_df.loc['Total',:] = result_df.mean()
    result_df['Class'] = result_df['Class'].astype(str)
    result_df.loc['Total','Class'] = 'Macro'
    
    return result_df


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
    

def CalculateRegressionPerformance(df,
                                   p,
                                   y='y',
                                   y_pred='y_pred'):
    
    '''
    Function to Generate a series of Performance Metrics for Regression Models.

    Parameters:
        df (Dataframe)
        p (int): Predictors, Number of X Variables (Independent Variables in Model)
        y (float): Target Value, Actual Observed Value
        y_pred (float) : Y-Hat, Model Predicted Value
    '''

    y = df[y].to_numpy()
    y_pred = df[y_pred].to_numpy()
    
    dict_ = {}
    dict_['MAE'] = mean_absolute_error(y, y_pred)
    dict_['MSE'] = mean_squared_error(y, y_pred)
    dict_['RMSE'] = np.sqrt(dict_['MSE'])
    dict_['R2'] = r2_score(y, y_pred)
    dict_['MAPE'] = np.mean(np.abs((y - y_pred) / y)) * 100
    dict_['SMAPE'] = 100 * np.mean(2 * np.abs(y_pred - y) / (np.abs(y) + np.abs(y_pred)))

    # Adjusted R^2 (assuming p predictors, here p=2 as example)
    n = len(y)
    dict_['ADJ_R2'] = 1 - (1 - dict_['R2']) * (n - 1) / (n - p - 1)

    return pd.DataFrame(dict_.values(),index=dict_.keys(),columns=['VALUES']).T


def MLPipelineSample(df, 
                     scaler,
                     ml_model_type='regressor',
                     target_column='Target',
                     sample_override=0,
                     run_all_size_models=0,
                     test_size=0.2):
    """
    Function to run the relevant models in SKlearn against Data set. Primary input is ml_model_type, which determines which class of models to run. 
    Models are selected from SKlearn library based on model Type. 

    This function is really meant to be a first pass on a sample of to provide insights into which models are likely to perform well.

    Parameters:
        df (DataFrame): Input dataset in form of DataFrame, with Target column labeled Target as default (it can be specified to change).
        scaler (str): One of None, 'normal', or 'standard'.
        ml_model_type (str): 'classifier', 'regressor', 'cluster', 'transformer'
        target_column (str): Name of the target column.
        sample_override (int): Binary flag to determine whether the entirety of Dataset should be run. Default is to NOT override and run a sample as defined by test size.
        run_all_size_models: (int): Binary flag to determine whether all models of the seleect model type should be run. Default is only to run models of relevant size, however
        override provided, espcially in the case of Smaller models, or models with limit parameters, which might run quickly against all models. Change implemented for Iris Dataset
        which ran in 1 second for selected models and 3 seconds for ALL models, opposed to MNIST, which timed out on all models.
        test_size (float): Proportion of data used for testing.

    Returns:
        pd.DataFrame: Summary of model performance.
    """

    # Get model list (you must have this function already working)
    sklearn_models_df = SKLearnModelList()

    if run_all_size_models==0:
        if len(df)<5000*(1+test_size):
            model_list = sklearn_models_df[(sklearn_models_df['Dataset Size'].str.contains('small',case=False))&(sklearn_models_df['Estimator Type'].str.contains(ml_model_type,case=False))]
        elif len(df)<100000*(1+test_size):
            model_list = sklearn_models_df[(sklearn_models_df['Dataset Size'].str.contains('medium',case=False))&(sklearn_models_df['Estimator Type'].str.contains(ml_model_type,case=False))]
        elif len(df)>100000*(1+test_size):
            model_list = sklearn_models_df[(sklearn_models_df['Dataset Size'].str.contains('large',case=False))&(sklearn_models_df['Estimator Type'].str.contains(ml_model_type,case=False))]
    else:
        model_list = sklearn_models_df[(sklearn_models_df['Estimator Type'].str.contains(ml_model_type,case=False))]

    if (len(df)>5000) & (sample_override==0):
        df = df.sample(frac=.15).copy()
        
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Apply scaler
    X_train, X_test, _ = apply_scaling(X_train, X_test, scaler=scaler)

    results_df = pd.DataFrame()

    for _, row in model_list.iterrows():
        name = row['Model Name']
        estimator_class = row['Estimator Class']
        print(f'Generating Predicition for {name}, {estimator_class}')
               
        try:
            start_time = time.time()
            model = estimator_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            binary_df = pd.concat([pd.DataFrame(y_pred,columns=['PREDICTION']),pd.DataFrame(y_test).rename(columns={target_column:'ACTUAL'}).reset_index(drop=True)],axis=1) 

            if ml_model_type == 'classifier':
                # Calculate AUC Score Outside of Classification as it requires information not being passed.
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    AUC_Score = roc_auc_score(y_test, y_proba, multi_class="ovr")
                    print(AUC_Score)
                else:
                    AUC_Score = 0            
                   
                model_perfom_stats =  ClassificationMetrics(binary_df,AUC_Score=AUC_Score)
                
            elif ml_model_type == 'regressor':
                model_perfom_stats = CalculateRegressionPerformance(binary_df,p=len(X.columns),y='ACTUAL',y_pred='PREDICTION')
            
            else:
                print('Need to develop Performance Measurement Metrics')
                pass
            
            model_perfom_stats['Model'] = name
            model_perfom_stats['Scaler'] = scaler
            model_perfom_stats['Time'] = time.time() - start_time
            results_df = pd.concat([results_df,model_perfom_stats])
            print(f'{name} Successfully Completed in {time.time() - start_time:.2f} seconds.')

        except Exception as e:
            print(f"Model Generation and Results Failed: {e}\n")
    return results_df



# def ClassificationMetrics(df,
#                           prediction='PREDICTION',
#                           actual='ACTUAL',
#                           AUC_Score=0,
#                           new_column_name='RESULT'):
#     '''
#     Function
        
    
#     Parameters
        
#     Returns
            
#     '''
#     results_dict = {}
    
#     condition = [(df[prediction]==1)&(df[actual]==df[prediction]),
#                  (df[prediction]==0)&(df[actual]==df[prediction]),
#                  (df[prediction]==1)&(df[actual]!=df[prediction]),
#                  (df[prediction]==0)&(df[actual]!=df[prediction])]
    
#     values = ['True Positives','True Negatives','False Positives (I)','False Negatives (II)']
    
#     df[new_column_name] = np.select(condition,values,default='Unknown')
    
#     results_dict['True Positives'] =       len(df[df['RESULT']=='True Positives'])
#     results_dict['True Negatives'] =       len(df[df['RESULT']=='True Negatives'])
#     results_dict['False Positives (I)'] =  len(df[df['RESULT']=='False Positives (I)'])
#     results_dict['False Negatives (II)'] = len(df[df['RESULT']=='False Negatives (II)'])
    
    
#     results_dict['Total Records'] = len(df)
#     results_dict['Correct Predictions'] =   len(df[df['RESULT'].isin(['True Negatives','True Positives'])])
#     results_dict['Incorrect Predictions'] = len(df[df['RESULT'].isin(['False Negatives (II)','False Positives (I)'])])
#     results_dict['Actual Positives'] = len(df[df['RESULT'].isin(['False Negatives (II)','True Positives'])])
#     results_dict['Actual Negatives'] = len(df[df['RESULT'].isin(['False Positives (I)','True Negatives'])])
    
#     try:
#         results_dict['Recall'] = results_dict['True Positives']/(results_dict['True Positives']+results_dict['False Negatives (II)'])     # How Many Positives were Actually Predicted
#         results_dict['Precision']    = results_dict['True Positives']/(results_dict['True Positives']+results_dict['False Positives (I)']) # How Many Predictions were Correct
#         results_dict['F1'] = 2*(results_dict['Precision']*results_dict['Recall'])/(results_dict['Precision']+results_dict['Recall'])  # Harmonic Mean 
        
#     except:
#         results_dict['Recall'] = 0
#         results_dict['Precision'] = 0
#         results_dict['F1'] = 0
        
#     results_dict['Accuracy']  = results_dict['Correct Predictions']/results_dict['Total Records']     # How Correct your Model was overall
#     results_dict['AUC'] = AUC_Score

#     return df,pd.DataFrame([results_dict.values()],columns=results_dict.keys())


# def MLPipeline(df, 
#                project_name,
#                scaler,
#                ml_model_type='regressor',
#                target_column='Target',
#                test_size=0.2):
#     """
#     Runs multiple scikit-learn estimators with MLflow tracking.

#     Args:
#         df (DataFrame): Input dataset.
#         project_name (str): MLflow experiment name.
#         scaler (str): One of None, 'normal', or 'standard'.
#         ml_model_type (str): 'classifier', 'regressor', 'cluster', 'transformer'
#         target_column (str): Name of the target column.
#         test_size (float): Proportion of data used for testing.

#     Returns:
#         pd.DataFrame: Summary of model performance.
#     """

#     # Set MLflow experiment
#     #mlflow.set_experiment(project_name)

#     # Prepare data
#     X = df.drop(columns=[target_column])
#     y = df[target_column]

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

#     # Apply scaler
#     X_train, X_test, _ = apply_scaling(X_train, X_test, scaler=scaler)

#     # Get model list (you must have this function already working)
#     sklearn_models_df = SKLearnModelList()

#     if len(X_train)<5000:
#         model_list = sklearn_models_df[(sklearn_models_df['Dataset Size'].str.contains('small',case=False))&(sklearn_models_df['Estimator Type'].str.contains(ml_model_type,case=False))]
#     elif len(X_train)<100000:
#         model_list = sklearn_models_df[(sklearn_models_df['Dataset Size'].str.contains('medium',case=False))&(sklearn_models_df['Estimator Type'].str.contains(ml_model_type,case=False))]
#     elif len(X_train)>100000:
#         model_list = sklearn_models_df[(sklearn_models_df['Dataset Size'].str.contains('large',case=False))&(sklearn_models_df['Estimator Type'].str.contains(ml_model_type,case=False))]

#     results = []

#     for _, row in model_list.iterrows():
#         name = row['Model Name']
#         estimator_class = row['Estimator Class']
#         print(f'Generating Predicition for {name}, started processing {datetime.datetime.now()}')
#         try:
#             start_time = time.time()
#             model = estimator_class()

#             #with mlflow.start_run(run_name=name):
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             if ml_model_type == "classifier":
#                 metric = accuracy_score(y_test, y_pred)
#                 mlflow.log_metric("Accuracy", metric)
#             else:
#                 metric = mean_squared_error(y_test, y_pred) ** 0.5
#                 mlflow.log_metric("RMSE", metric)

#                 #mlflow.sklearn.log_model(model, name)
#                 #mlflow.log_param("Model", name)
#                 #mlflow.log_param("Training Time", round(time.time() - start_time, 2))

#             results.append({
#                 "Model": name,
#                 "Metric": metric,
#                 "Time (s)": round(time.time() - start_time, 2)
#                 })

#         except Exception as e:
#             print(f"{name} failed: {str(e)}")

#     return pd.DataFrame(results)


# results_df = MLPipeline(df,
#                         project_name='MNIST_ML_Comparison',
#                         scaler='normal',
#                         ml_model_type='classifier',
#                         target_column='Target',
#                         test_size=0.2)

# results_df