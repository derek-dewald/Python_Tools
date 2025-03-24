from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

def generate_polynomial_features(df, target_col='Target', degree=2, include_bias=False):
    """
    Generates polynomial features for all numerical columns except the target.

    Args:
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

def BinaryColumnCreator(df,
                        column_name,
                        new_column_name,
                        value,
                        calculation):
  
  '''
  
  
  
  '''
  
  if calculation=='>':
    df[new_column_name] = np.where(df[column_name]>value,1,0)
  elif calculation =='=':
    df[new_column_name] = np.where(df[column_name]==value,1,0)
  elif calculation =='<':
    df[new_column_name] = np.where(df[column_name]<value,1,0)
  elif calculation =='isin':
    df[new_column_name] = np.where(df[column_name].isin(value),1,0)
  elif calculation =='contains':
    df[new_column_name] = np.where(df[column_name].str.contains(value),1,0)
  elif calculation == 'dict':
    for key,value in value.items():
        try:
          df[new_column_name] = np.where((df[key]==value)&(df[new_column_name]==1),1,0)
        except:
          df[new_column_name] = np.where(df[key]==value,1,0)


def TextClean(df, 
              column_list, 
              clean_type='only_digits'):
    '''
    
    
    '''

    if clean_type == 'only_digits':
        for column in column_list:
            df[column] = df[column].astype(str).str.replace(r'[^\d.]', "", regex=True)
            df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

def FillFromAbove(df,
                 column_name,
                 new_column_name=''):
    
    '''
    Force Fill Information from above defined values when Blank. 

    Args:

    
    Returns:

     
    df = pd.DataFrame({'Test': [None, 'Value1', None, 'Value2', None, None, 'Value3', None]})
    FillFromAbove(df,'Test','Test1')
    
    
    '''
    if new_column_name=="":
        new_column_name = column_name
    
    df[new_column_name] = df[column_name].ffill()


def BracketColumn(df,
                   column_name,
                   new_column_name,
                   bins=[0,10,20,30,40,50,60,70,80,90,100],
                   formating="",
                   assign_cat=0,
                   last_bin_lowest_value=1,):
    
    '''
    Function Created to analyze the contents of a Column in a Dataframe and return a summarized view of the distribution.
    Function differs from ColumnPartioner, in that it does not look for uniformity, a higher level summary.
    
    Parameters:
        df {dataframe}
        
        column_name (str): Column Name of DF column to review
        
        new_column_name (str): Name of Column to be created in Dataframe as result of Function
        
        bins [list]: Number of Bins to include in Segmentation. Need atleast 2, no upper bound limit.
        
        formating (str): Value which can be appended for str output in return value text. $ Only Viable in current iteration 
        
        assign_cat (int): Binary Flag which allows user to save the column in Categorical Order to ease filtering.
        this does impact how groupby works and size of data, so proceed with caution.
    
        last_bin_lowst_value (int): Binary Flag, allows user to either Place a Floor on bottom limit based on Bin Value, or 
        search everything below bottom limit. EI, if you have 0 for deposit it would not include balances below 0 in the count if 1.
    
    Returns:
        Dataframe with New Column
    
    
    
    '''
    
    df[column_name] = df[column_name].fillna(0)
    
    condition_list = []
    value_list = []
    
    for count,value in enumerate(bins):
        if count == 0:
            if last_bin_lowest_value==1:
                condition_list.append(df[column_name]==value)
                value_list.append(f"Equal to {formating}{value}")
            else:
                condition_list.append(df[column_name]<=value)
                value_list.append(f"Less than or Equal to {formating}{value}")
        
        elif count < len(bins)-1:
            condition_list.append(df[column_name]<value)
            value_list.append(f"Between {formating}{bins[count-1]} and {formating}{bins[count]}")
        
        else:
            condition_list.append(df[column_name]<value)
            value_list.append(f"Between {formating}{bins[count-1]} and {formating}{bins[count]}")
            
            condition_list.append(df[column_name]>value)
            value_list.append(f"Greater than {formating}{value}")
            
        df[new_column_name]=np.select(condition_list,value_list,'Problem')
        
        if assign_cat ==1:
            df[new_column_name] = pd.Categorical(df[new_column_name], categories=value_list)
    
    print(df[new_column_name].value_counts())
    
def CleanStrtoNumber(df,
                     column_name,
                     new_column_name=""):
    
    '''
    
    
    '''
    
    if new_column_name=="":
        new_column_name = column_name
        
    df[new_column_name] = np.where((df[column_name]=="")|
                                   (df[column_name].isnull()),0,df[column_name])

def VarianceInflationFactor(df):
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
