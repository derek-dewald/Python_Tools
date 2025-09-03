from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import math

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
              'Severe Multicollinearity']
        
    vif_data['Assessement'] = np.select(condition,values,default="NA")
    vif_data['Action'] = np.where(vif_data['VIF']>5,'Review','Pass')
    
    
    return vif_data

def CleanStrtoNumber(df,
                     column_name,
                     new_column_name=""):
    
    '''
    
    
    '''
    
    if new_column_name=="":
        new_column_name = column_name
        
    df[new_column_name] = np.where((df[column_name]=="")|
                                   (df[column_name].isnull()),0,df[column_name])


def FirstObservanceFlag(df,column_name):
    df['FirstObs'] = (~df[column_name].duplicated()).astype(int)

def ConvertToBinary(df,column_name,return_value=1):
    
    if str(return_value)== '1':
        df[column_name] = np.where(df[column_name]>0,1,0)
    else:
        df[column_name] = np.where(df[column_name]>0,'Y','N')


def BinaryColumnCreator(df,
                        column_name,
                        new_column_name=None,
                        value=0,
                        calculation='=',
                        balance_column=None):
    '''
    Function to Create a Binary Flag. 
    
    Updated to remove Dictionary capabilities. Which seems overtly complex and unncessary. 23Jul25
    
    '''
    
    if not new_column_name:
        flag = f"{column_name.upper()}_FLAG"
        bal = f"{column_name.upper()}_BAL"
    else:
        flag = f"{new_column_name.upper()}_FLAG"
        bal = f"{new_column_name.upper()}_BAL"
        
    if calculation=='>':
        df[flag] = np.where(df[column_name]>value,1,0)
    elif calculation =='=':
        df[flag] = np.where(df[column_name]==value,1,0)
    elif calculation =='<':
        df[flag] = np.where(df[column_name]<value,1,0)
    elif calculation =='isin':
        df[flag] = np.where(df[column_name].isin(value),1,0)
    elif calculation =='contains':
        df[flag] = np.where(df[column_name].str.contains(value),1,0)
    if balance_column:
        df[bal] = np.where(df[flag]==1,df[balance_column],0)



def ComingDueCalculation(df,
                         days_column,
                         balance_column,
                         new_column_text,
                         condition_dict):
    '''
    
    '''
    
    lower_limit = 0
    
    for key,value in condition_dict.items():
        upper_limit = key
        value = f"{new_column_text}{value}"
        df[value] = np.where((df[days_column]>lower_limit)&
                             (df[days_column]<=upper_limit),df[balance_column],0)
        lower_limit= key
    
def WeightedAverageBalance(df,
                           balance_column,
                           days_column,
                           new_column_name):
    
    '''
    
    
    '''
    
    df[new_column_name] = df[balance_column].fillna(0)*df[days_column].fillna(0)


def BracketColumn(df,
                  column_name,
                  new_column_name,
                  bins=[0,10,20,30,40,50,60,70,80,90,100],
                  formating="",
                  assign_cat=0,
                  last_bin_lowest_value=1,
                  print_=0):
    
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
                value_list.append(f"{count+1})Equal to {formating}{value:,}")
            else:
                condition_list.append(df[column_name]<value)
                value_list.append(f"{count+1})Less than {formating}{value:,}")
        
        elif count < len(bins)-1:
            condition_list.append(df[column_name]<=value)
            value_list.append(f"{count+1})Between {formating}{bins[count-1]:,} and {formating}{bins[count]:,}")
        
        else:
            condition_list.append(df[column_name]<=value)
            value_list.append(f"{count+1})Between {formating}{bins[count-1]:,} and {formating}{bins[count]:,}")
            
            condition_list.append(df[column_name]>value)
            value_list.append(f"{count+1})Greater than {formating}{value:,}")
            
        df[new_column_name]=np.select(condition_list,value_list,'Problem')
        
        if assign_cat ==1:
            df[new_column_name] = pd.Categorical(df[new_column_name], categories=value_list)
    
    if print_==1:
        print(df[new_column_name].value_counts())

def BinaryComplexEquivlancey(df, col, col1, new_column_name):
    try:
        # Try numeric comparison
        df[new_column_name] = np.where(
            (df[col].isna() & df[col1].isna()) |
            ((df[col].fillna(0) == 0) & df[col1].isna()) |
            ((df[col1].fillna(0) == 0) & df[col].isna()) |
            (df[col].fillna(0) == df[col1].fillna(0)),
            1, 0
        )
    except Exception:
        # Fallback to string comparison
        df[new_column_name] = np.where(
            (df[col].isna() & df[col1].isna()) |
            ((df[col].fillna('') == '') & df[col1].isna()) |
            ((df[col1].fillna('') == '') & df[col].isna()) |
            (df[col].fillna('').astype(str).str.strip().str.lower() ==
             df[col1].fillna('').astype(str).str.strip().str.lower()),
            1, 0
        )
    return df

def DFCalculateColumnDifference(df, col1, col2, new_column_name):
    
    """
    Function which Calculates the Difference between 2 Columns in a DataFrame. Most importantly, it will not fail out
    When there is Text in the columns. 
    
    Important to note that Null Records are Interpretted as 0 in calculation.
    
    
    Parameters:
        df (Dataframe):
        col1 (str):
        col2 (str):
        new_column_name (str)
    
    Returns:
	Appends New Column Name into existing DF
    
    
    Date Created: August 18, 2025
    Date Last Modified:
    
    """
    # Convert to numeric, coercing errors to NaN
    col1_numeric = pd.to_numeric(df[col1], errors='coerce').fillna(0)
    col2_numeric = pd.to_numeric(df[col2], errors='coerce').fillna(0)
    
    # Perform subtraction
    df[new_column_name] = col1_numeric - col2_numeric



def CreateRandomDFColumn(df, new_column_name,value_list=None,value_dict=None):
    '''
    Function to Create a New Column in a Dataframe by randomly Selecting values from a list
    based on specified probabilities.
    
    If Value List is Populated, then it will be Utilized

    Parameters:
        df (DataFrame)
        new_column_name (str): Name of New Column
        value_list (list):  List Values to Randomly Assign
        value_dict (dict): Dictionary of key value pairs between desired Random Item and preferred Distribution
        
    Returns:
        Newly Created Column in Existing DataFrame
        
    Date Created: August 18, 2025
    Date Last Modified:
        
    
    '''
    
    if value_list:
        df[new_column_name] = np.random.choice(value_list, size=len(df))
    
    elif value_dict:
        value_list = list(value_dict.keys())
        probability = list(value_dict.values())
    
    try:
        df[new_column_name] = np.random.choice(value_list, size=len(df), p=probability)
    except:
        print("Could Not Complete")
