# File Description: Related to teh Creation of Enhancements to Dataframes which result in the creation of New DataFrames, Columns, Rows, Etc. 

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
            value_list.append(f"{count+2})Greater than {formating}{value:,}")
            
        df[new_column_name]=np.select(condition_list,value_list,'Problem')
        
        if assign_cat ==1:
            df[new_column_name] = pd.Categorical(df[new_column_name], categories=value_list)
    
    print(df[new_column_name].value_counts())

    
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

def BalanceTargetDistribution(df,
                              Target,
                              desired_percentage,
                              TargetValue=1):
    
    '''
    Function to support reduction of Dataset based on desire to Target weight the percentage of observations.
    
    Parameters:
        df (DataFrame)
        Target (str): Column Name of Target
        desired_percentage (float): Number between 0 - 1, which is desired weighting of Target between 1 and 0.
        Target Value (int): Value used to balance, perferably Int, can be str.
        
        
    Returns:
        Dataset
    
    '''
    df = df.copy()

    df0 = df[df[Target]==TargetValue].copy()
    df1 = df[df[Target]!=TargetValue].copy()
    
    if (desired_percentage>0)&(desired_percentage<1):
        req_columns = int(round(len(df0)/desired_percentage,0))
        
        return pd.concat([df0,df1.sample(req_columns)]).sample(frac=1).reset_index(drop=True)
        
    else:
        print('Desired Percentage Outside of Allowable Range (0-1), Please Select a New Value')


def CompareTextColumns(df,
                       col1,
                       col2,
                       new_column_name=''):
    '''
    Function to Create a Binary Flag when to Dataframe Columns are Equal.


    Parameters
        df (dataframe)
        col1 (str): Name of Column 1
        col2 (str): Name of Column 2 (to be compared with column1)
        new_column_name (str): Name of New Column, if left blank then it will be Default to BINARY_MATCH_
    
    
    '''
    
    if new_column_name=='':
        new_column_name = f'BINARY_MATCH_{col1}_{col2}'
    
    df[new_column_name] = np.where(df[col1].str.strip().str.lower()==df[col2].str.strip().str.lower(),1,0)
    


def CategorizeBinaryChange(df,
                           change_column='VARIANCE'):
    '''
    Function to Simply Apply a Condition to generate whether a particular column, which is meant to be a Change over a 
    time series dataset, has Increased, Decreased or Stayed the Same. Can be stand alone, created for use in ColumnElementalChange
    
    Args:
        change_column (str): Name of Column Created. 
        
    Returns:
        dataframe, with change_column added
        
    '''
    
    condition = [
        df[change_column]>0,
        df[change_column]<0,
        df[change_column]==0
    ]
        
    value = ['Records Increasing',
             'Records Decreasing',
             'Records Not Changing']
        
    df['CHANGE_CLASSIFICATION'] = np.select(condition,value,'Null Value Present')
    
    try:
        # Supplement Change to increase visability of granularity so we can understand true variance, in addition to Add/Losses
        df['CHANGE_CLASSIFICATION'] = np.where(df['_merge']=='right_only','Records Lost',df['CHANGE_CLASSIFICATION'])
        df['CHANGE_CLASSIFICATION'] = np.where(df['_merge']=='left_only','Records Added',df['CHANGE_CLASSIFICATION'])
        
    except:
        pass
    
def ConvertToBinary(df, column_name, return_value=1):
    '''
    Function to convert a column into a Binary Variable. Specifically useful when aggregating counts, and in place you'd like to return
    a simple binary indicator of the possession of a particular attribute.

    Parameters
    df (DataFrame)
    column_name (str): The column to be converted.
    return_value  (int): Binary Flag, 1 for Int, 'Y' for text.

    Returns
        Modifies the DataFrame in place.
    '''
    if str(return_value) == '1':
        df[column_name] = np.where(df[column_name] > 0, 1, 0)
    else:
        df[column_name] = np.where(df[column_name] > 0, 'Y', 'N')

def FirstObservanceFlag(df, column_name):
    '''
    Function to creates a binary flag column indicating the first observation of a Value in a Column. Reteurns a 0 for all subsequent values

    Parameters
    df (DataFrame)
    column_name (str): The column name where values will be counted

    Returns
        Modifies the DataFrame in place.
    '''
    df['FirstObs'] = (~df[column_name].duplicated()).astype(int)


def CreateRandomDFColumn(df, value_list, new_column_name):
    '''
    Function to Create a New Column in a Dataframe by randomly Selecting values from a list

    Parameters:
    df (Dataframe)
    value_list (list): List of values to be randomly choosen from
    new_column_name: Name of New Column
    
    Returns:
        new dataframe column
    
    '''
    
    df[new_column_name] = np.random.choice(value_list, size=len(df))


def BinaryComplexEquivlancey(df, col, col1, new_column_name):
    '''
    Function to Compare Whether two Columns are Equal. 
    Equivalancy in this broadened definition includes Null to Null, Null to 0, "" to Null. and 0 to 0.

    Parameters:
        df (dataframe)
        col(str): Column Name from Dataframe
        col1(str): Column Name from Dataframe (should 
        new_column_name (str): Name of New Binary Column

    Returns:
        New Dataframe Column, NEW_COLUMN_NAME.
     
    '''
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