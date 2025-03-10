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

def ColumnPartitioner(df,
                      column_name,
                      new_column_name='Partition',
                      new_value_column='Total Balance in Partion',
                      partitions=10,
                      exclude_blanks=1,
                      exclude_zeros=0,
                      return_value=''):
    '''
    Function to create partions from Float or INT column which returns the Upper Partion Bound for a Column in a DataFrame. 
    Inspired by the Decile Analysis, it quickly highlights the distribution of a given dataset.
    
    Args:
        partitions:
            Total Number of desired Partitions. Default 10 as a homage to DR and his love of the Decile Analysis.
    
        Exclude Blanks:
            Binary flag to determine whether null value records  are to be considered in the Analysis. If 1 then 
            they are excluded, otherwise, they are given a value of 0 and included. Note that this can Materially 
            Impact Distribution and Inference, so should be carefully considered.
        
        Exclude Zeros:
            Binary flag to determine whether 0 value records are to be considered in the analysis. If 1 then they are excluded,
            otherwise they are included. Note that this can Materially Impact Distribution and Inference, so should be carefully
            considered.
        
        Return Value:
            Value to be returned:
            default (""):       DF of Value at Individual Partition Locations
            list_index(list):   Returns list of Index Locations in Dataframe
            list_value(list):   List of Value at Individual Partition Locations 
            merge(df):          New Column in existing DF which is numerical value of segment which value belongs
            agg_value(df):      DF of Aggregate Value total Impact of Each Segment
            position_value(df)  DF of Position (Transposed Default DF) and agg_value dataframe.
            

        New Column Name:
            Name of New Column if original Partition is choosen. By Default, Parition is choosen.
            
    '''
    if partitions <2:
        return print('Requries a Minimum of 2 partitions, recommends no less than 3 partitions')
    
    # Make a copy to ensure no overwriting
    temp_df = df.copy()
    
    # Clean Dataset 
    if exclude_blanks ==1:
        blanks_removed = len(temp_df[temp_df[column_name].isnull()])
        #print(f"Blank Entries Removed: {blanks_removed}")
        temp_df = temp_df[temp_df[column_name].notnull()]
    else:
        temp_df[column_name] = temp_df[column_name].fillna(0)
        
    if exclude_zeros ==1:
        zeroes_removed = len(temp_df[temp_df[column_name]==0])
        #print(f"Zero Entries Removed: {zeroes_removed}")
        temp_df = temp_df[temp_df[column_name]!=0]
        
    column_list = temp_df[column_name].tolist()
    column_list.sort()
    length_of_df = len(column_list)
    break_point = math.ceil(length_of_df/partitions)
    
    if partitions >=length_of_df:
        return print(f'Sample Size insufficient to Warrant Calculation for column {column_name}, please review data')
        
    record_position = list(range(0,length_of_df,break_point))
    record_value = [column_list[x] for x in record_position]
    #print(record_value)
    
    
    # Parition Value DF
    
    partition_df = pd.DataFrame(record_value,index=[f"{new_column_name} {x+1}" for x in range(len(record_value))],columns=[column_name]).T
            
    if return_value == '':
        return partition_df
    elif return_value == 'list_value':
        return record_value
    elif return_value == 'list_index':
        return record_position
    elif (return_value == 'merge')|(return_value == 'agg_value')|(return_value=='position_value'):
        temp_df = temp_df.sort_values(column_name).reset_index(drop=True)
        temp_df[new_column_name] = np.searchsorted(record_position,temp_df.index,side='right')
        
        if (return_value == 'agg_value')|(return_value=='position_value'):
            agg_impact = temp_df[['Partition',column_name]].groupby('Partition').sum()[column_name].values
            agg_impact_df = pd.DataFrame(agg_impact,
                                         columns=['VALUE'],
                                         index=[f"{new_value_column} {x+1}" for x in range(len(agg_impact))])
            
            if return_value=='position_value':
                agg_impact_df.reset_index(drop='True',inplace=True)
                agg_impact_df['index'] = [f"{new_column_name} {x+1}" for x in range(len(agg_impact_df))]
                agg_impact_df.set_index('index',inplace=True)
                
                temp_df1 =  partition_df.T.merge(agg_impact_df,
                                                 left_index=True,
                                                 right_index=True,
                                                 how='left').rename(columns={'VALUE':"AGGREGATE_VALUE","VARIANCE":"PARTITION",})
                return temp_df1
            return agg_impact_df
        return temp_df
