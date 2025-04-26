## File Description: A more generalized Libary, including simple tricks, shortcuts and helpful functions for reviewing, summarizing or understanding. More Information, and less Procedural related. EDA for standardized processes.

from itertools import product,permutations,combinations


import pandas as pd
import numpy as np


def DataFrameColumnObservations(df,
                                columns,
                                include_obs=1,
                                include_perc=1):
    
    '''
    Function to provide a quick summary of how a Dataframe is distributed amongst a set of determined columns.
    
    
    Parameters:
        columns (list): Columns to which you wish to be invluded
        include_obs (int): Binary Flag to indicate whether column Observation Count Totals are included
        include_perc (int): Binary Flag to indicate whether column Percentage Calculations are to be included
    
    Returns:
        Dataframe
    
    '''

    temp_df = df[columns].copy()
    temp_df['COUNT'] = 1
    
    final_df = temp_df.fillna("").groupby(columns).sum().reset_index()
    
    for column in columns:
        temp_df1 = temp_df[[column,'COUNT']].groupby(column).sum().reset_index().rename(columns={'COUNT':f"{column}_OBS"})
        final_df = final_df.merge(temp_df1,on=column,how='left')

    if include_perc==1:
        final_df['PERC_DATA'] = final_df['COUNT']/len(temp_df)
        final_df = final_df.sort_values('COUNT',ascending=False)
        final_df['CUMMULATIVE_PERC'] = final_df['PERC_DATA'].cumsum()
        
        for column in columns:
            final_df[f"{column}_PERC"] = final_df[f"{column}_OBS"]/len(temp_df)
        
    if include_obs==0:
        return final_df.drop([f"{column}_OBS" for x in columns],axis=1)
    
    else:
        return final_df
    

def CombineLists(list_,
                 combo=1,
                 r=2):
        
    '''
    Function to 
    
    
    Parameters:
        
    
    Return:
        
    '''
    
    items = sum([1 if isinstance(x,list) else 0 for x in list_])

    if items==0:
        if combo==1:
            return list(map(list,combinations(list_,r)))
        else:
            return list(map(list,permutations(list_,r)))
    
    return list(map(list,product(*list_)))

def FilterDataframe(df,
                    binary_include={},
                    binary_exclude={},
                   ):
    
    temp_df = df.copy()
    
    if len(binary_include)!=0:
        for key, value in binary_include.items():
            temp_df = temp_df[temp_df[key]==value].copy()
            
    if len(binary_exclude)!=0:
        for key, value in binary_exclude.items():
            temp_df = temp_df[temp_df[key]!=value].copy()

    return temp_df


def TransposePivotTable(df,
                        reset_index=1,
                       x_axis='X_Axis',
                       y_axis='Y_Axis',
                       value='VALUE'):
    '''
    Function to take Pivot Table and Turn it into a Vertically Stacked List.
    
    Parameters:
        reset_index(int): Binary Flag to determine whether intex needs to be reset or not.
        x_axis (str): Column Naming from X_axis. Default X_Axis
        y_axis (str):
        value (str):
        
    Returns
        Dataframe
    
    '''
    
    if reset_index==1:
        df1 = df.stack().reset_index()
    else:
        df1 = df.stack()
    try:
        df1 =  df1.rename(columns={'level_0':x_axis,'level_1':y_axis,0:value})
    except:
        pass
    
    df1 = df1[df1[x_axis] < df1[y_axis]].copy()
    
    return df1[df1[x_axis]!=df1[y_axis]].sort_values(value)


def MissingCartesianProducts(list1_,
                             list2_,
                             columns,
                             merge_df=None,
                             remove_values=['0',"",'N/A']):
    '''
    Function which Looks at the Combination of Two Lists and explores all possible Combinations. 
    Developed for the purpose of understanding how many combinations exist and generating a list of Values which do not
    exist, this list can be valueable for pending to previously Aggregagted Datasets to insure all possible records have
    representation
    
    Parameters
        list1_ (list)
        list2_ (list)
        columns (Columns to be included, should represent the expected Column Name of list1_ and list2_)
        merge_df (Dataframe): To be used to Validate the number of missing records, if not included, then 
        returns only combination.
    
    Returns
    
    
    '''

    list1_ = [x for x in list1_ if x not in remove_values]
    list2_ = [x for x in list2_ if x not in remove_values]
    
    required_records = CombineLists([list1_,list2_])
    df = pd.DataFrame(required_records,columns=['METRIC_NAME','BRANCHNAME'])
    
    if len(merge_df)==0:
        return df
    
    else:
        df = df.merge(merge_df[columns].drop_duplicates(),on=columns,how='left',indicator=True)
        print(f"Distribution of Records and Missing Records:\n{df['_merge'].value_counts()}")
        df = df[df['_merge']=='left_only'].drop('_merge',axis=1)
        return df