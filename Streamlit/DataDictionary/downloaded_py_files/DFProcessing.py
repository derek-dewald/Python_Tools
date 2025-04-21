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