'''
module_word: time_series
module_definition: Functions and tools which support the management, anaylsis and evaluation of Time Series Specific Datasets

'''

import numpy as np
import pandas as pd

from scipy.stats import t

def statistical_analytics_time_series(df,index=None):
    '''
    Definition:
        Apply Statistical Analysis to all columns in a Time Series Dataframe
    Parameters:
        df(dataframe): DataFrame

    Returns:
        df
    Date Created:
        10-Sep-26
    Date Last Modified:
        10-Sep-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        df = statistical_analytics_time_series(df)
    Notes:
        None
    Required Functions:
        None
    
    '''
    
    df = df.copy()
    
    if index:
        df=df.set_index(index)
    
    columns = df.columns
    
    value_dict = {
        'TOTAL':df.sum(axis=1),
        'MEAN':df.mean(axis=1),
        'STD DEV':df.std(axis=1),
        'MIN':df.min(axis=1),
        'MAX':df.max(axis=1),
        'CHG_1M':df[columns[-1]]-df[columns[-2]],
        'CHG_PERIOD':df[columns[-1]]-df[columns[0]],
    }
    
    value_dict['PCT_CHG_1M'] = np.where(df[columns[-2]] != 0,(df[columns[-1]] - df[columns[-2]]) / df[columns[-2]],np.nan)
    
    if len(columns)>=4:
        value_dict['CHG_3M'] = df[columns[-1]]-df[columns[-4]]
        value_dict['PCT_CHG_3M'] = np.where(df[columns[-4]] != 0,(df[columns[-1]] - df[columns[-4]]) / df[columns[-4]],np.nan)
    if len(columns)>=7:
        value_dict['CHG_6M'] = df[columns[-1]]-df[columns[-7]]
        value_dict['PCT_CHG_6M'] = np.where(df[columns[-7]] != 0,(df[columns[-1]] - df[columns[-7]]) / df[columns[-7]],np.nan)
    
    if len(columns)>=13:
        value_dict['CHG_12M'] = df[columns[-1]]-df[columns[-13]]
        value_dict['PCT_CHG_12M'] = np.where(df[columns[-13]] != 0,(df[columns[-1]] - df[columns[-13]]) / df[columns[-13]],np.nan)
    
        recent_6m    = df[columns[-6:]].sum(axis=1)
        previous_6m  = df[columns[-12:-6]].sum(axis=1)
        
        value_dict['SUM_6M_t0'] = recent_6m
        value_dict['SUM_6M_t1'] = previous_6m
        
        value_dict['INTERVAL_6M_CHG'] = recent_6m - previous_6m
        value_dict['INTERVAL_6M_CHG_PERC'] = np.where(previous_6m !=0, (recent_6m - previous_6m)/previous_6m,np.nan)

    if len(columns)>=4:
        
        historical = df[columns[:-1]]
        
        n = historical.count(axis=1)
        mean = historical.mean(axis=1)
        std = historical.std(axis=1)
        
        current = df[columns[-1]]
        
        value_dict['CURRENT_ZSCORE'] = (current - mean) / std.replace(0, np.nan)                    # Standardized anomaly score
        value_dict['CURRENT_TSTAT'] = (current - mean) /(std * np.sqrt(1 + 1/n)).replace(0, np.nan) # Predictive t-statistic
        value_dict['CURRENT_PVALUE'] = 2 * t.sf(np.abs(value_dict['CURRENT_TSTAT']),df=n - 1)       # Two-sided p-value
        value_dict['ANOMALY_FLAG'] = (value_dict['CURRENT_ZSCORE'].abs() >= 2).astype(int)
        value_dict['SIGNIFICANCE_FLAG'] = (value_dict['CURRENT_PVALUE'] < 0.05).astype(int)

    for key in value_dict:
        df[key] = value_dict[key]
    
    return df
