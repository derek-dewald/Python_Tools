'''
module_word: feature_engineering
module_definition: Repo for functions which create data elements within the context of an existing Data Frame. Does not include the creation of New Data Sets.

'''

import numpy as np
import pandas as pd

def binary_complex_equivalency(
    df,
    column_name,
    column_name1,
    new_column_name='EQ_FLAG',
    eq=1,
    ne=0,
    tolerance=.001,
    include_difference=False
):

    '''
    Definition:
        Function which tests the Equivalence of 2 Columns in a dataframe to determine whether they are Equal or Not.
    Parameters:
        df(dataframe): DataFrame
        column_name(str): Name of Column to be tested against column_name1.
        column_name1(str): Name of Column to be tested against column_name.
        new_column_name(str): Name of Created Column, representing where values are either Equal, or Not Equal. Default Name is EQ_FLAG
        eq(str): Any Value which will represent condition Matching.
        ne(str): Any value, representing Not Equal Condition
        tolerance(float): Degree of grace to be applied by np.is_close() to minimize python rounding errors.
        include_difference(bool): Boolean flag to determine whether to include arthematic difference in addition to binary flag.
        
    Returns:
        df
    Date Created:
        28-Aug-26
    Date Last Modified:
        28-Aug-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        binary_complex_equivalency(df,'SCORE1','SCORE2')
    Notes:
        None
    Required Functions:
        None
    
    
    
    '''
    
    s1 = df[column_name]
    s2 = df[column_name1]

    both_null = s1.isna() & s2.isna()

    if (
        pd.api.types.is_numeric_dtype(s1)
        and pd.api.types.is_numeric_dtype(s2)
    ):
        equal = np.isclose(
            s1,
            s2,
            atol=tolerance,
            rtol=0,
            equal_nan=True
        )
        if include_difference:
            df['COLUMN_DIFF'] = s1 - s2
            
    else:
        equal = s1.eq(s2) | both_null

        if include_difference:
            df['COLUMN_DIFF'] = np.nan

    df[new_column_name] = np.where(equal, eq, ne)

    return df


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
    
    if len(columns)>=3:
        value_dict['CHG_3M'] = df[columns[-1]]-df[columns[-4]]
        value_dict['PCT_CHG_3M'] = np.where(df[columns[-4]] != 0,(df[columns[-1]] - df[columns[-4]]) / df[columns[-4]],np.nan)
    if len(columns)>=6:
        value_dict['CHG_6M'] = df[columns[-1]]-df[columns[-7]]
        value_dict['PCT_CHG_6M'] = np.where(df[columns[-7]] != 0,(df[columns[-1]] - df[columns[-7]]) / df[columns[-7]],np.nan)
    
    if len(columns)>=12:
        value_dict['CHG_12M'] = df[columns[-1]]-df[columns[-13]]
        value_dict['PCT_CHG_12M'] = np.where(df[columns[-13]] != 0,(df[columns[-1]] - df[columns[-13]]) / df[columns[-13]],np.nan)
    
        recent_6m    = df[columns[-6:]].sum(axis=1)
        previous_6m  = df[columns[-12:-6]].sum(axis=1)
        
        value_dict['SUM_6M_t0'] = recent_6m
        value_dict['SUM_6M_t1'] = previous_6m
        
        value_dict['INTERVAL_6M_CHG'] = recent_6m - previous_6m
        value_dict['INTERVAL_6M_CHG_PERC'] = np.where(previous_6m !=0, (recent_6m - previous_6m)/previous_6m,np.nan)
    
    value_dict['CURRENT_ZSCORE'] = (
        (df[columns[-1]] - value_dict['MEAN'])/ 
        value_dict['STD DEV'].replace(0, np.nan))
    
    value_dict['SIGNIFICANCE_FLAG'] = (value_dict['CURRENT_ZSCORE'].abs() >= 2).astype(int)
    
    for key in value_dict:
        df[key] = value_dict[key]
    
        
    return df