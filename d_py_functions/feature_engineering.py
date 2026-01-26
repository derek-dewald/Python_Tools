import pandas as pd
import numpy as np
import html
import textwrap
import random

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")


def binary_complex_equivlance(df, col, col1, new_column_name):
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


def binary_column_creator(df,
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
