'''
module_word: eda_functions
module_definition: Repo for functions required to implement Exploratory Data Analysis in Machine Learning Lifecycle

'''

from IPython.display import Markdown,display
from object_rendering import visualize_dataframe_in_notebook
import numpy as np
import pandas as pd

def review_dataset_lvl1( df, top_n=5, min_top_count=10 ):
    '''
    Definition:
        Function used to provide a level 1 assessment of the data to understand a mimimum state of the dataset. 
        Used as Step 1 in Exploratory Data Analysis.
    Parameters:
        df(dataframe): Dataset.
        top_n(int): Number of Variables to include the Top Observation in right most column of Dataset.
        min_top_count(int): Minimum Number of records which must be present in dataset in order to include in output.
    Returns:
        dict, df
    Date Created:
        09-Oct-26
    Date Last Modified:
        09-Oct-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        test = review_dataset_lvl1(df)
    Notes:
        None
    Required Functions:
        visualize_dataframe_in_notebook
        display
        Markdown
    
    '''
    records = []
    
    for column in df.columns:
    
        s = df[column]
        numeric = pd.api.types.is_numeric_dtype(s)
    
        is_empty_record = s[(s.isna())|((s==""))]
    
        record = {
            'VARIABLE': column,
            'DTYPE': str(s.dtype),
            'COUNT': s.count(),
            'UNIQUE': s.nunique(dropna=True),
            'NULL_COUNT':s.isna().sum(),           
            'ZERO_COUNT':(s==0).sum(),
            'BLANK_COUNT':(s=='').sum(),
            'NULL_BLANK_COUNT': len(is_empty_record),
            'NULL_BLANK_PCT': (len(is_empty_record) / len(s))*100 if s.count() > 0 else np.nan,
        }
        text_columns = list(record.keys())
    
        if numeric:
            q01 = s.quantile(.01)
            q05 = s.quantile(.05)
            q25 = s.quantile(.25)
            q50 = s.quantile(.50)
            q75 = s.quantile(.75)
            q95 = s.quantile(.95)
            q99 = s.quantile(.99)
    
            iqr = q75 - q25
            record.update({
                # Central tendency
                'MEAN': s.mean(),
                'MEDIAN': q50,
    
                # Variation
                'STD': s.std(),
    
                # Range / Percentiles
                'Q1': q25,
                'Q2': q50,
                'Q3': q75,            
                'MIN': s.min(),
                'MAX': s.max(),
                'P01': q01,
                'P05': q05,
                'P95': q95,
                'P99': q99
            })
                
        vc = s.value_counts(dropna=False)

        vc = vc[vc >= min_top_count]
        
        for i in range(top_n):
            record[f'TOP_{i+1}'] = np.nan
    
        for i, (value, count) in enumerate(vc.head(top_n).items()):
            pct = count / len(s) * 100
            record[f'TOP_{i+1}'] = f'{value}: {count:,} ({pct:.2f}%)'
        records.append(record)
        
    result = pd.DataFrame(records)

    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    
    numerical_col_df = result[result['VARIABLE'].isin(numeric_columns)]
    text_columns.extend([f'TOP_{i+1}' for i in range(top_n)])
    text_col_df      = result[~result['DTYPE'].isin(['int64','float64'])][text_columns]
    
    display(Markdown('## Results'))
    display(Markdown('## Numeric Columns'))
    visualize_dataframe_in_notebook(
        numerical_col_df,
        show_as_int=['ZERO_COUNT','NULL_COUNT','UNIQUE','COUNT'],
        show_as_number=['MEAN','MEDIAN','STD','IQR','SKEW','MIN','MAX','P01','P05','Q1','Q2','Q3','P95','P99']
    )
    
    display(Markdown('## Non Numeric Columns'))
    
    visualize_dataframe_in_notebook(
        text_col_df)
    
    return {"Numerical Columns":numerical_col_df,'Text Columns':text_col_df}

