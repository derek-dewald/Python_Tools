'''
module_word: data_validation
module_definition: Repo for functions which validate the structure, completeness and validity of Dataframes. This DOES NOT, attempt to understand the underlying Data, but rather the structure. Note the distinction between Dataframe Structure and Underlying Data, which is EDA.

'''

import numpy as np
import pandas as pd

from IPython.display import Markdown,display
from eda_functions import binary_complex_equivalency
from object_rendering import visualize_dataframe_in_notebook

def df_compare(df,df1,primary_key_list=[]):
    """
    Definition:
        Function which compares the Column Structure of 2 Dataframe. When a Primary Key List is Included, it will also test at a Element level the equivalency of information contained within
    Parameters:
        df(df)
        df1(df)
        primary_key_list(list): List of Elements to Merge DataFrame on. If left blank it will only test column structure of Dataframes.
    Returns:
        df
    Date Created:
        26-Sep-26
    Date Last Modified:
        26-Sep-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        df_compare(df, df1)
        df_compare(df,df1,['Function','Folder'])
    Notes:
        This is to set Structure and changes in information at a high level. For a more granular understanding of data level change, refer to EDA functions.
    Required Functions:
        binary_complex_equivalency
        visualize_dataframe_in_notebook
        display
        markdown
        pandas
        
    """

    # Compare Columns.
    columns = set(df.columns)
    columns1 = set(df1.columns)

    common_columns = sorted(columns & columns1)

    display(Markdown(f'### Reviewing Column Structure of Dataframes'))
    display(Markdown(f'#### Assessment:'))
    
    if columns==columns1:
        display(Markdown(f'##### Structure of Dataframes is Equal'))

    else:
        display(Markdown(f'##### Structure of Dataframes is Not Equal.'))
        display(Markdown(f'###### Common Columns:{common_columns}'))
        display(Markdown(f'###### Columns only in DF:{sorted(columns-columns1)}'))
        display(Markdown(f'###### Columns only in DF1:{sorted(columns1-columns)}'))
        
    if (len(common_columns)>0)&(len(primary_key_list)>0):
         # Compare values in columns
        display(Markdown(f'### Reviewing Data in Common Columns'))
    
        temp_df = df.merge(df1,on=primary_key_list,how='outer',indicator=True,suffixes=('',"_"))

        test_columns = []
        for column in [x for x in common_columns if x not in primary_key_list]:
            temp_test = f"{column}_EQ_FLAG"
            binary_complex_equivalency(temp_df,column,f"{column}_",new_column_name=temp_test)
            test_columns.append(temp_test)

        result_df = pd.DataFrame({
            'EQ': temp_df[test_columns].sum(),
            'NE': len(temp_df) - temp_df[test_columns].sum(),
            'PERC': (temp_df[test_columns].mean() * 100)
        })
            
        display(Markdown(f'#### Equivalency Results'))
        visualize_dataframe_in_notebook(result_df.reset_index().rename(columns={'index':'Column'}),show_as_percentage=['PERC'])
    
        final_df = pd.DataFrame()
    
        for column in result_df[result_df['NE']!=0].index.tolist():
            col = column.replace('_EQ_FLAG',"")
            col1 = column.replace('EQ_FLAG',"")
        
            print
            final_columns = [col,col1] + primary_key_list
        
            d = temp_df[temp_df[column]==0][final_columns].rename(columns={col:'COLUMN_DF',col1:'COLUMN_DF1'})
            d['COLUMN'] = col
            final_df = pd.concat([final_df,d])
    
        display(Markdown(f'#### Sample of results which do not equal'))
        visualize_dataframe_in_notebook(final_df)
        return final_df
    
    return pd.DataFrame()