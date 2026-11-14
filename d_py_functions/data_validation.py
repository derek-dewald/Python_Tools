'''
module_word: data_validation
module_definition: Repo for functions which validate the structure, completeness and validity of Dataframes. This DOES NOT, attempt to understand the underlying Data, but rather the structure. Note the distinction between Dataframe Structure and Underlying Data, which is EDA.

'''

import numpy as np
import pandas as pd

from IPython.display import Markdown,display
from feature_engineering import binary_complex_equivalency
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
        visualize_dataframe_in_notebook(final_df.drop_duplicates('COLUMN'))
        return final_df
    
    return pd.DataFrame()


def df_column_comparison(
    df,
    column_name,
    primary_key,
    column_name1=None,
    secondary_filter=None,
    column_distinction='_',
    bracketing=None
):
    """
    Definition:
        Compare 2 Columns in a Dataframe. Excellent for a quick understand of Time Series Data, or DQ analysis when attempting to validate lack (or presence of change).
    
    Parameters:
        df(df):Dataframe.
        column_name (str): Primary column to compare.
        primary_key (str): Reference Column (used when presenting Account DF).
        column_name1 (str):Comparison column. Defaults to f"{column_name}{column_distinction}".
        secondary_filter (str, optional): Optional column used to segment summary results.
        column_distinction (str): Suffix used to identify the comparison column.
        bracketing (list): Breakpoints used to categorize numeric differences. Defaults to [-10000, -1000, -100, -1,0,1, 100, 1000, 10000]

    Returns:
        dict
    Date Created:
        21-Aug-25
    Date Last Modified:
        11-Nov-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        a=ColumnDQComparison(df=activity_df[['MEMBERNBR','RENT_M']],column_name='RENT_M',primary_key='MEMBERNBR',column_name1='RENT_M1',)
    Notes:
        None
    Required Functions:
        None

    """

    if bracketing is None:
        bracketing = [-10000, -1000, -100, -1,0,1, 100, 1000, 10000]

    if column_name1 is None:
        column_name1 = f"{column_name}{column_distinction}"

    # --------------------------------------------------
    # Build working dataframe
    # --------------------------------------------------

    columns = [primary_key]

    if secondary_filter:
        columns.append(secondary_filter)

    columns.extend([column_name, column_name1])

    temp_df = (
        df[columns]
        .rename(
            columns={
                column_name: 'DF',
                column_name1: 'DF1'
            }
        )
        .copy()
    )

    temp_df['COLUMN_NAME'] = column_name

    # --------------------------------------------------
    # Equality / null checks
    # --------------------------------------------------

    binary_complex_equivalency(temp_df,'DF','DF1','VALUES_EQUAL',eq=1,ne=0)

    temp_df['VALUES_NOT_EQUAL'] = 1 - temp_df['VALUES_EQUAL']

    temp_df['NULL_RECORD_DF'] = temp_df['DF'].isna().astype(int)
    temp_df['NULL_RECORD_DF1'] = temp_df['DF1'].isna().astype(int)

    # --------------------------------------------------
    # Difference
    # --------------------------------------------------

    if (
        pd.api.types.is_numeric_dtype(temp_df['DF'])
        and pd.api.types.is_numeric_dtype(temp_df['DF1'])
    ):
        temp_df['DIFFERENCE'] = temp_df['DF'].fillna(0)- temp_df['DF1'].fillna(0)
        create_segment_range(temp_df,'DIFFERENCE',bracketing,'DIFF_SEGMENT')

    else:
        temp_df['DIFFERENCE'] = 0
        temp_df['DIFF_SEGMENT'] = 'Could Not Calculate'

    # --------------------------------------------------
    # Grouped comparison
    # --------------------------------------------------

    temp_df1 = temp_df.assign(RECORD_COUNT=1)

    group_columns = ['COLUMN_NAME', 'DF', 'DF1']

    if secondary_filter:
        group_columns.insert(0, secondary_filter)

    groupby_df = (
        temp_df1
        .groupby(group_columns, dropna=False)
        .agg(
            RECORD_COUNT=('RECORD_COUNT', 'sum'),
            VALUES_EQUAL=('VALUES_EQUAL', 'sum'),
            VALUES_NOT_EQUAL=('VALUES_NOT_EQUAL', 'sum'),
            NULL_RECORD_DF=('NULL_RECORD_DF', 'sum'),
            NULL_RECORD_DF1=('NULL_RECORD_DF1', 'sum')
        )
        .reset_index()
        .sort_values('VALUES_EQUAL', ascending=False)
        .head(20)
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def create_summary(data):
        record_count = len(data)
        values_equal = data['VALUES_EQUAL'].sum()

        return pd.Series({
            'Total Combined Records': record_count,
            'Values Equal': values_equal,
            'Values Not Equal': data['VALUES_NOT_EQUAL'].sum(),
            'Percent Values Equal':
                (values_equal / record_count * 100)
                if record_count else 0,
            'Null Records DF': data['NULL_RECORD_DF'].sum(),
            'Null Records DF1': data['NULL_RECORD_DF1'].sum(),
            'Total Difference': data['DIFFERENCE'].sum(),
            'Mean Difference': data['DIFFERENCE'].mean()
        })

    if secondary_filter:

        summary_df = (
            temp_df
            .groupby(secondary_filter, dropna=False)
            .apply(create_summary)
            .reset_index()
        )

        summary_df.insert(1, 'COLUMN_NAME', column_name)

    else:

        summary_df = (
            create_summary(temp_df)
            .to_frame()
            .T
        )

        summary_df.insert(0, 'COLUMN_NAME', column_name)

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    return {
        'summary_df': summary_df,
        'account_df': temp_df,
        'groupby_df': groupby_df
    }