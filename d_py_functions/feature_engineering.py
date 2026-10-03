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

def create_segment_range(
    df,
    column_name,
    segment_range,
    return_value='text'
):
    """
    """

    segment_range = sorted(segment_range)

    labels = [f"1) LTE {segment_range[0]}"]
    bins = [-float('inf')] + segment_range + [float('inf')]

    for i, (low, high) in enumerate(zip(segment_range[:-1], segment_range[1:]), start=2):
        labels.append(f"{i}) GT {low} - LTE {high}")

    labels.append(f"{len(segment_range) + 1}) GT {segment_range[-1]}")

    df[f"{column_name}_SEGMENT"] = pd.cut(
        df[column_name],
        bins=bins,
        labels=labels,
        right=True
    )
    
    display(df[f"{column_name}_SEGMENT"].value_counts().sort_index())