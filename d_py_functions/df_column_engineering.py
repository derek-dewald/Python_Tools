'''
module_word: df_column_engineering
module_definition: Repo for functions which update, clean, change, enhance DF Columns. Primarily Text Related. Numerical Changes in Feature Engineering.

'''

import numpy as np
import pandas as pd

def standardize_column_text(
    df,
    column_name,
    new_column_name=None,
):
    '''
    Definition:
        Function which is used to Standardize Text, specifically it replaces anything that is [^\w\s] - anything that is not a word character (\w) and not whitespace (\s) or [\d] is a digit (0–9), and lower cases it.
    Parameters:
        df (dataframe)
        column_name (str): Column Name of DF column to update.
        new_column_name (str): New Column Name of Updated Field. If Blank, then it replaces exists column
    Returns:
        TBD
    Date Created:
        09-Sep-26
    Date Last Modified:
        09-Sep-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        TBD
    Notes:
        None
    Required Functions:
        None
    '''
    if not new_column_name:
        new_column_name = column_name

    df[new_column_name] = df[column_name].fillna("").apply(lambda x:x.lower().strip())
    df[new_column_name] = df[new_column_name].str.replace(r'[^\w\s]|[\d]','',regex=True)


def word_counts_from_column(df,
                            column_name,
                            lower= True,
                            pattern= r"[A-Za-z']+",
                            min_len= 1,
                            dropna=True):
    """
    Vectorized word counting over a DataFrame column. As default it applies word cleaning, and standardization and only counts Letters.
    
    Parameters:
        df (df): DataFrame
        column_name(str): Column containing text to iterate through.
        lower(bool): Apply Lowercasing for consistency before counting.
        pattern(re): r"[A-Za-z']+". Regex for tokens (words). Adjust for unicode if needed. [A-Za-z0-9']+ if you want to include Numbers
        min_len(int): Minimum token length to keep.
        dropna(bool): Whether to drop empty tokens.
        
    Returns:
        pd.Series. Word counts indexed by token (sorted desc).

    date_created:01-Feb-26
    date_last_modified: 01-Feb-26
    classification:TBD
    sub_classification:TBD
    usage:
        Example Function Call

    """
    s = df[column_name]

    # Convert everything to string safely; keep NaN out of the way
    # Using astype(str) turns NaN into 'nan', so instead fillna('') first.
    s = s.fillna("").astype(str)

    if lower:
        s = s.str.lower()

    # Find all tokens per row (vectorized)
    tokens = s.str.findall(pattern).explode()

    if dropna:
        tokens = tokens.dropna()

    if min_len > 1:
        tokens = tokens[tokens.str.len() >= min_len]

    return tokens.value_counts()



def set_manual_column_order(df,column_name,column_order):
    
    '''
    
    '''
    
    df[column_name] = pd.Categorical(df[column_name], categories=column_order, ordered=True)

def calculate_boundaries(df, column_name, breaks=10):
    """
    Calculates percentile-based segmentation boundaries and cumulative
    value distributions for a numerical column.

    Purpose:
        Identifies segment thresholds based on the distribution of values, comparing boundaries calculated with and without zero observations.
        Also measures how total value accumulates across these boundaries.

    Parameters:
        df(df) : DataFrame containing the observations.
        column_name(str): Numeric column to analyze.
        breaks(int): Number of percentile divisions.Defaults to 10 (deciles).

    Retuns:
        df
    Date Created:
        11-Nov-26
    Date Last Modified:
        11-Nov-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        from synthetic_member import member_df,activity_df,mbr_profile
        a = calculate_boundaries(activity_df,'RENT_M')
    Notes:
        None
    Required Functions:
        None
    
    """

    s = df[column_name]
    s_nonzero = s[s != 0]

    percentiles = [i / breaks for i in range(1, breaks + 1)]

    boundaries = s.quantile(percentiles).values
    boundaries_nonzero = s_nonzero.quantile(percentiles).values

    total_value = s.sum()

    cumulative_value = [
        s[s <= boundary].sum()
        for boundary in boundaries
    ]
    
    cumulative_pct = [
        s[s <= boundary].sum() / total_value
        if total_value != 0 else 0
        for boundary in boundaries
    ]

    final_df = pd.DataFrame({
        'Segment': range(1, breaks + 1),
        f'{column_name}_SEG': boundaries,
        f'{column_name}_SEG1': boundaries_nonzero,
        'CUMULATIVE_VALUE': cumulative_value,
        'CUMULATIVE_VALUE_PCT': cumulative_pct
    })

    return final_df.set_index('Segment')
