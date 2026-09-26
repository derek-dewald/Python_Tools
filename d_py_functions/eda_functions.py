'''
module_word: eda_functions
module_definition: Repo for functions required to implement Exploratory Data Analysis in Machine Learning Lifecycle

'''

import numpy as np
import pandas as pd

from feature_engineering import binary_complex_equivalency

def univariate_analysis(
    df,
    sparsity_threshold=0.90,
    skew_threshold=2,
    extreme_threshold=3,
    high_cardinality_threshold=0.50
):

    records = []

    for column in df.columns:

        s = df[column]
        numeric = pd.api.types.is_numeric_dtype(s)

        record = {
            'VARIABLE': column,
            'DTYPE': str(s.dtype),
            'COUNT': s.count(),
            'UNIQUE': s.nunique(dropna=True),
            'UNIQUE_PCT': s.nunique(dropna=True) / s.count()
                if s.count() > 0 else np.nan
        }

        if numeric:

            q01 = s.quantile(.01)
            q05 = s.quantile(.05)
            q25 = s.quantile(.25)
            q50 = s.quantile(.50)
            q75 = s.quantile(.75)
            q95 = s.quantile(.95)
            q99 = s.quantile(.99)

            iqr = q75 - q25

            non_zero = s[(s != 0) & s.notna()]

            record.update({

                # Central tendency
                'MEAN': s.mean(),
                'MEDIAN': q50,

                # Variation
                'STD': s.std(),
                'IQR': iqr,

                # Distribution
                'SKEW': s.skew(),

                # Range / Percentiles
                'MIN': s.min(),
                'P01': q01,
                'P05': q05,
                'Q1': q25,
                'Q3': q75,
                'P95': q95,
                'P99': q99,
                'MAX': s.max(),

                # Sparsity
                'ZERO_COUNT': s.eq(0).sum(),
                'ZERO_PCT': s.eq(0).mean(),

                # Non-zero population
                'NON_ZERO_COUNT': len(non_zero),
                'NON_ZERO_PCT': len(non_zero) / s.count()
                    if s.count() > 0 else np.nan,

                'NON_ZERO_MEDIAN': non_zero.median()
                    if len(non_zero) > 0 else np.nan,

                'NON_ZERO_STD': non_zero.std()
                    if len(non_zero) > 1 else np.nan,

                'NON_ZERO_SKEW': non_zero.skew()
                    if len(non_zero) > 2 else np.nan,

                # Potential extreme observations using IQR
                'EXTREME_LOW_COUNT': (
                    (s < (q25 - extreme_threshold * iqr)).sum()
                    if iqr > 0 else 0
                ),

                'EXTREME_HIGH_COUNT': (
                    (s > (q75 + extreme_threshold * iqr)).sum()
                    if iqr > 0 else 0
                )
            })

        records.append(record)

    result = pd.DataFrame(records)

    # --------------------------------------------------
    # Statistical Property Flags
    # --------------------------------------------------

    result['HIGH_SPARSITY'] = (
        result['ZERO_PCT'] >= sparsity_threshold
    ).astype(int)

    result['HIGH_SKEW'] = (
        result['SKEW'].abs() >= skew_threshold
    ).astype(int)

    result['NO_VARIATION'] = (
        result['STD'] == 0
    ).astype(int)

    result['EXTREME_VALUES'] = (
        (result['EXTREME_LOW_COUNT'] > 0) |
        (result['EXTREME_HIGH_COUNT'] > 0)
    ).astype(int)

    result['HIGH_CARDINALITY'] = (
        result['UNIQUE_PCT'] >= high_cardinality_threshold
    ).astype(int)

    # Number of statistical properties requiring attention
    assessment_columns = [
        'HIGH_SPARSITY',
        'HIGH_SKEW',
        'NO_VARIATION',
        'EXTREME_VALUES',
        'HIGH_CARDINALITY'
    ]

    result['ASSESSMENT_COUNT'] = (
        result[assessment_columns]
        .sum(axis=1)
    )

    return result

def ml_validation_data_cleaning(df):


    records = []

    for column in df.columns:

        s = df[column]
        numeric = pd.api.types.is_numeric_dtype(s)

        records.append({
            'VARIABLE': column,
            'DTYPE': str(s.dtype),
            'RECORDS': len(s),
            'MISSING_COUNT': s.isna().sum(),
            'MISSING_PCT': s.isna().mean(),
            'UNIQUE_COUNT': s.nunique(dropna=True),
            'BLANK_COUNT': s.fillna('').astype(str).str.strip().eq('').sum() if not numeric else 0,
            'INFINITE_COUNT': np.isinf(s).sum() if numeric else 0,
            'CONSTANT_FLAG': int(s.nunique(dropna=True) <= 1),
            'ALL_MISSING_FLAG': int(s.isna().all())
        })

    return pd.DataFrame(records)

def stastical_compare_column_diff(
    df,
    column1,
    column2
):
    
    '''
    Definition:
        Statistically Analyze the Difference Value between two columns.
    Parameters:
        df(df): Dataframe
        column1(str): Name of Column to be represented as Column1. 
        column2(str): Name of Column to be represented as Column2. 
        
    Returns:
        df

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
        binary_complex_equivalency
    
    
    Need to document that is makes an assumption that Zeros and Blanks are handled through DQ Analysis. Not going to explicilty report here.

    '''

    temp_df = df[[column1,column2]].copy()
    
    binary_complex_equivalency(temp_df,column1,column2,include_difference=True)

    diff_stats  = temp_df['COLUMN_DIFF'].describe()

    top_5 = temp_df[temp_df['COLUMN_DIFF']!=0]['COLUMN_DIFF'].value_counts().head(5)
    
    dict_ = {
        'COLUMN1':column1,
        'COLUMN2':column2,
        'RECORDS':len(df),
        'EQUAL_VALUES':temp_df['EQ_FLAG'].sum(),
        'PERCENT_EQUAL':round((temp_df['EQ_FLAG'].sum()/len(df))*100,2),
        'DIFF_MEAN':temp_df['COLUMN_DIFF'].mean(),
        'DIFF_STD':temp_df['COLUMN_DIFF'].std(),
        'MAX_DIFF':temp_df['COLUMN_DIFF'].max(),
        '1ST_QUARTILE':diff_stats['25%'],
        '2ND_QUARTILE':diff_stats['50%'],
        '3RD_QUARTILE':diff_stats['75%']    
    }

    for i, (value, frequency) in enumerate(top_5.items(), start=1):
        dict_[f'TOP_{i}_OBSERVATION'] = f'Value: {value}, Frequency: {frequency}'

    return pd.DataFrame([dict_.values()],columns=dict_.keys())


def stastical_compare_column_diff_grouped(
    df,
    column1,
    column2,
    group_columns
):
    '''
    Definition:
        Extends stastical_compare_column_diff_grouped to include a Group By on variable, or list.
    Parameters:
        df(df): Dataframe
        column1(str): Name of Column to be represented as Column1. 
        column2(str): Name of Column to be represented as Column2. 
        group_columns(str): Can also be a list, column to apply group.
        
    Returns:
        df

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
        binary_complex_equivalency
    
    
    Need to document that is makes an assumption that Zeros and Blanks are handled through DQ Analysis. Not going to explicilty report here.

    
    
    
    '''
    if isinstance(group_columns, str):
        group_columns = [group_columns]

    temp_df = df[
        group_columns + [column1, column2]
    ].copy()

    binary_complex_equivalency(
        temp_df,
        column1,
        column2,
        include_difference=True
    )

    summary = (
        temp_df
        .groupby(group_columns)
        .agg(
            RECORDS=('EQ_FLAG', 'size'),
            EQUAL_VALUES=('EQ_FLAG', 'sum'),
            DIFF_MEAN=('COLUMN_DIFF', 'mean'),
            DIFF_STD=('COLUMN_DIFF', 'std'),
            MAX_DIFF=('COLUMN_DIFF', 'max')
        )
    )

    quartiles = (
        temp_df
        .groupby(group_columns)['COLUMN_DIFF']
        .quantile([.25, .50, .75])
        .unstack()
        .rename(columns={
            .25: '1ST_QUARTILE',
            .50: '2ND_QUARTILE',
            .75: '3RD_QUARTILE'
        })
    )

    result = (
        summary
        .join(quartiles)
        .reset_index()
    )

    result['PERCENT_EQUAL'] = (
        result['EQUAL_VALUES']
        / result['RECORDS']
        * 100
    ).round(2)

    result.insert(0, 'COLUMN1', column1)
    result.insert(1, 'COLUMN2', column2)

    return result