'''
module_name: eda_functions
module_purpose: Repo for functions required to implement Exploratory Data Analysis in Machine Learning Lifecycle

'''

import numpy as np
import pandas as pd


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