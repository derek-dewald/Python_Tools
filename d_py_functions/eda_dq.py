import pandas as pd
import numpy as np
import math

import sys
#sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\BEEM_PY\\')
#sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\d_py_functions\\')
#from FeatureEngineering import BinaryComplexEquivlancey as binary_complex_equivlance

sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

def top_x_records(data, column_name=None, top_n=5, dropna=False):
    s = data if isinstance(data, pd.Series) else data[column_name]
    col = column_name or getattr(s, "name", "VALUE")

    vc = s.value_counts(dropna=dropna).head(top_n)
    if vc.empty:
        return pd.DataFrame()

    # e.g. "Montreal Expos (1234)"
    rows = [f"{idx} ({cnt})" for idx, cnt in vc.items()]
    out = pd.DataFrame({col: rows}, index=[f"TOP_{i+1}" for i in range(len(rows))])
    return out


def column_statistical_review(
    df,
    column_name,
    partitions=10,
    top_records_to_include=5,
    cummulative_sum=False,
    cummulative_percent=False,
    remove_null_from_decile=False,
    remove_zero_from_decile=False):

    '''
    Function to compute statistical properities of a particular column, function approaches Numeric and Non Numberic Differntly, non numeric
    calculations, Total Records, Unique Records, Null Records and the Mode.

    For Numreic Calculations, can also return, Mean, Mode, Standard Deviation, Max, Min, Mode, Number of Nulls, Zeroes, Non Zeros.

    Additionally, there are 4 Additional Components, which can include a top Value count of the most frequent records, the decile positions o
    the records, the Cummulative Sum of the Records, and the Percentage of Cummulative POsition at each decile.

    Parameters: 
        df (df): Any DataFrame.
        column_name(str): Name of particular column to where function will be applied.
        partitions(int): Number of Partitions to be applied in returned DF (default is 10, if 0 nothing returned.) 
        top_records_to_include(int): Number of Top Records to be returned in DF (default is 5, if 0 nothing returned.) 
        cummulative_sum (bool): Boolean, if True, will add the Cummulative Sum Value into the Return DF
        cummulative_percent (bool): Boolean, if True, will add the Cummulative Sum Percent  into the Return DF
        remove_null_from_decile (bool): Boolean, if true, it will remove Null and NA values from Caluclation of Decile
        remove_zero_from_decile (bool): Boolean, if true, it will remove Zero values from Caluclation of Decile

    Returns:
        Dataframe

    date_created:24-Jan-26
    date_last_modified: 24-Jan-26
    classification:TBD
    sub_classification:TBD
    usage:
        df = column_statistical_review(final_mbr_df,'AGE')
    
    '''
    # Create a Series
    s = df[column_name]

    # Boolean
    is_numeric = pd.api.types.is_numeric_dtype(s)

    # Calculations
    total = len(s)
    is_null = s.isna()  # covers NaN / None
    null_count = int(is_null.sum())

    # Value Dict.
    out = {
        "TOTAL_RECORDS": total,
        "UNIQUE_RECORDS": s.nunique(dropna=False),   # closer to “unique records” intent
        "NULL_RECORDS": null_count,
        "MODE":s.mode().to_list()
    }

    if is_numeric:
        # quick numeric stats (skipna=True by default)
        out.update({
            "SUM": s.sum(),
            "MEAN": s.mean(),
            "STD_DEV": s.std(),
            "MEDIAN": s.median(),
            "MAX": s.max(),
            "MIN": s.min(),
        })

        # compute once and reuse
        is_zero = s.eq(0) & ~is_null
        zero_count = int(is_zero.sum())
        out["ZERO_RECORDS"] = zero_count
        out["NON_ZERO_RECORDS"] = total - null_count - zero_count

    # Build output frame
    temp_df = pd.Series(out, name=column_name).to_frame()
        
    if total == null_count:
        return temp_df

    if (remove_null_from_decile | remove_zero_from_decile):
    
        mask = pd.Series(True, index=s.index)
    
        if remove_null_from_decile:
            mask &= s.notna()
    
        if remove_zero_from_decile:
            mask &= s.ne(0)

        s = s.loc[mask]
    
    if top_records_to_include>0:
        try:
            temp_df = pd.concat([temp_df,top_x_records(s,column_name)])
        except:
            pass
    
    if partitions>0:
        segments = s.sort_values().quantile(np.linspace(1/10, 1, 10), interpolation="higher").tolist()
        df_segments = pd.DataFrame({
            column_name: segments,
            'Decile': [f"Decile {x+1}" for x in range(len(segments))]
        }).set_index('Decile')
        temp_df = pd.concat([temp_df,df_segments])

    if cummulative_sum:
        cumm_segments = s.sort_values().cumsum().quantile(np.linspace(1/10, 1, 10), interpolation="higher").tolist()
        df_cumm_sum = pd.DataFrame({
            column_name: cumm_segments,
            'CUMMALATIVE_SUM': [f"Cummulative Sum at Decile {x+1}" for x in range(len(segments))]
        }).set_index('CUMMALATIVE_SUM')
        temp_df = pd.concat([temp_df,df_cumm_sum])
        
    if cummulative_percent:
        cumm_seg_perc = [(x/cumm_segments[-1])*100 for x in cumm_segments]

        df_cumm_sum_perc = pd.DataFrame({
            column_name: cumm_seg_perc,
            'CUMMALATIVE_SUM_PERC': [f"Cummulative Sum Percentage at Decile {x+1}" for x in range(len(segments))]
        }).set_index('CUMMALATIVE_SUM_PERC')
        temp_df = pd.concat([temp_df,df_cumm_sum_perc])
        
    return temp_df


def ReviewEntireDataframe(df,file_name=None):
    
    final_df = pd.DataFrame()
    
    for column in df.columns:
        start_time = timeit.default_timer()
        temp_df = ColumnStatisticalReview(df,column)
        print(f'Elapsed time to process {column}:{timeit.default_timer() - start_time:,.2f}')
        final_df = pd.concat([final_df,temp_df],axis=1)
    if file_name:
        final_df.to_csv(f"{file_name}.csv")
        
    return final_df

