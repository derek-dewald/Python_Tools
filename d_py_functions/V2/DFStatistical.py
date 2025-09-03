import pandas as pd
import numpy as np
import datetime
from scipy.stats import norm
import timeit

import sys
sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\BEEM_PY\\')
sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\d_py_functions\\')


from DFProcessing import ColumnPartitioner
from UtilityFunctions import PauseProcess

def SampleDataFrame(df, 
                    conf=.95, 
                    me=0.05,
                    mv=0.5,
                    print_=0,
                    new_column_name=""):
    """
    Returns a random sample from a DataFrame based on confidence level and margin of error.

    Parameters:
        df (pd.DataFrame): The dataset to sample from.
        conf(float): Desired Confidence Percentage Level (e.g., 90, 95, 99).
        me (float): Margin of Error, (default is 5%).
        mv (float): Maximum Variability (Expected Level of Default)

    Returns:
        pd.DataFrame: A random sample of the required size.
    """
    df = df.copy()
    
    if not 0 <= mv <= 1:
        raise ValueError("mv (failure rate) must be between 0 and 1.")

    N = len(df)
    if N == 0:
        raise ValueError("DataFrame is empty")

    # Calculate the Z-score based on the confidence level
    z = norm.ppf(1 - (1 - conf) / 2)
    

    # Calculate the initial sample size (without finite population correction)
    n0 = (z**2 * mv * (1 - mv)) / (me**2)
    
    # Apply finite population correction if the population is smaller than 100,000
    if N >= 10000:  # For large populations, skip the correction
        n = int(n0)
    else:
        n = int((n0 * N) / (n0 + N - 1))

    if print_==1:
        print(f"Z-score: {z}")  # Debug Z-score
        print(f"Initial sample size (n0): {n0}")  # Debug n0
        print(f"Sample size with FPC: {n}")  # Debug final sample size
    
    sample = df.sample(n=n, random_state=42)
    
    if len(new_column_name)==0:
        return sample 

    else:
        sample_index = sample.index
        df[new_column_name] = 0
        df.loc[sample_index, new_column_name] = 1
        return df


def ColumnStatisticalReview(df,
                            column_name,
                            partitions=10,
                            top_x_records=10,
                            exclude_blanks_from_segments=1,
                            exclude_zeroes_from_segments=1,
                            print_warning=False):

    '''
    Function to Conduct a Simple Statistical Review of a Column, Including Understanding the positional distribution
    of values. 

    Parameters:
        column_name (str): Name of Column

        partitions (int): Number of partitions to include (Decile 10)

        exclude_blanks_from_segments (int): Binary Flag, whether to exclude Blank Values from Segment determination.
        If blank values are excluded it gives a better representation for the members of the set, however it might 
        provide a misleading representation of the population.

        exclude_zeroes_from_segments (int): As above, with respect to 0 values. Is processed after exclude_blanks, as
        such it can include both blanks and true 0 values. 
        
    Returns:
        DataFrame
        
    Date Created:
    Date Last Modified: August 18, 2025
    Added Print Warning. 
    Moved to DFStatisticalReview

    '''

    temp_dict = {}
    
    is_numeric = pd.api.types.is_numeric_dtype(df[column_name])
    
    if is_numeric:
        temp_dict['SUM'] = df[column_name].sum()
        temp_dict['MEAN'] = df[column_name].mean()
        temp_dict['STD_DEV'] =  df[column_name].std()
        temp_dict['MEDIAN'] = df[column_name].median()
        temp_dict['MAX'] = df[column_name].max()
        temp_dict['MIN'] = df[column_name].min()
        
    temp_dict['TOTAL_RECORDS'] = len(df)
    temp_dict['UNIQUE_RECORDS'] = len(df.drop_duplicates(column_name))
    temp_dict['NA_RECORDS'] = len(df[df[column_name].isna()])
    temp_dict['NULL_RECORDS'] = len(df[df[column_name].isnull()])
    
    if is_numeric:
        temp_dict['ZERO_RECORDS'] = len(df[df[column_name]==0])
        temp_dict['NON_ZERO_RECORDS'] = len(df[df[column_name]!=0])    

    temp_df = pd.DataFrame(temp_dict.values(),index=temp_dict.keys(),columns=[column_name])
    
    if temp_dict['TOTAL_RECORDS']==len(df[df[column_name].isnull()]):
        return temp_df
    
    
    try:
        # Add top X records Based on Frequency
        if top_x_records>0:
            top_instances = pd.DataFrame(df[column_name].value_counts(dropna=False).head(top_x_records)).reset_index().rename(columns={column_name:'count','index':column_name})
            if len(top_instances)>0:
                top_instances[column_name] = top_instances.apply(lambda row: f"Value: {row[column_name]}, Frequency: {row['count']}", axis=1)
                top_instances['index'] = [f"Top {x+1}" for x in range(len(top_instances[column_name]))]
                top_instances = top_instances.drop('count',axis=1).set_index('index')
                temp_df = pd.concat([temp_df,top_instances])

        if (partitions>0)&(pd.api.types.is_numeric_dtype(df[column_name]))&(temp_dict['UNIQUE_RECORDS']>1):
            segment_df = ColumnPartitioner(df=df,
                                           column_name=column_name,
                                           partitions=partitions,
                                           exclude_blanks=exclude_blanks_from_segments,
                                           exclude_zeros=exclude_zeroes_from_segments,
                                           return_value='')
            seg_val_df = ColumnPartitioner(df=df,
                                               column_name=column_name,
                                               partitions=partitions,
                                               exclude_blanks=exclude_blanks_from_segments,
                                               exclude_zeros=exclude_zeroes_from_segments,
                                               return_value='agg_value').rename(columns={'VALUE':column_name})
            return pd.concat([temp_df,segment_df.T,seg_val_df])
    except:
        if print_warning:
            print(f'Could not Calculate Partitions for {column_name}')
        pass
            
    return temp_df

def DFStatisticalReview(df,
                        file_name=None,
                        print_=0,
                        time_check=20,
                        partitions=10,
                        top_x_records=10,
                        exclude_blanks_from_segments=1,
                        exclude_zeroes_from_segments=1):
    
    '''
    Function to Apply ColumnStatisticalReview to a entire Dataframe.
    
    Parameters:
        df(Dataframe): DataFrame to apply entire dataframe to ColumnStatisticalREview Function
        file_name(Str): Optional Argument to produce a .csv File OUtput
        print(bin): Optional Argument to generate a summary note of how long individual refresh took.
        time_check: Optional Argument to place Opt out to prevent program from Timing out. If you pass, the program will add
        5 seconds each time between prompting, based on the understanding that we want to minimize hte number of promptings
        and if you were willing to wait the previous time, then waiting 5 more seconds is likely.
    
    Returns:
        DataFrame
        (Optional .csv File in default directory, can also include specific path in file name)
        
    '''
    
    final_df = pd.DataFrame()
    
    for column in df.columns:
        start_time = timeit.default_timer()
        temp_df = ColumnStatisticalReview(df,
                                          column,
                                          partitions=partitions,
                                          top_x_records=top_x_records,
                                          exclude_blanks_from_segments=exclude_blanks_from_segments,
                                          exclude_zeroes_from_segments=exclude_zeroes_from_segments)
        
        elapsed_time = timeit.default_timer() - start_time
        final_df = pd.concat([final_df,temp_df],axis=1)
        time_check = PauseProcess(elapsed_time,time_check,column)
        if print_==1:
            print(f'Elapsed time to process {column}:{timeit.default_timer() - start_time:,.2f}')
                    
    if file_name:
        final_df.to_csv(f"{file_name}.csv")
        
    return final_df.T
