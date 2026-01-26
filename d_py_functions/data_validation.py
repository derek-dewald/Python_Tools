import pandas as pd
import numpy as np
import math

import sys
#sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\BEEM_PY\\')
#sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\d_py_functions\\')
#from FeatureEngineering import BinaryComplexEquivlancey as binary_complex_equivlance

sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from feature_engineering import binary_complex_equivlance

def df_column_compare(df,df1,return_df=False):
    
    '''
    
    Function to Compare Column Values contained within 2 Dataframes, Identifying Equivalency and Variance.
    
    Parameters:
        df(dataframe): Dataframe
        df1(dataframe): DataFrame
        
    Returns:
        Multiple Dataframes
        

    date_created:18-Aug-25
    date_last_modified: 18-Aug-25
    classification:TBD
    sub_classification:TBD
    usage:
        Example Function Call
    
    '''
    
    prop_dict = {}
    
    cols_df =  set(df.columns.values)
    cols_df1 = set(df1.columns.values)
    
    
    prop_dict['cols_missing_df'] = [x for x in list(cols_df) if x not in cols_df1]
    prop_dict['cols_missing_df1']  = [x for x in list(cols_df1) if x not in cols_df]
    
    prop_dict['cols_same'] = [x for x in list(cols_df) if x in cols_df1]
    
    print(f"Total Columns in DF: {len(df.columns.values)}")
    print(f"Total Columns in DF1: {len(df1.columns.values)}")
    print(f"Total Columns in both Data Frames: {len(prop_dict['cols_same'])}\n")
    print(f"Shared Columns: {prop_dict['cols_same']}\n")
    
    missingdf = prop_dict['cols_missing_df']
    missingdf1  = prop_dict['cols_missing_df1']
    
    print(f"Total Columns in DF Not in DF1 : {len(missingdf1)}")
    print(f"Columns Unique to DF : {prop_dict['cols_missing_df1']}\n")
    
    print(f"Total Columns in DF1 Not in DF : {len(missingdf)}")
    print(f"Columns Unique to DF1 : {prop_dict['cols_missing_df']}")
    
    if return_df:
        return missingdf,missingdf1

def merge_identical_df(df,df1,primary_key_list,column_distinction='_',include_merge=False):
    
    '''
    Function Designed to Simplify the Joining of Two Identical Dataframes. 
    Created for the purposes of DQ Validaiton, where taking 2 identical Dataframes with similar records and near identical 
    columns.

    
    Parameters:
        df (dataframe)
        df1 (dataframe) 
        primary_key_list (list): List of Primary Key Values
        column_distinction (str): Value which will be added to Columns in DF1 to distinguish records once connsolidated
        include_merge (True/False): Value to identify whether inclusion or _merge field when combining columns
        
    Returns:
        DataFrame (Combined Columns)
    
    
    Date Created: 18-Aug-25
    date_last_modified: 18-Aug-25
    classification:TBD
    sub_classification:TBD
    usage:
        
    
    
    '''
    
    # Check to Insure that Primary Keys are Unique
    if len(df)==len(df.drop_duplicates(primary_key_list)):
        pass
    else:
        raise CustomError('DF is not unique on primary Key')
    
    if len(df1)==len(df1.drop_duplicates(primary_key_list)):
        pass
    else:
        raise CustomError('DF1 is not unique on primary Key')
    
    df_column_compare(df,df1)
    
    df = df.copy()
    df1 = df1.copy()
    
    # Rename Columns in Table 1 for consistency
    df1 = df1.rename(columns={x:f"{x}{column_distinction}" for x in df1.columns if x not in primary_key_list})
    
    # Merge Table 1 into Table 0
    temp_df= df.merge(df1,on=primary_key_list,how='outer',indicator=include_merge)
    
    if include_merge:
        print(temp_df['_merge'].value_counts())
    
    return temp_df

def get_decile(df, column_name, n):
    
    '''
    Function designed to Seperate Data into N equal segments.
    Differs from get_segments, which is about segmenting for the purposes of Creating Segments for Slicing, not utilizing the
    bottom position, changing the top position, and doing providing a safety net against duplication in Segment Duplication, 
    as when more than 10% of observations share a common record segmenting becomes problematic.
    
    parameters:
        df(df): Any DataFrame
        column_name(str): Column Name of Any Numeric Column
        
    returns: 
        List 
    
    date_created:20-Jan-26
    date_last_modified: 20-Jan-26
    classification:TBD
    sub_classification:TBD
    usage:
        bin_list = get_decile(df1,column_name='DIFFERENCE',number_segments=10)
    
    
    '''
    
    return (
        df[column_name]
        .sort_values()
        .quantile(np.linspace(1/n, 1, n), interpolation="higher")
        .tolist()
    )

def get_segments(df,
                 column_name,
                 number_segments):
    '''
    Function Designed to seperate a Dataframe Column into N Segments. 
    Note that it simply looks for the Position of the respective column in the Dataframe, it will return duplication, which can be a 
    problem for many functions, including NP.CUT

    Input Function into: column_segment
    
    Parameters:
        df(df): Any Dataframe
        column_name(str): Name of Column
        number_segments(int): Number of Segments to be Calculated. Note that to cut segments it is required to calculate N+1

    Returns:
        List

    date_created:19-Jan-26
    date_last_modified: 19-Jan-26
    classification:TBD
    sub_classification:TBD
    usage:
        bin_list = get_segments(df1,column_name='DIFFERENCE',number_segments=10)
        
    '''
    edges = df[column_name].quantile(np.linspace(0, 1, number_segments+2)).to_numpy().tolist()
    if len(edges) > 0:
        edges[-1] = edges[-1] + 1e-12    
    
    return edges

def column_segmenter(df,
                     column_name,
                     new_column_name=None,
                     bin_list=[],
                     number_segments=10,
                     force_segmentation=True,
                     right_edge_is_min=False,
                     left_edge_is_max=False,
                     leading_text="$",
                     format_=',.2f',
                    return_value='df_column'):
    
    '''
    
    Function which Takes a SINGLE column in a DataFrame and calculates the segments as designed by the appropriate size of
    the dataframe by number of segments, as an example, a Decile Analysis would include 10 Segments.

    NOTE: This is not a PERFECT EQUAL Segment Funcition. When segments Share Edges it is Problematic, specifically, in Dataframes which have a 
    median value which can include multiple segments, the mathematical representation can be problematic. There are multiple approaches to handle
    must be aware of what one is trying to accomplish and whether this function and the pre-determined process support exactly the desired end goal
    
    The issue with this is that Uniform datasets, or those which have a material number of common values will not have clean
    edges. As an example, Binary Data sets will have 2 records, Loan Balance will have a material number of 0 balances, 
    so if you look for 10 Edges, 5 of the 10 are likely to be 0. This causes issue when using the np and pd options. 
    
    Some assumptions and work arounds have been implemented, primary related to force_number_edges, you can force and it will 
    remove duplicates, guaranteeing a minimum number of edges, but as a result you will lose equality in segment size.
    
    Parameters:
        df(df): DataFrame
        column_name(str): Name of Column to be Analyzed.
        new_column_name(str): Name of column to be created in DF if a list of edges is not selected
        segments(list): Predetermined and requested listing from User.
        return_value(str): Indication of what user would like in return.
            df_column: New Column with Index Position of Segment/Edge
            segment_name: New Column with Text Description of Segment, for Visualization
            segments: List of Edge Positions
        number_edges(int): Number of distinct Edges to return
        force_segmentation(bool): Item to force the return of edges as defined in function by removing duplication in DF
        leading_text(str): Additional Leading Text to be Added when using segment_name
        leading_text(str): Desired Format of Value to be returned when using segment_name
        
        right_edge_is_min(bool): When Using Custom Edges, whether values in Column exist which are less than defined Minimum.
        left_edge_is_max(bool):  When Using Custom Edges, whether values in Column exist which are greated than defined Maximum
        

    TO BE DEVELOPED
    
         impute_blanks=False,
         remove_zeros=False,
        
    '''
    # If New Column Name is not defined, Create it.
    if not new_column_name:
        new_column_name = f"{column_name}_SEGMENT"
    
    # Whether you want to Force Edges or Not
    if force_segmentation:
        temp = df.drop_duplicates(column_name)[[column_name]]    
    else:
        temp = df[[column_name]]
        
    if len(bin_list)==0:
        bin_list = get_segments(temp,
                                 column_name=column_name,
                                 number_segments=number_segments)
    else:
        number_segments = len(bin_list)+1
        
    if return_value=='bin_list':
        return bin_list
    try:
        seg = np.searchsorted(bin_list, df[column_name], side="right") 
        df[new_column_name] = seg
    except:
        pass    

    return df,bin_list

def segment_to_text(df,
                    bin_list,
                    column_name=None,
                    new_column_name=None,
                    leading_text="$",
                    format_=',.2f'):
    
    '''
    Function which takes a Dataframe Column, which takes a Column Segment or Bin List and converts this to a text based
    definition, which can be more easily applied and read in Visualizations. 
    
    '''
    
    if not column_name:
        column_name = 'DIFFERENCE'
    if not new_column_name:
        new_column_name = 'DIFF_TEXT_SEGMENT'
    
    # Test to see if Lower Bound is actual Limit.
    if (df[column_name].lt(bin_list[0])).any():
        right_is_min=False
    else:
        right_is_min=True
    
    if (df[column_name].gt(bin_list[-1])).any():
        left_is_max=False
    else:
        left_is_max=True
        
    count = 0
    
    # Create Mapping Dict
    mapping_dict = {}
    
    for value in bin_list:
        if count == 0:
            # Increase Count by 1 for Lower Bound Hard Edge
            if right_is_min:
                count +=1
                count_=0
                le_max_adj = -1
            else:
                count_ =1
                le_max_adj=1
                
            mapping_dict[count] = f"1) LT {bin_list[count]:{format_}}"
        
        elif count == len(bin_list)&left_is_max:
            mapping_dict[count-1+count_] = f"{count+le_max_adj}) GT {bin_list[-1]:{format_}}"
            
        elif count == len(bin_list):
            mapping_dict[count-1+count_] = f"{count+le_max_adj}) GT {bin_list[-2]:{format_}}"
            
        else:
            mapping_dict[count] = f"{count+count_}) GT {bin_list[count-1]:{format_}}, LT {bin_list[count]:{format_}}"
    
        # Increase Count for Iteration
        count+=1
            
    df[new_column_name] = df[new_column_name].map(mapping_dict)
    
def round_up_power10(series, infer_neg_vals=True):
    """
    Function which takes a Series (or Array) and determines the base 10 value of the Logrithm.
    
    This function is meant to support with EDA/Visualization/ Validation tasks, but trying to help standardize and 
    simplify how Segments are Presented.
    
    Parameters:
        series(pd.Series): Series or Array.
        infer_neg_vals(bool): Logical condition to determine whether values of 0 or 
        
    Returns:
        List
        
        
    
    """
    x = s.astype(float).copy()

    # Prepare result, keep NaNs
    result = pd.Series(np.nan, index=s.index, dtype=float)

    # Masks
    pos = x > 0
    neg = x < 0
    zero = x == 0

    # Positive values: standard round-up to next power of 10
    if pos.any():
        e_pos = np.ceil(np.log10(x[pos])).astype(int)
        result.loc[pos] = 10.0 ** e_pos

    # Zero stays zero
    if zero.any():
        result.loc[zero] = 0.0

    # Negative values
    if neg.any():
        if infer_neg_vals:
            # Compute using abs, then apply the negative sign
            e_neg = np.ceil(np.log10(np.abs(x[neg]))).astype(int)
            result.loc[neg] = -(10.0 ** e_neg)
        else:
            result.loc[neg] = 0.0
            
    final_list = pd.Series(result).dropna().unique().tolist()
    final_list.sort()
    
    return final_list[1:-1]


def column_comparison(df,
                      column_name,
                      column_name1,
                      metric_name=None,
                      bin_list=[],
                      retain_columns=[],
                      number_segments=10,
                      force_segmentation=True):
    
    '''
    Function which takes a dataframe with 2 Columns which are identical and compare the values for equivalency.
    In addition to comparing, it attempts to calculate the value differences and report on statistical properties 
    of the differences. Uses a number of input functions, including: 
        Binary Complex Equivalency, column_segmenter, 
    
    Parameters:
        column_name (str):
        column_name1 (str):
    
    Returns:
    
    '''
    temp_df = df[[column_name,column_name1]].copy()
    
    if not metric_name:
        metric_name = f"{column_name}"
    
    temp_df['METRIC_NAME'] = metric_name
    
    binary_complex_equivlance(temp_df,column_name,column_name1,'VALUES_EQ')
    
    temp_df['VALUES_NE'] = np.where(temp_df['VALUES_EQ']==0,1,0)
    temp_df['NULL_DF'] = np.where(temp_df[column_name].isnull(),1,0)
    temp_df['NULL_DF1'] = np.where(temp_df[column_name1].isnull(),1,0)

    try:
        temp_df['DIFFERENCE'] = temp_df[column_name].fillna(0)-temp_df[column_name1].fillna(0)
        
        # If User Hard Codes Bins, then use them, otherwise calculate based on Extreme Values
        if len(bin_list)>0:
            temp_df,bin_list = column_segmenter(temp_df,
                                                column_name='DIFFERENCE',
                                                new_column_name='DIFF_INDEX_SEGMENT',
                                                bin_list=bin_list,
                                                force_segmentation=force_segmentation,
                                                return_value='segment_name')
        
        else:   
            temp_df,bin_list = column_segmenter(temp_df,
                                                column_name='DIFFERENCE',
                                                new_column_name='DIFF_INDEX_SEGMENT',
                                                number_segments=number_segments,
                                                force_segmentation=force_segmentation,
                                                return_value='segment_name')
            
        bin_df = pd.DataFrame({"SEGMENT": [f"SEGMENT{x}" for x in range(len(bin_list))],"THRESHOLD": bin_list})
        bin_df['METRIC'] = column_name
        
    
        # Create a Text Based Segment Definition
        segment_to_text(temp_df,bin_list,'DIFFERENCE')
                
        gb = temp_df[['DIFFERENCE','DIFF_TEXT_SEGMENT']].groupby('DIFF_TEXT_SEGMENT').agg(
            SEG_COUNT=('DIFFERENCE','count'),
            SEG_SUM=('DIFFERENCE','sum'),
            SEG_MEAN=('DIFFERENCE','mean'),
            SEG_STD=('DIFFERENCE','std'),
            SEG_MAX=('DIFFERENCE','max'),
            SEG_MIN=('DIFFERENCE','min')).reset_index()

        gb['METRIC'] = column_name
        
    except:
        temp_df['DIFFERENCE'] = 0
        temp_df['DIFF_INDEX_SEGMENT'] = 'Could Not Calculate'
        temp_df['DIFF_TEXT_SEGMENT'] = 'Could Not Calculate'
        bin_df = pd.DataFrame()
        gb = pd.DataFrame()
    
    # Output Dict
    output_dict = {}
    output_dict['bins'] = bin_df
    output_dict['segmentation'] = gb
    output_dict['summary'] = temp_df.groupby('METRIC_NAME',as_index=False).agg(
        VALUES_EQ_SUM=("VALUES_EQ",'sum'),
        VALUES_NE_SUM=("VALUES_NE",'sum'),
        NULL_DF_SUM=("NULL_DF",'sum'),
        NULL_DF1_SUM=("NULL_DF1",'sum'),
        DIFFERENCE_SUM=("DIFFERENCE",'sum'),
        DIFFERENCE_MEAN=("DIFFERENCE",'mean'),
        DIFFERENCE_STD=("DIFFERENCE",'std'),
        DIFFERENCE_MAX=("DIFFERENCE",'max'),
        DIFFERENCE_MIN=("DIFFERENCE",'min')
    )

    if len(retain_columns)>0:
        temp_df = pd.concat([df[retain_columns],temp_df],axis=1)
    
    temp_df.rename(columns={column_name:'VAL_COL1',column_name1:'VAL_COL2'},inplace=True)
    output_dict['df'] = temp_df
        
    return output_dict


def df_column_comparison(df,
                         column_list=[],
                         column_distinction='_',
                         retain_columns=[],
                         bin_list=[],
                         number_segments=10,
                         force_segmentation=True):
    '''
    
    Function which applies column_comparison across an Entire Dataframe.
    It expands the original in 2 distinct ways, 
    1) It looks to iterate across the entire Dataframe.
    2) It looks to consolidate on the Group By Level, although it only applies the calculation of EQ,GT,LT, once to improve 
    process and reduce duplication.
    
    ############## 
    what happens when a column doesn't have a corresponding Value?
    # Need to Use Bin Segments. But How?
    
    ##############
    
    
    '''
    
    # Give User the Opportunity to define the Columns they want to iterate through. If Blank, infer.
    # If Inferring Need to know which columns have actual Values.

    if len(column_list)==0:
        
        df_cols = [x for x in df.columns if x not in retain_columns]
        column_list = [x for x in df_cols if x[-len(column_distinction)]!=column_distinction]
        #list2 = [x for x in df_cols if x[-len(column_distinction)]==column_distinction]
        
    # Iterate through AVAILABLE Columns and generate a dataset for all subsequent Calculations.
    # Create Empty DF to Fill
    
    output_dict = {
        'df':pd.DataFrame(),
        'segmentation':pd.DataFrame(),
        'bins':pd.DataFrame(),
        'summary':pd.DataFrame(),
    }
    
    temp_df = pd.DataFrame()
    
    for column in column_list:
        temp_dict = column_comparison(df=df,
                                        column_name=column,
                                        column_name1=f"{column}{column_distinction}",
                                        bin_list=bin_list,
                                        retain_columns=retain_columns,
                                        number_segments=number_segments,
                                        force_segmentation=force_segmentation)
        
        # Iterate through Dictionary for all Files 
        for name in ['df','segmentation','bins','summary']:
            output_dict[name] = pd.concat([output_dict[name],
                                           temp_dict[name]])
    return output_dict


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