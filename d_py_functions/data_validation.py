import pandas as pd
import numpy as np
import math

import sys
import os
if os.getcwd().find('Users/derekdewald/Doc')!=-1:
    sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")
    from feature_engineering import binary_complex_equivlance
else:
    sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\BEEM_PY\\')
    sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\d_py_functions\\')
    from FeatureEngineering import BinaryComplexEquivlancey as binary_complex_equivlance

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

    date_created:20-Jan-26
    date_last_modified: 20-Jan-26
    classification:TBD
    sub_classification:TBD
    usage:
        
        
        
    
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

    date_created:20-Jan-26
    date_last_modified: 20-Jan-26
    classification:TBD
    sub_classification:TBD
    usage:
    
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

    Parameters:
        df(df): Dataframe

    Returns:
        Something

    date_created:20-Jan-26
    date_last_modified: 20-Jan-26
    classification:TBD
    sub_classification:TBD
    usage:
    
    
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



def column_segmenter(df,
                     column_name,
                     new_column_name=None,
                     bin_list=[],
                     number_segments=9,
                     force_segmentation=True,
                     leading_text="$",
                     format_=",.2f",
                     print_=True,
                     return_value=''):

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
        bin_list (list): LIst of predetermined Bins where User wants to define themselves.
        number_segments(list): Predetermined and requested listing from User.
        force_segmentation(bool): Item to force the return of edges as defined in function by removing duplication in DF
        leading_text (str): Optional Text to be added into Text Segment before Number is presented
        format_(str): Optional f-string based formating to apply to number
        print_(bool): Optional Print statement so you can easily see the brackets in print as created.

        return_value(str): Indication of what user would like in return. ("",bin_list,segment_dict)
            
    Returns:
        Contingent on Return Value.
            If "" then it updates df and returns nothing
            if return_value == 'bin_list', it returns a List of the Bin Position
            if return_value == 'segment_dict', it returns a dict of the index name and bin value, with creating a column in DF
        
    TO BE DEVELOPED
    
         impute_blanks=False,
         remove_zeros=False,

        Incoporated Changes from WorkFile, which removed counting of Min/Max and applied better logic.
         
    date_created:20-Jan-26
    date_last_modified: 01-Feb-26
    classification:TBD
    sub_classification:TBD
    usage:
        
    '''


    if not new_column_name:
        new_column_name = f"TEXT_SEGMENT"

    if force_segmentation:
        temp = df.drop_duplicates(column_name)[[column_name]]
    else:
        temp = df[[column_name]]

    # Determine bin_list if not provided
    if len(bin_list) == 0:
        bin_list = get_segments(temp,
                                column_name=column_name,
                                number_segments=number_segments)[1:]
        print(bin_list)
    else:
        number_segments = len(bin_list)

    # Return only bin_list if requested
    if return_value == "bin_list":
        return bin_list

    # Assign segments using searchsorted
    seg = np.searchsorted(bin_list, df[column_name], side="right")
    df['SEGMENT_INDEX'] = seg

    # Create a Dictionary to insure Consistency in Application
    segment_dict = {
        idx: edge for idx, edge in enumerate(bin_list)
    }
    
    # Used to Insure that a GT Value exists where highest Bin is not Highest Value
    segment_dict[len(bin_list)] = None
    
    if return_value=='segment_dict':
        return segment_dict

    else:
        segment_to_text(df,
                        segment_dict=segment_dict,
                        value_col=column_name,
                        index_col='SEGMENT_INDEX',
                        new_column_name=new_column_name,
                        leading_text=leading_text,
                        format_=format_,
                        print_=print_)
        


def segment_to_text(df,
                    segment_dict,
                    value_col='DIFFERENCE',
                    index_col='INDEX_SEGMENT',
                    new_column_name="TEXT_SEGMENT",
                    leading_text="$",
                    format_=",.2f",
                    print_=True):
    """
    Function to Create a Text Based Segmentation in a dataframe, based off of column_segmntation, which creates the 
    index. Input is a Dictionary, to insure consistency, as there where some logic based issues. 
    
    Parameters:
        df(df):
        segment_dict(dict):
        
        
        leading_text (str): Optional Text to be added into Text Segment before Number is presented
        format_(str): Optional f-string based formating to apply to number
        print_(bool): Optional Print statement so you can easily see the brackets in print as created.
    
    Returns: 
        Inputs Column into existing DataFrame

    date_created:20-Jan-26
    date_last_modified: 01-Feb-26
    classification:TBD
    sub_classification:TBD
    usage:
    
    
    """

    mapping = {}

    segment_indices = sorted(segment_dict.keys())

    for seg_index in segment_indices:
        edge = segment_dict[seg_index]

        # --- CASE 1: LOWER EDGE (first segment) -------------------
        if seg_index == 0:
            mapping[seg_index] = (
                f"{seg_index+1}) LT {leading_text}{float(edge):{format_}}"
            )

        # --- CASE 2: UPPER EDGE (GT last bin) ----------------------
        elif edge is None:
            last_edge = segment_dict[seg_index - 1]
            mapping[seg_index] = (
                f"{seg_index+1}) GT {leading_text}{float(last_edge):{format_}}"
            )

        # --- CASE 3: MIDDLE BINS ----------------------------------
        else:
            prev_edge = segment_dict[seg_index - 1]
            mapping[seg_index] = (
                f"{seg_index+1}) GTE {leading_text}{float(prev_edge):{format_}}, "
                f"LT {leading_text}{float(edge):{format_}}"
            )

    # Apply mapping directly
    df[new_column_name] = df[index_col].map(mapping)

    if print_:
        display(df[[new_column_name, value_col]].groupby(new_column_name)
              .agg(['mean','count','max','min']).reset_index())
