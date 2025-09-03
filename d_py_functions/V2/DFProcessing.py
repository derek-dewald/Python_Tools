## File Description: A more generalized Libary, including simple tricks, shortcuts and helpful functions for reviewing, summarizing or understanding. More Information, and less Procedural related. EDA for standardized processes.

from itertools import product,permutations,combinations
import pandas as pd
import numpy as np
import itertools

import sys
sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\BEEM_PY\\')
sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\d_py_functions\\')

from FeatureEngineering import BinaryComplexEquivlancey
    
def FilterDataframe(df,
                    binary_include={},
                    binary_exclude={},
                   ):
    
    temp_df = df.copy()
    
    if len(binary_include)!=0:
        for key, value in binary_include.items():
            temp_df = temp_df[temp_df[key]==value].copy()
            
    if len(binary_exclude)!=0:
        for key, value in binary_exclude.items():
            temp_df = temp_df[temp_df[key]!=value].copy()

    return temp_df

def SummarizedDataSetforBITool(df, dimensions, metrics):
    """
    Builds a DataFrame with all combinations of ALL-level rollups 
    across the specified dimensions and metrics.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        dimensions (list of str): Dimension column names.
        metrics (list of str): Metric column names to aggregate.

    Returns:
        pd.DataFrame: Aggregated DataFrame with 'ALL' rollups.
    """
    result_frames = []
    
    available_metrics = [x for x in metrics if x in df.columns]

    for r in range(len(dimensions) + 1):
        for dims in itertools.combinations(dimensions, r):
            group_cols = list(dims)
            
            # Aggregate metrics with or without groupby
            if group_cols:
                agg_df = df.groupby(group_cols, dropna=False)[available_metrics].sum().reset_index()
            else:
                # Grand total (ALL for all dims)
                sums = df[available_metrics].sum().to_frame().T
                agg_df = sums
                for col in dimensions:
                    agg_df[col] = 'ALL'

            # Fill missing dimension columns with 'ALL'
            for col in dimensions:
                if col not in group_cols:
                    agg_df[col] = 'ALL'

            # Ensure consistent column order
            agg_df = agg_df[dimensions + available_metrics]
            result_frames.append(agg_df)

    final_df = pd.concat(result_frames, ignore_index=True)
    return final_df

def DFColumnCompare(df,df1,return_df=False):
    
    '''
    
    Function to Compare Column Values contained within 2 Dataframes, Identifying Equivalency and Variance.
    
    Parameters:
        df(dataframe): Dataframe
        df1(dataframe): DataFrame
        
    Returns:
        Multiple Dataframes
        
    Date Created:
    Date Last Modified: August 18, 2025
    Moved to DFProcessing. Added Option to Export DF, not mandatory.
    
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
    print(f"Columns Unique to DF : {prop_dict['cols_missing_df']}\n")
    
    print(f"Total Columns in DF1 Not in DF : {len(missingdf)}")
    print(f"Columns Unique to DF1 : {prop_dict['cols_missing_df1']}")
    
    if return_df:
        return missingdf,missingdf1

def MergeIdenticalDF(df,df1,primary_key_list,column_distinction='_',include_merge=False):
    
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
    
    
    Date Created: August 18, 2025
    Date Last Modified: 
    
    
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
    
    DFColumnCompare(df,df1)
    
    df = df.copy()
    df1 = df1.copy()
    
    # Rename Columns in Table 1 for consistency
    df1 = df1.rename(columns={x:f"{x}{column_distinction}" for x in df1.columns if x not in primary_key_list})
    
    # Merge Table 1 into Table 0
    temp_df= df.merge(df1,on=primary_key_list,how='outer',indicator=include_merge)
    
    if include_merge:
        print(temp_df['_merge'].value_counts())
    
    return temp_df

def TranposeDF(df, index, columns=None):
    '''
    Transposes a non-time-series DataFrame from wide to long format by melting specified columns.

    This is especially useful for flattening columns into a single column to support tools 
    like Power BI, where long format enables dynamic pivoting and aggregation.

    Parameters:
        df (DataFrame): The input pandas DataFrame.
        index (list): Columns to retain as identifiers (will remain unchanged).
        columns (list): Columns to unpivot into key-value pairs.

    Returns:
        DataFrame: A long-format DataFrame with 'variable' and 'value' columns.


    '''
    if not columns:
        columns = [col for col in df.columns if col not in index]
    return df.melt(id_vars=index, value_vars=columns)   




def ColumnDQComparison(df,
                       column_name,
                       column_name1=None,
                       additional_filter=None,
                       column_distinction='_',
                       bracketing=[-10000,-1000,-100,-1,0,1,100,1000,10000]):
    
    '''
    
    Function which takes a dataframe with 2 Columns which are identical and attempts to Compare.
    Designed with the intention of comparing,  BALANCE, BALANCE_, however can explicitly utilize column_name1 to override.

    Function DfDqComparison takes this function and applies it to an Entire Dataframe
    
    Parameters:
        column_name (str):
        column_name1 (str):
        additional_filter (str): Default parameter to distinguish combined dataframes, also used in MergeAndRenameColumnsDf
        bracketing(list): Value to create Distintion when Calculated Difference between columns is numeric.
    
    Returns:
        Dictionary of 3 Dataframes, Account, Summary and Group By.
    
    Values:
    
    
    Date Created: August 21, 2025
    Date Last Modified:
    


    '''
    if not column_name1:
        column_name1 = f"{column_name}{column_distinction}"    
    
    # Change Names of Individual Columns to Something Generic so datasets can be Concatenated.
    temp_df = df.rename(columns={column_name:'DF',column_name1:'DF1'}).copy()
    
    output_dict = {}
    
    temp_df['COLUMN_NAME'] = column_name
    
    BinaryComplexEquivlancey(temp_df,'DF','DF1','VALUES_EQUAL')
    
    temp_df['VALUES_NOT_EQUAL'] = np.where(temp_df['VALUES_EQUAL']==0,1,0)
    temp_df['NULL_RECORD_DF'] = np.where(temp_df['DF'].isnull(),1,0)
    temp_df['NULL_RECORD_DF1'] = np.where(temp_df['DF1'].isnull(),1,0)
    
    try:
        temp_df['DIFFERENCE'] = temp_df['DF'].fillna(0)-temp_df['DF1'].fillna(0)
    except:        
        temp_df['DIFFERENCE'] = 0
        
    try:
        BracketColumn(temp_df,'DIFFERENCE','DIFF_SEGMENT',bracketing)
    except:
        temp_df['DIFF_SEGMENT'] = 'Could Not Calculate'
    
    # Removed Column Partitioner as it wasn't being Used.
    
    temp_df1 = temp_df.copy()
    temp_df1['RECORD_COUNT']=1
    
    if additional_filter:
        output_dict['groupby_df'] = temp_df1[[additional_filter,'COLUMN_NAME','DF','DF1','RECORD_COUNT','VALUES_EQUAL','VALUES_NOT_EQUAL','NULL_RECORD_DF','NULL_RECORD_DF1']].groupby([additional_filter,'COLUMN_NAME','DF','DF1'],dropna=False).sum().sort_values('VALUES_EQUAL',ascending=False).head(20).reset_index()
        
    else:
        output_dict['groupby_df'] = temp_df1[['COLUMN_NAME','DF','DF1','RECORD_COUNT','VALUES_EQUAL','VALUES_NOT_EQUAL','NULL_RECORD_DF','NULL_RECORD_DF1']].groupby(['COLUMN_NAME','DF','DF1'],dropna=False).sum().sort_values('VALUES_EQUAL',ascending=False).head(20).reset_index()
    
    if additional_filter:
        summary_df = pd.DataFrame()
        for value in temp_df[additional_filter].unique():
            temp = temp_df[temp_df[additional_filter]==value]
            value_dict = {
                'Total Combined Records':len(temp),
                'Values Equal':temp['VALUES_EQUAL'].sum(),
                'Values Not Equal':len(temp[temp['VALUES_EQUAL']==0]),
                'Percent Values Equal': (temp['VALUES_EQUAL'].sum()/len(temp))*100,
                'Null Records DF':temp['NULL_RECORD_DF'].sum(),
                'Null Records DF1':temp['NULL_RECORD_DF1'].sum()}
            
            try:
                value_dict['Total Difference']=temp['DIFFERENCE'].sum()
            except:
                value_dict['Total Difference']=0
                
            sum_df = pd.DataFrame(value_dict.values(),index=value_dict.keys(),columns=[column_name]).T.reset_index().rename(columns={'index':"COLUMN_NAME"})
            sum_df[additional_filter] = value
            summary_df = pd.concat([summary_df,sum_df])
    else:
        value_dict = {
            'Total Combined Records':len(temp_df),
            'Values Equal':temp_df['VALUES_EQUAL'].sum(),
            'Values Not Equal':len(temp_df[temp_df['VALUES_EQUAL']==0]),
            'Percent Values Equal': (temp_df['VALUES_EQUAL'].sum()/len(temp_df))*100,
            'Null Records DF':temp['NULL_RECORD_DF'].sum(),
            'Null Records DF1':temp['NULL_RECORD_DF1'].sum()}
        
        try:
            value_dict['Total Difference']=temp['DIFFERENCE'].sum()
        except:
            value_dict['Total Difference']=0
            
        summary_df = pd.DataFrame(value_dict.values(),index=value_dict.keys(),columns=[column_name]).T.reset_index().rename(columns={'index':"COLUMN_NAME"})
        
    output_dict['summary_df'] = summary_df
    output_dict['account_df'] = temp_df
    
    return output_dict

def DfDqComparison(df,
                   primary_key_list,
                   additional_filter,
                   column_distinction='_',
                   bracketing=[-10000,-1000,-100,-1,0,1,100,1000,10000],
                   file_name=None):
    
    '''
    Function to Apply ColumnDQComparison against DataFrame.
    Assumes you start with a Dataframe with Multiple Columns Different only by Column Distinction.
    
    Parameters:
        df (DataFrame)
        primary_key_list (list): List of Primary Keys, which are REMOVE from comparison Loop.
        additional_filter (str): Filter Used to Create a distinct Dimension. Currently DOES NOT accept List
        column_distinction (str): String which is expected to compare Columns. Added as default with MergeIdenticalDF
        bracketing (list): Numbers which can be used to Calculate a Bracketed difference Column in COmparison
        file_name (str): If Included, it will generate Excel Copies (Excel Used as CSV had issues uploading to DF)
        
    Return:
        DataFrame of Groupby, Account and Summary calculations.
        
        Account: Listing of All Account Values, with Calculations
        Summary: A summary Calculation Speaking to Overall Comparison
        Groupby: List of Equivalent Values, to compare Material Record Change/Consistency
    
    Date Created: August 21, 2025
    Date Last Modified: 
    
    
    '''
    
    df = df.copy()
    
    # Only Need to Test Common Records Can do a Simple Dataframe Analysis on Non Common Records.
    
    #Iterate Through All Columns in Common to create Final Values.
    
    account_df = pd.DataFrame()
    groupby_df =  pd.DataFrame()
    summary_df = pd.DataFrame()

    for column_name in [x for x in df.columns if (x not in primary_key_list)&(x[-1]!=column_distinction)]:
        column_name1 = f"{column_name}{column_distinction}"
        try:
            temp_dict = IdenticalColumnDQValidation(df=df[['ACCTNBR',additional_filter,column_name,column_name1]],
                                                    column_name=column_name,
                                                    additional_filter=additional_filter,
                                                    bracketing=bracketing)
    
            account_df = pd.concat([account_df,temp_dict['account_df']])
            summary_df = pd.concat([summary_df,temp_dict['summary_df']])
            groupby_df = pd.concat([groupby_df,temp_dict['groupby_df']])
            
        except:
            print(f'Could Not Compute: {column_name}') 
            
    if file_name:
        account_df.to_csv(f"{file_name}_ACCOUNT.csv",index=False)
        summary_df.to_csv(f"{file_name}_SUMMARY.csv",index=False)
        groupby_df.to_csv(f"{file_name}_GROUPBY.csv",index=False)

    return account_df,summary_df,groupby_df


def PartitionDF(df, index_, func,include_all=True,*args,**kwargs):
    '''
    Function to automate the task of Applying a Specific Function on a dataframe which has been Segregatted based on 
    a combination of Columns as defined in the argument index_
    
    An example, where the function was originally designed: PartitionDF(temp_df1,['ENTITY','MEMBER_DURATION'],DFStatisticalReview) 
    
    Noting that to generate 32 distinct subsets on 50K Records and 124 Columns 6.25M records is 60 Seconds.
    
    Parameters:
        df (DataFrame): 
        index_ (list): List of Columns which will be included in Group By 
        func (Function): Function which will be applied to the function
        include_all (Bol): Option to include a Aggregated Calculation in Dataset, represented of calculation for All
        *args: Arguements for inclusion in Function
        **kwargs: Keyword Arguments for inclusion in Function (if required)
        
        
    Returns:
        Dictionary with Unique DataFrame and Indicataors for the record place and [FUNC] output.
        
    Date Created: August 20, 2025
    Date Last Modified: August 22, 2025. Added ValueError to Stop when DF missing Index Values, which could not be processed
    
    '''
    
    # Can not Partion Null Values. Dangerous to Assume. Return Error, force User to Solve Data Problems
    
    for record in index_:
        if len(df[df[record].isnull()])>0:
            raise ValueError(f'Null Value found in Index {record}, please review data. Fill or Remove Value')
    
    partitioned_list = []
    
    # Full dataset (All for all index columns)
    
    if include_all:
    
        temp_ = {value: 'All' for value in index_}
        temp_['DF'] = df
        temp_func = func(df).reset_index().rename(columns={'index': "COLUMN_NAME"})
        for column in index_:
            temp_func[column] = 'All'
        temp_['FUNC'] = temp_func
        partitioned_list.append(temp_)
    else:
        temp_ = dict()

    # Generate all partial combinations of index columns
    for i in range(1, len(index_)):
        for combo in itertools.combinations(index_, i):
            gb = df.groupby(list(combo))
            for selection in gb.groups.keys():
                temp_df = gb.get_group(selection).copy()
                if include_all:
                    temp_ = {col: 'All' for col in index_}
                if isinstance(selection, tuple):
                    for col, val in zip(combo, selection):
                        temp_[col] = val
                else:
                    temp_[combo[0]] = selection

                temp_['DF'] = temp_df
                temp_func = func(temp_df).reset_index().rename(columns={'index': "COLUMN_NAME"})
                for col in index_:
                    temp_func[col] = temp_[col]
                temp_['FUNC'] = temp_func
                partitioned_list.append(temp_)
    
    # Fully segmented combinations
    gb = df.groupby(index_)
    for selection in gb.groups.keys():
        temp_df = gb.get_group(selection).copy()
        temp_ = dict(zip(index_, selection))
        temp_['DF'] = temp_df
        temp_func = func(temp_df,*args,**kwargs).reset_index().rename(columns={'index': "COLUMN_NAME"})
        for col_name, col_value in zip(index_, selection):
            temp_func[col_name] = col_value
        temp_['FUNC'] = temp_func
        partitioned_list.append(temp_)

    return partitioned_list

def ColumnPartitioner(df,
                      column_name,
                      new_column_name='Partition',
                      new_value_column='Total Balance in Partion',
                      partitions=10,
                      exclude_blanks=1,
                      exclude_zeros=0,
                      return_value=''):
    '''
    Function to create partions from Float or INT column which returns the Upper Partion Bound for a Column in a DataFrame. 
    Inspired by the Decile Analysis, it quickly highlights the distribution of a given dataset.

    Args:
        partitions:
            Total Number of desired Partitions. Default 10 as a homage to DR and his love of the Decile Analysis.

        Exclude Blanks:
            Binary flag to determine whether null value records  are to be considered in the Analysis. If 1 then 
            they are excluded, otherwise, they are given a value of 0 and included. Note that this can Materially 
            Impact Distribution and Inference, so should be carefully considered.

        Exclude Zeros:
            Binary flag to determine whether 0 value records are to be considered in the analysis. If 1 then they are excluded,
            otherwise they are included. Note that this can Materially Impact Distribution and Inference, so should be carefully
            considered.

        Return Value:
            Value to be returned:
            default (""):       DF of Value at Individual Partition Locations
            list_index(list):   Returns list of Index Locations in Dataframe
            list_value(list):   List of Value at Individual Partition Locations 
            merge(df):          New Column in existing DF which is numerical value of segment which value belongs
            agg_value(df):      DF of Aggregate Value total Impact of Each Segment
            position_value(df)  DF of Position (Transposed Default DF) and agg_value dataframe.


        New Column Name:
            Name of New Column if original Partition is choosen. By Default, Parition is choosen.

    '''
    if partitions <2:
        return print('Requries a Minimum of 2 partitions, recommends no less than 3 partitions')

    # Make a copy to ensure no overwriting
    temp_df = df.copy()

    # Clean Dataset 
    if exclude_blanks ==1:
        blanks_removed = len(temp_df[temp_df[column_name].isnull()])
        #print(f"Blank Entries Removed: {blanks_removed}")
        temp_df = temp_df[temp_df[column_name].notnull()]
    else:
        temp_df[column_name] = temp_df[column_name].fillna(0)

    if exclude_zeros ==1:
        zeroes_removed = len(temp_df[temp_df[column_name]==0])
        #print(f"Zero Entries Removed: {zeroes_removed}")
        temp_df = temp_df[temp_df[column_name]!=0]

    column_list = temp_df[column_name].tolist()
    column_list.sort()
    length_of_df = len(column_list)
    break_point = math.ceil(length_of_df/partitions)

    if partitions >=length_of_df:
        return print(f'Sample Size insufficient to Warrant Calculation for column {column_name}, please review data')

    record_position = list(range(0,length_of_df,break_point))
    record_value = [column_list[x] for x in record_position]
    #print(record_value)


    # Parition Value DF

    partition_df = pd.DataFrame(record_value,index=[f"{new_column_name} {x+1}" for x in range(len(record_value))],columns=[column_name]).T

    if return_value == '':
        return partition_df
    elif return_value == 'list_value':
        return record_value
    elif return_value == 'list_index':
        return record_position
    elif (return_value == 'merge')|(return_value == 'agg_value')|(return_value=='position_value'):
        temp_df = temp_df.sort_values(column_name).reset_index(drop=True)
        temp_df[new_column_name] = np.searchsorted(record_position,temp_df.index,side='right')

        if (return_value == 'agg_value')|(return_value=='position_value'):
            agg_impact = temp_df[['Partition',column_name]].groupby('Partition').sum()[column_name].values
            agg_impact_df = pd.DataFrame(agg_impact,
                                         columns=['VALUE'],
                                         index=[f"{new_value_column} {x+1}" for x in range(len(agg_impact))])

            if return_value=='position_value':
                agg_impact_df.reset_index(drop='True',inplace=True)
                agg_impact_df['index'] = [f"{new_column_name} {x+1}" for x in range(len(agg_impact_df))]
                agg_impact_df.set_index('index',inplace=True)

                temp_df1 =  partition_df.T.merge(agg_impact_df,
                                                 left_index=True,
                                                 right_index=True,
                                                 how='left').rename(columns={'VALUE':"AGGREGATE_VALUE","VARIANCE":"PARTITION",})
                return temp_df1
            return agg_impact_df
        return temp_df









