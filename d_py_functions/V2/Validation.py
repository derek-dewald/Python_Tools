import pandas as pd
import numpy as np
import datetime
from scipy.stats import norm

import sys
sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\BEEM_PY\\')
sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\d_py_functions\\')


from DateFunctions import generate_day_list
from DB_CONNECTIONS import TIME_SQL
from FeatureEngineering import BinaryComplexEquivlancey

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
                            exclude_zeroes_from_segments=1):

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
        pass
            
    return temp_df
    
def ColumnStatisticalCompare(df,df1,column_name1,column_name2=None):
    
    '''
    Function to Compare the Values contained within 2 Columns.
    
    
    Parameters:
        df (DataFrame): Dataframe of Primary Dataset
        df1 (DataFrame): Dataframe of ALternative Dataset, if both values in same dataframe already, value to be None and 
        column_name2 should be included
        column_name1 (str): Column Name to be compared ('Should be the same')
        column_name2 (str): If Column is Contained in the SAME dataframe, then it WILL NOT have the same name, so must identify the name
        
    Returns:
        DataFrame Analyzing Differences

    '''
    
    a = ColumnStatisticalReview(df,column_name1).rename(columns={column_name1:f'{column_name1}_0'})
    
    
    if not df1:
        b = ColumnStatisticalReview(df,column_name2).rename(columns={column_name2:f'{column_name}_DF1'})
    else:
        b = ColumnStatisticalReview(df1,column_name1).rename(columns={column_name2:f'{column_name}_DF1'})

    temp_df = pd.concat([a,b],axis=1)
    temp_df[f'{column_name}_VAR_AMT'] = temp_df[f'{column_name}_DF'] - temp_df[f'{column_name}_DF1'] 
    temp_df[f'{column_name}_VAR_PERC'] = temp_df[f'{column_name}_VAR_AMT']/temp_df[f'{column_name}_DF1']
    temp_df[f'{column_name}_VAR_PERC'] = temp_df[f'{column_name}_VAR_PERC'].fillna(0)
    
    return temp_df


def DFStatisticalReview(df,file_name=None,print_=0,time_check=20):
    
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
        temp_df = ColumnStatisticalReview(df,column)
        elapsed_time = timeit.default_timer() - start_time
        final_df = pd.concat([final_df,temp_df],axis=1)
        
        if elapsed_time>time_check:
            print(f'Elapsed time to process {column}:{timeit.default_timer() - start_time:,.2f}')
            resume_ = input(f'Would you like to continue? 1 Column is taking more than {time_check} Seconds. (Yes/Y)')
            if (resume_.lower()!='yes')|(resume_.lower()!='y'):
                time_check += 5
                return final_df
        
        if print_==1:
            print(f'Elapsed time to process {column}:{timeit.default_timer() - start_time:,.2f}')
            
    if file_name:
        final_df.to_csv(f"{file_name}.csv")
        
    return final_df

def DFColumnCompare(df,df1):
    
    '''
    
    Function to Compare Column Values contained within 2 Dataframes, Identifying Equivalency and Variance.
    
    Parameters:
        df(dataframe): Dataframe
        df1(dataframe): DataFrame
        
    Returns:
        Multiple Dataframes, 
    
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
    
    return missingdf,missingdf1



def IdenticalColumnDQValidation(df,
                                column_name,
                                column_name1=None,
                                additional_filter=None,
                                column_distinction='_',
                                bracketing=[-10000,-1000,-100,-1,0,1,100,1000,10000]):
    
    '''
    Function which takes a dataframe with 2 Columns which are identical and attempts to Compare.
    Designed with the intention of comparing,  BALANCE, BALANCE_, however can explicitly utilize column_name1 to override.

    Function CompareIdenticalDF applies to Entire DataFrame
    
    
    Parameters:
        column_name (str):
        column_name1 (str):
        additional_filter (str): Default parameter to distinguish combined dataframes, also used in MergeAndRenameColumnsDf
        bracketing(list): Value to create Distintion when Calculated Difference between columns is numeric.
        
    
    Values:
    
    
    Example Usage:
        
        df= df[[START_BAL, START_BAL_, ACCTNBR]],
        column_name='START_BAL'

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

def CompareIdenticalDF(df,
                       primary_key_list,
                       additional_filter,
                       column_distinction='_',
                        bracketing=[-10000,-1000,-100,-1,0,1,100,1000,10000]):
    
    '''
    Function to Apply ElementLevelDFColumnReview against an Entire DataFrame.
    Assumes you start with a Dataframe with Multiple Columns Different only by Column Distinction.
    
    
    
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

    return account_df,summary_df,groupby_df


def IdenticalColumnStatisticalReviewDF(df,index_list,primary_key_list,column_name1=None,column_distinction='_'):
    
    '''
    Function to Apply ColumnStaticalReview against multiple Indexs. Automation of muliple index was slightly confusing, 
    and could not easily automate and regularization, so manually did it.
    
    Parameters:
        df (dataframe):
        index_list (list): List of Values which are used as Filter
    
    
    '''
    final_df = pd.DataFrame()
    
    if len(index_list)==1:
        index1 = df[index_list[0]].unique()
        
        for entity in index1:
            temp_df = df[df[index_list[0]]==entity]
            
            for column in [x for x in df.columns if (x not in primary_key_list)&(x[-1]!=column_distinction)]:
                temp_df1 = ColumnStatisticalReview(temp_df,column).reset_index().rename(columns={'index':"COLUMN_NAME",column:"VALUE"})
                temp_df1['FILTER1'] = entity
                temp_df1['FILTER2'] = column
                final_df = pd.concat([final_df,temp_df1])
    return final_df


