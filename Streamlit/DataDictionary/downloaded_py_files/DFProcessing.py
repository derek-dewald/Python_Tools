## File Description: A more generalized Libary, including simple tricks, shortcuts and helpful functions for reviewing, summarizing or understanding. More Information, and less Procedural related. EDA for standardized processes.

from itertools import product,permutations,combinations
from FeatureEngineering import BracketColumn,CategorizeBinaryChange
from IPython.display import display, HTML
import pandas as pd
import numpy as np
import math

def CombineLists(list_,
                 combo=1,
                 r=2):
        
    '''
    Function to 
    
    
    Parameters:
        
    
    Return:
        
    '''
    
    items = sum([1 if isinstance(x,list) else 0 for x in list_])

    if items==0:
        if combo==1:
            return list(map(list,combinations(list_,r)))
        else:
            return list(map(list,permutations(list_,r)))
    
    return list(map(list,product(*list_)))

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


def TransposePivotTable(df,
                        reset_index=1,
                       x_axis='X_Axis',
                       y_axis='Y_Axis',
                       value='VALUE'):
    '''
    Function to take Pivot Table and Turn it into a Vertically Stacked List.
    
    Parameters:
        reset_index(int): Binary Flag to determine whether intex needs to be reset or not.
        x_axis (str): Column Naming from X_axis. Default X_Axis
        y_axis (str):
        value (str):
        
    Returns
        Dataframe
    
    '''
    
    if reset_index==1:
        df1 = df.stack().reset_index()
    else:
        df1 = df.stack()
    try:
        df1 =  df1.rename(columns={'level_0':x_axis,'level_1':y_axis,0:value})
    except:
        pass
    
    df1 = df1[df1[x_axis] < df1[y_axis]].copy()
    
    return df1[df1[x_axis]!=df1[y_axis]].sort_values(value)

def GenerateBinaryChange(df,
                           change_column='VARIANCE',
                           print_=1):
    '''
    Function to Simply Apply a Condition to generate whether a particular column, which is meant to be a Change over a 
    time series dataset, has Increased, Decreased or Stayed the Same. Can be stand alone, created for use in ColumnElementalChange
    
    Parameters:
        change_column (str): Name of Column Created. 
        
    Returns:
        dataframe, with change_column added
        
    '''
    
    condition = [
        df[change_column]>0,
        df[change_column]<0,
        df[change_column]==0
    ]
        
    value = ['Records Increasing',
             'Records Decreasing',
             'Records Not Changing']
        
    df['CHANGE_CLASSIFICATION'] = np.select(condition,value,'Null Value Present')
    if print_==1:
        print(df['CHANGE_CLASSIFICATION'].value_counts())



def ColumnStatisticalCompare(df,df1,column_name):
    '''
    

    Args:

    Returns:

    
    
    '''

    a = ColumnStatisticalReview(df,column_name).rename(columns={column_name:f'{column_name}_DF'})
    b = ColumnStatisticalReview(df1,column_name).rename(columns={column_name:f'{column_name}_DF1'})
    
    temp_df = pd.concat([a,b],axis=1)
    temp_df[f'{column_name}_VAR_AMT'] = temp_df[f'{column_name}_DF'] - temp_df[f'{column_name}_DF1'] 
    temp_df[f'{column_name}_VAR_PERC'] = temp_df[f'{column_name}_VAR_AMT']/temp_df[f'{column_name}_DF1']
    temp_df[f'{column_name}_VAR_PERC'] = temp_df[f'{column_name}_VAR_PERC'].fillna(0)
    
    return temp_df

def DFStructureReview(df,
                      primary_key='',
                      df1=""):
    
    '''
    Function to Create a simplified view of the overall structure of a dataframe, or where 2 similiar dataframes are 
    present, to help understand the difference between.
    
    Args:
        primary_key(list): Primary Keys between 
    
    Returns:
        dictionary
        
    '''
    
    prop_dict = {}
    
    prop_dict['TotalRecords_DF']= len(df)
    prop_dict['TotalColumns_DF'] = len(df.columns)
    
    try:
        prop_dict['UniqueRecords_DF']= len(df.drop_duplicates(primary_key))
    except:
        pass
    
    # If there are 2 dataframes:
    
    if len(df1)!=0:
        cols_df =  set(df.columns.values)
        cols_df1 = set(df1.columns.values)
    
        prop_dict['cols_missing_df']  = [x for x in list(cols_df1) if x not in cols_df]
        prop_dict['cols_missing_df1'] = [x for x in list(cols_df) if x not in cols_df1]
        prop_dict['TotalRecords_DF_1'] = len(df1)
        prop_dict['TotalColumns_DF1']  = len(df1.columns)
        
        try:
            temp_df = df[primary_key].merge(df1[primary_key],on=primary_key,how='outer',indicator=True)
            prop_dict['UniqueRecords_DF1'] = len(df1.drop_duplicates(primary_key))
            prop_dict['SharedRecords']     = len(temp_df[temp_df['_merge']=='both'])
            prop_dict['Only_DF']           = len(temp_df[temp_df['_merge']=='left_only'])
            prop_dict['Only_DF1']          = len(temp_df[temp_df['_merge']=='right_only'])
        
        except:
            pass

    return prop_dict

def ColumnStatisticalReview(df,
                            column_name,
                            partitions=10,
                            top_x_records=10,
                            exclude_blanks_from_segments=1,
                            exclude_zeroes_from_segments=1):
    
    '''
    Function to Conduct a Simple Statistical Review of a Column, Including Understanding the positional distribution
    of values. 
    
    Args:
        column_name (str): Name of Column
        
        partitions (int): Number of partitions to include (Decile 10)
        
        exclude_blanks_from_segments (int): Binary Flag, whether to exclude Blank Values from Segment determination.
        If blank values are excluded it gives a better representation for the members of the set, however it might 
        provide a misleading representation of the population.
        
        exclude_zeroes_from_segments (int): As above, with respect to 0 values. Is processed after exclude_blanks, as
        such it can include both blanks and true 0 values. 
        
        
    '''
    
    temp_dict = {}
    
    try:
        temp_dict['SUM'] = df[column_name].sum()
        temp_dict['MEAN'] = df[column_name].mean()
        temp_dict['STD_DEV'] =  df[column_name].std()
        temp_dict['MEDIAN'] = df[column_name].median()
        temp_dict['MAX'] = df[column_name].max()
        temp_dict['MIN'] = df[column_name].min()
    
    except:
        pass

    temp_dict['TOTAL_RECORDS'] = len(df)
    temp_dict['UNIQUE_RECORDS'] = len(df.drop_duplicates(column_name))
    temp_dict['ZERO_RECORDS'] = len(df[df[column_name]==0])
    temp_dict['NON_ZERO_RECORDS'] = len(df[df[column_name]!=0])
    temp_dict['NA_RECORDS'] = len(df[df[column_name].isna()])
    temp_dict['NULL_RECORDS'] = len(df[df[column_name].isnull()])
                             
    temp_df = pd.DataFrame(temp_dict.values(),index=temp_dict.keys(),columns=[column_name])
    
    # Add top X records Based on Frequency
    if top_x_records>0:
        top_instances = pd.DataFrame(df[column_name].value_counts().head(top_x_records)).reset_index()
        top_instances[column_name] = top_instances.apply(lambda row: f"Value: {row[column_name]}, Frequency: {row['count']}", axis=1)
        top_instances['index'] = [f"Top {x+1}" for x in range(len(top_instances[column_name]))]
        top_instances = top_instances.drop('count',axis=1).set_index('index')

        temp_df = pd.concat([temp_df,top_instances])
    
    try:
        temp_dict['SUM']
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
        return temp_df
    
def CountBlanksZeroes(df):
    '''
    Function to Count the Number of Blanks, Zereos and Nulls in a Dataframe. 

    Parameters:
        df (dataframe)

    Returns
        Dataframe, with all columns as column and 3 rows, with count of observed values.
    
    
    '''

    final_dict = {}
    
    for column in df.columns:
        final_dict[column] = {'Blanks':len(df[df[column]==""]),
                            'Zeros':len(df[(df[column]==0)|(df[column]=="0")]),
                            'Null':len(df[df[column].isnull()])}
        
    return pd.DataFrame(final_dict)

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
    


def CalculateColumnWiseCorrelation(df,
                                   column_name=None,
                                   tail_head_count=5):
    
    '''
    Function to summarize Correlation Coeffient within dataframe Columns (or sample of columns if defined).
    Differs from Heatmap, in that it is a numerical lead summary.
    
    Parameters
        df (dataframe)
        column_name (list): List of Column Names to be included in summary, default to include all as indicated by None
        tail_head_count (int): Number of Records to return Top and Bottom (specifically includes in seperate DF)
    
    Return
        Multiple DataFrames
        
    '''
    
    if column_name == None:
        column_name = df.columns.tolist()
       
    df1 = TransposePivotTable(df[column_name].fillna(0).corr(),value="CorrelationCoefficient")
    
    BracketColumn(df1,'CorrelationCoefficient','Segment',[-1,-.5,0,.5,1])
    
    bot = df1.head(tail_head_count).copy()
    bot['Description'] = [f'Top {x} Negative Correlation' for x in range(1,tail_head_count+1)]
    
    top = df1.tail(tail_head_count).copy()
    top['Description'] = [f'Top {x-1} Positive Correlation' for x in range(tail_head_count+1,1,-1)]
    
    temp_ = pd.concat([bot,top]).reset_index(drop=True)
    
    display(temp_)
    
    return df1,temp_

def DFStructureReview(df,
                      primary_key='',
                      df1=""):
    
    '''
    Function to Create a simplified view of the overall structure of a dataframe, or where 2 similiar dataframes are 
    present, to help understand the difference between.
    
    Args:
        primary_key(list): Primary Keys between 
    
    Returns:
        dictionary
        
    '''
    
    prop_dict = {}
    
    prop_dict['TotalRecords_DF']= len(df)
    prop_dict['TotalColumns_DF'] = len(df.columns)
    
    try:
        prop_dict['UniqueRecords_DF']= len(df.drop_duplicates(primary_key))
    except:
        pass
    
    # If there are 2 dataframes:
    
    if len(df1)!=0:
        cols_df =  set(df.columns.values)
        cols_df1 = set(df1.columns.values)
    
        prop_dict['cols_missing_df']  = [x for x in list(cols_df1) if x not in cols_df]
        prop_dict['cols_missing_df1'] = [x for x in list(cols_df) if x not in cols_df1]
        prop_dict['TotalRecords_DF_1'] = len(df1)
        prop_dict['TotalColumns_DF1']  = len(df1.columns)
        
        try:
            temp_df = df[primary_key].merge(df1[primary_key],on=primary_key,how='outer',indicator=True)
            prop_dict['UniqueRecords_DF1'] = len(df1.drop_duplicates(primary_key))
            prop_dict['SharedRecords']     = len(temp_df[temp_df['_merge']=='both'])
            prop_dict['Only_DF']           = len(temp_df[temp_df['_merge']=='left_only'])
            prop_dict['Only_DF1']          = len(temp_df[temp_df['_merge']=='right_only'])
        
        except:
            pass

    return prop_dict

def ColumnStatisticalCompare(df,df1,column_name):
    '''
    Function to Compare Two Columns, primarily derived for TimeSeries Analysis.


    Parameters:

    
    Returns:

    
    
    '''

    a = ColumnStatisticalReview(df,column_name,top_x_records=0).rename(columns={column_name:f'{column_name}_DF'})
    b = ColumnStatisticalReview(df1,column_name,top_x_records=0).rename(columns={column_name:f'{column_name}_DF1'})
    
    temp_df = pd.concat([a,b],axis=1)
    temp_df[f'{column_name}_VAR_AMT'] = temp_df[f'{column_name}_DF'] - temp_df[f'{column_name}_DF1'] 
    temp_df[f'{column_name}_VAR_PERC'] = temp_df[f'{column_name}_VAR_AMT']/temp_df[f'{column_name}_DF1']
    temp_df[f'{column_name}_VAR_PERC'] = temp_df[f'{column_name}_VAR_PERC'].fillna(0)
    
    return temp_df
    
def DataFrameColumnObservations(df,
                                columns,
                                include_obs=1,
                                include_perc=1):
    
    '''
    Function to provide a quick summary of how a Dataframe is distributed amongst a set of determined columns.
    
    
    Parameters:
        columns (list): Columns to which you wish to be invluded
        include_obs (int): Binary Flag to indicate whether column Observation Count Totals are included
        include_perc (int): Binary Flag to indicate whether column Percentage Calculations are to be included
    
    Returns:
        Dataframe
    
    '''

    temp_df = df[columns].copy()
    temp_df['COUNT'] = 1
    
    final_df = temp_df.fillna("").groupby(columns).sum().reset_index()
    
    for column in columns:
        temp_df1 = temp_df[[column,'COUNT']].groupby(column).sum().reset_index().rename(columns={'COUNT':f"{column}_OBS"})
        final_df = final_df.merge(temp_df1,on=column,how='left')

    if include_perc==1:
        final_df['PERC_DATA'] = final_df['COUNT']/len(temp_df)
        final_df = final_df.sort_values('COUNT',ascending=False)
        final_df['CUMMULATIVE_PERC'] = final_df['PERC_DATA'].cumsum()
        
        for column in columns:
            final_df[f"{column}_PERC"] = final_df[f"{column}_OBS"]/len(temp_df)
        
    if include_obs==0:
        return final_df.drop([f"{column}_OBS" for x in columns],axis=1)
    
    else:
        return final_df
    
def CreatePivotTableFromTimeSeries(df,
                                   index,
                                   columns,
                                   values,
                                   aggfunc='sum',
                                   skipna=True):
    
    '''
    Function to Summaryize a Time Series Dataframe into a Pivot. Creating a number of critical Metrics.
    
    
    
    '''
    
    # 1. Pivot
    df1 = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)

    # 2. Capture original month columns IMMEDIATELY after pivot
    month_cols = df1.columns.tolist()
 
    # 3. Add rolling window stats
    if len(month_cols) >= 3:
        df1['AVG_3M'] = df1[month_cols[-3:]].mean(axis=1, skipna=skipna)
        df1['CHG_3M'] = df1[month_cols[-1]]-df1[month_cols[-3]]
    
    if len(month_cols) >= 6:
        df1['AVG_6M'] = df1[month_cols[-6:]].mean(axis=1, skipna=skipna)
        df1['CHG_6M'] = df1[month_cols[-1]]-df1[month_cols[-6]]
        
    if len(month_cols) >= 12:
        df1['AVG_12M'] = df1[month_cols[-12:]].mean(axis=1, skipna=skipna)
        df1['CHG_12M'] = df1[month_cols[-1]]-df1[month_cols[-12]]
        
    df1['CHG_MOM'] = df1[month_cols[-1]]-df1[month_cols[-12]]
    df1['CHG_DF']  = df1[month_cols[-1]]-df1[month_cols[0]]

    # 4. Now calculate global stats **only using the original month columns**
    stats = pd.DataFrame({
        'MEAN': df1[month_cols].mean(axis=1, skipna=skipna),
        'STD': df1[month_cols].std(axis=1, skipna=skipna),
        'MAX': df1[month_cols].max(axis=1, skipna=skipna),
        'MIN': df1[month_cols].min(axis=1, skipna=skipna),
        'COUNT': df1[month_cols].count(axis=1)
    })

    # 5. Merge the stats
    df1 = pd.concat([df1, stats], axis=1)
    
    return df1.fillna(0)

def MissingCartesianProducts(list1_,
                             list2_,
                             columns,
                             merge_df=None,
                             remove_values=['0',"",'N/A']):
    '''
    Function which Looks at the Combination of Two Lists and explores all possible Combinations. 
    Developed for the purpose of understanding how many combinations exist and generating a list of Values which do not
    exist, this list can be valueable for pending to previously Aggregagted Datasets to insure all possible records have
    representation
    
    Parameters:
        list1_ (list)
        list2_ (list)
        columns (Columns to be included, should represent the expected Column Name of list1_ and list2_)
        merge_df (Dataframe): To be used to Validate the number of missing records, if not included, then 
        returns only combination.
    
    Returns:
    
    
    '''

    from DFProcessing import CombineLists
    
    list1_ = [x for x in list1_ if x not in remove_values]
    list2_ = [x for x in list2_ if x not in remove_values]
    
    required_records = CombineLists([list1_,list2_])
    df = pd.DataFrame(required_records,columns=columns)

    
    if len(merge_df)==0:
        return df
    
    else:
        df = df.merge(merge_df[columns].drop_duplicates(),on=columns,how='left',indicator=True)
        print(f"Distribution of Records and Missing Records:\n{df['_merge'].value_counts()}")
        df = df[df['_merge']=='left_only'].drop('_merge',axis=1)
        return df
    

def TranposeNonTimeSeriesDF(df, index, columns):
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
    return df.melt(id_vars=index, value_vars=columns)





def ColumnElementalChange(df,
                          df1,
                          column_name,
                          primary_key=['MEMBERNBR']):
    '''
    Function to Compare the Element Level change of two dataframes. Includes: Summary of Records Increasing and Decreasing,
    A summary of where the respective partitions are 
    
    Args:
        column_name (str): Name of Column to which you wish to analyze
        primary_key (list): List of Primary Key(s), used for purposes of Merging df and df1
        
    Returns:
        Single Rowed Dataframe including a summary of Record Changing Count (Increase, Decrease, No Change), 
        Position of respective Partitions in Dataset (remembering Position 0 is MIN value and position -1 is 90% value)
        Total Value Change which is attributed to respective Partition.
        
    '''
    
    # Using primary Key and Column Name create conditions for Merge
    key = primary_key.copy()
    key.append(column_name)    
    
    # Merge DF and DF1 for record level comparison 
    temp_df = df[key].rename(columns={column_name:f"{column_name}_DF"}).merge(df1[key].rename(columns={column_name:f"{column_name}_DF1"}),
                                                                                                       on=primary_key,
                                                                                                       how='outer',
                                                                                                       indicator=True)
    # Create a Column to Track Change at Record Level 
    
    try:
        temp_df['VARIANCE'] = temp_df[f"{column_name}_DF"].fillna(0) - temp_df[f"{column_name}_DF1"].fillna(0)
        
        # Create a variable which defines change in text terms, Increase, Decrease, No Change, Null Value Detered for VALUES
        CategorizeBinaryChange(temp_df,'VARIANCE')
    
    except:
        temp_df['VARIANCE'] = np.where(temp_df[f"{column_name}_DF"].fillna(0) != temp_df[f"{column_name}_DF1"].fillna(0),1,0)
        print(temp_df['VARIANCE'].sum())

    # Developing Components that deal with text.
    return temp_df
        
    
    # Add Partition Column for Purposes of Calculating Aggregate Change
    agg_impact_df = ColumnPartitioner(temp_df,'VARIANCE',return_value='agg_value')
    
    # Create a Dataframe Stacked Vertically for Increase, Decrease, No CHange
    chg_val_df = pd.DataFrame(temp_df['CHANGE_CLASSIFICATION'].value_counts()).rename(columns={'CHANGE_CLASSIFICATION':'VALUE'})

    # Create a DataFrame Stacked Vertically for Decile Partions
    partition_val_df = ColumnPartitioner(temp_df,'VARIANCE',new_column_name='Variance Partition').T.rename(columns={'VARIANCE':'VALUE'})

    # Add all individual elements
    final_df = pd.concat([chg_val_df,
                          partition_val_df,
                          agg_impact_df])
                        
    return final_df


def ConvertDicttoDF(dict_, key_name="KEY", value_name="VALUE"):
    '''
    Function to convert a straight Dictionary into a Dataframe.

    Parameters
    dict_ (dict)
    key_name (str): Name of the column for dictionary keys. Default is 'KEY'.
    value_name (str): Name of the column for dictionary values. Default is 'VALUE'.

    Returns
        DataFrame
    '''
    return pd.DataFrame.from_dict(dict_, orient='index', columns=[value_name]).reset_index().rename(columns={'index': key_name})

def ConvertListstoDF(dict_lists):
    '''
    Function to Take a Dictionary of Lists and convert into a Dataframe. 

    dict_lists = {'Column1':list1,'Column2':List2}

    Parameter:
        Dictionary of Lists, Keys will become column Name

    Returns
        DataFrame
    
    

    '''

    return pd.DataFrame(dict_lists)