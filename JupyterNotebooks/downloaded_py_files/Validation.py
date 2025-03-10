import pandas as pd
import numpy as np

def BinaryColumnCreator(df,
                        column_name,
                        new_column_name,
                        value,
                        calculation):
  
  if calculation=='>':
    df[new_column_name] = np.where(df[column_name]>value,1,0)
  elif calculation =='=':
    df[new_column_name] = np.where(df[column_name]==value,1,0)
  elif calculation =='<':
    df[new_column_name] = np.where(df[column_name]<value,1,0)
  elif calculation =='isin':
    df[new_column_name] = np.where(df[column_name].isin(value),1,0)
  elif calculation =='contains':
    df[new_column_name] = np.where(df[column_name].str.contains(value),1,0)
  elif calculation == 'dict':
    for key,value in value.items():
        try:
          df[new_column_name] = np.where((df[key]==value)&(df[new_column_name]==1),1,0)
        except:
          df[new_column_name] = np.where(df[key]==value,1,0)


def TextClean(df, 
              column_list, 
              clean_type='only_digits'):

    if clean_type == 'only_digits':
        for column in column_list:
            df[column] = df[column].astype(str).str.replace(r'[^\d.]', "", regex=True)
            df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

def FillFromAbove(df,
                 column_name,
                 new_column_name=''):
    
    '''
    Purpose: Force Fill Information from above defined values when Blank. 


    Example: 
    df = pd.DataFrame({'Test': [None, 'Value1', None, 'Value2', None, None, 'Value3', None]})
    FillFromAbove(df,'Test','Test1')
    
    
    '''
    if new_column_name=="":
        new_column_name = column_name
    
    df[new_column_name] = df[column_name].ffill()


def ColumnStatisticalCompare(df,df1,column_name):
    '''
    
    
    
    '''



    a = ColumnStatisticalReview(df,column_name).rename(columns={column_name:f'{column_name}_DF'})
    b = ColumnStatisticalReview(df1,column_name).rename(columns={column_name:f'{column_name}_DF1'})
    
    temp_df = pd.concat([a,b],axis=1)
    temp_df[f'{column_name}_VAR_AMT'] = temp_df[f'{column_name}_DF'] - temp_df[f'{column_name}_DF1'] 
    temp_df[f'{column_name}_VAR_PERC'] = temp_df[f'{column_name}_VAR_AMT']/temp_df[f'{column_name}_DF1']
    temp_df[f'{column_name}_VAR_PERC'] = temp_df[f'{column_name}_VAR_PERC'].fillna(0)
    
    return temp_df

def CategorizeBinaryChange(df,
                           change_column='VARIANCE'):
    '''
    Function to Simply Apply a Condition to generate whether a particular column, which is meant to be a Change over a 
    time series dataset, has Increased, Decreased or Stayed the Same. Can be stand alone, created for use in ColumnElementalChange
    
    Args:
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
    
    try:
        # Supplement Change to increase visability of granularity so we can understand true variance, in addition to Add/Losses
        df['CHANGE_CLASSIFICATION'] = np.where(df['_merge']=='right_only','Records Lost',df['CHANGE_CLASSIFICATION'])
        df['CHANGE_CLASSIFICATION'] = np.where(df['_merge']=='left_only','Records Added',df['CHANGE_CLASSIFICATION'])
        
    except:
        pass

def ColumnElementalChange(df,
                          df1,
                          column_name,
                          primary_key=['MEMBERNBR']):
    '''
    Function to Compare the Element Level change of two dataframes. Includes: Summary of Records Increasing and Decreasing,
    A summary of where the respective partitions are 
    
    Arg:
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
    temp_df['VARIANCE'] = temp_df[f"{column_name}_DF"].fillna(0) - temp_df[f"{column_name}_DF1"].fillna(0)
    
    # Create a variable which defines change in text terms, Increase, Decrease, No Change, Null Value Detered
    CategorizeBinaryChange(temp_df,'VARIANCE')
    
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