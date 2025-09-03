import pandas as pd
import numpy as np

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
    if index==None:
        df1 = df.pivot_table(columns=columns,values=values,aggfunc=aggfunc)
    else:
        df1 = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)

    # 2. Capture original month columns IMMEDIATELY after pivot
    month_cols = df1.columns.tolist()
 
    # 3. Add rolling window stats
    if len(month_cols) >= 3:
        df1['AVG_3M'] = df1[month_cols[-3:]].mean(axis=1, skipna=skipna)
        df1['CHG_3M'] = df1[month_cols[-1]]-df1[month_cols[-3]]
        try:
            df1['PERC_CHG_3M'] = df1['CHG_3M']/df1[month_cols[-3]]
        except:
            df1['PERC_CHG_3M'] = 0
    
    if len(month_cols) >= 6:
        df1['AVG_6M'] = df1[month_cols[-6:]].mean(axis=1, skipna=skipna)
        df1['CHG_6M'] = df1[month_cols[-1]]-df1[month_cols[-6]]
        try:
            df1['PERC_CHG_6M'] = df1['CHG_6M']/df1[month_cols[-6]]
        except:
            df1['PERC_CHG_6M'] = 0
            
    if len(month_cols) >= 12:
        df1['AVG_12M'] = df1[month_cols[-12:]].mean(axis=1, skipna=skipna)
        df1['CHG_12M'] = df1[month_cols[-1]]-df1[month_cols[-12]]
        try:
            df1['PERC_CHG_12M'] = df1['CHG_12M']/df1[month_cols[-12]]
        except:
            df1['PERC_CHG_12M'] = 0

    df1['CHG_DF']  = df1[month_cols[-1]]-df1[month_cols[0]]
    df1['AVG_DF'] = df1[month_cols[-1:]].mean(axis=1, skipna=skipna)
    df1['PERC_CHG_DF'] = df1['AVG_DF']/df1[month_cols[-1]]

    
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


def CreateMultiplePivotTableFromTimeSeries(df,
                                           index_list,
                                           metric_list,
                                           column):
    '''
    Function to utilize when Attempting to Create Multip[le Times Series. Specifically Multiple Metrics, and Multiple Index's
    



    '''
    

    final_df = pd.DataFrame()
    
    # Iterate through all Possible Metrics Selected.
    
    for metric in metric_list:
        print(f'Attempting to Process:{metric}')
        try:
            all_df = CreatePivotTableFromTimeSeries(df=df,index=None,columns=column,values=metric,aggfunc='sum') 
            cols = list(all_df.columns)
            all_df = all_df.reset_index(drop=True)
            all_df['METRIC'] = metric
            cols.insert(0,'METRIC')

            for key in index_list:
                cols.insert(0,key)
                all_df[key] = 'All'

            final_df = pd.concat([final_df,all_df[cols]])
            # Iterate through all Index Items Individually
            for key in index_list:
                temp = CreatePivotTableFromTimeSeries(df,
                                                      index=key,
                                                      values=metric,
                                                      columns=column).reset_index() 
                for missing in [x for x in index_list if x != key]:
                    temp[missing] = 'All'
                temp['METRIC'] = metric
                final_df = pd.concat([final_df,temp])

            # Add Value for Metric with Entire Index Combination
            temp = CreatePivotTableFromTimeSeries(df,index=index_list,values=metric,columns=column).reset_index()
            temp['METRIC'] = metric
            final_df = pd.concat([final_df,temp])
        except:
            print(f'Could Not Process Metric:{metric}.')

    return final_df

def SummarizeTimeSeriesDf(df,
                          summary_cols,
                          primary_key_list):
    '''
    Function to Summarize a Time Series dataframe based on a finite number of identified Columns.
    
    Parameters
        df (Dataframe): TimeSeries in Nature
        summary_cols (List): List of Columns which are to be included in SUmmary
        primary_key_list (list): Primary Key of Dataframe
    
    Returns
        temp_df1: Raw Data of SUmmary Cols with a Count of Observations. If include Month Variable Easy to add to Pivot Table
        summary: Summary (Excluding Primary Key). including Total Observations, MEan, Max, Min.
    
    
    '''
    temp_df = df[summary_cols].copy()
    temp_df['COUNT'] = 1
    
    # Unique Occurances by Pivot Criteria. Important to Include Month
    temp_df1 = temp_df.groupby(summary_cols).sum().reset_index().rename(columns={'COUNT':'TOTAL_DAYS'})
    
    pivot_columns1 = [x for x in summary_cols if x not in primary_key_list]
    
    summary = temp_df1.groupby(pivot_columns1).agg(
        TOTAL=('TOTAL_DAYS', 'count'),
        AVG_DAYS_OPEN=('TOTAL_DAYS', 'mean'),
        MAX_OBS=('TOTAL_DAYS', 'min'),
        MIN_OBS=('TOTAL_DAYS', 'max')).reset_index()
    
    return temp_df1,summary
