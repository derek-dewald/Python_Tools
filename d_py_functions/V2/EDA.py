## File Description: This File is for Functions related to completing EDA. Including Setting up standard processes. Individual Image or DF manipulation which might be utilized in this function, but exist for broader purposes should be stored in more generic categories.

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from Visualization import Heatmap,plot_histograms,plot_scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import pandas as pd
import numpy as np
import math



def analyze_distribution(df):
    """
    Analyzes skewness, kurtosis, and visualizes the distribution of all numeric columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    None (displays plots and summary in an X by 3 format)
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    num_vars = len(numeric_cols)
    
    fig, axes = plt.subplots(num_vars, 3, figsize=(18, 6 * num_vars))
    
    if num_vars == 1:
        axes = [axes]  # Ensure iterable for single variable
    
    for i, col in enumerate(numeric_cols):
        target_data = df[col].dropna()
        
        # Compute skewness & kurtosis
        skewness = skew(target_data)
        kurt = kurtosis(target_data)
        
        if abs(skewness) > 1:
            skewness_comment = "Highly Skewed, Consider Transformation"
        elif skewness > 0:
            skewness_comment = "Right Skewed"
        elif skewness < 0:
            skewness_comment = "Left Skewed"
        else:
            skewness_comment = "Symmetric"

        # Correct Kurtosis Comments
        if kurt >= 2.75 and kurt <= 3.25:
            kurt_comment = "Mesokurtic (Normally Distributed), No Action Necessary."
        elif kurt > 3.25:
            kurt_comment = "Leptokurtic (High Kurtosis), More Extreme Values"
        elif kurt < 2.75:
            kurt_comment = "Platykurtic (Low Kurtosis), Less Extreme Values"
        
        summary_text = f"Skewness: {skewness:.2f}\n{skewness_comment}\nKurtosis: {kurt:.2f}\n{kurt_comment}"
        
        # Plot Histogram
        sns.histplot(target_data, kde=True, bins=30, color="blue", ax=axes[i][0])
        axes[i][0].set_title(f"Histogram of {col} (Skew: {skewness:.2f})")
        
        # Plot Boxplot
        sns.boxplot(x=target_data, color="red", ax=axes[i][1])
        axes[i][1].set_title(f"Boxplot of {col}")
        
        # Display Text
        axes[i][2].text(0.5, 0.5, summary_text, fontsize=12, va='center', ha='center', bbox=dict(facecolor='white', alpha=0.8))
        axes[i][2].axis("off")
        
    plt.tight_layout()
    plt.show()


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
