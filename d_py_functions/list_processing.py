import pandas as pd
import numpy as np

def list_to_dataframe(list_,
                      column_name_list=None):
    '''
    Function to Simplify the creation of a Dictionary into a Dataframe into a single Command.

    Parameters:
        list_ (list): List of Values to be iterated into Row.
        column_name_list (list): Name of Column to be added, add as List. 

    Returns:
        Object Type

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        temp_df = list_to_dataframe(dict)
    '''
    if not column_name_list:
        return pd.DataFrame(list_)
    else:
        return pd.DataFrame(list_,columns=column_name_list)
