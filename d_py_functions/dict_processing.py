import pandas as pd
import numpy as np

def convert_dict_to_parameters(module) -> pd.DataFrame:
    """
    Convert a .py file containing Lists into a Dataframe which can is used to populate Function Parameters

    Parameters:
        module (module): Py file containing Lists.

    Returns:
        Dataframe

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        import d_dictionaries
        temp = convertlisttoparameters(d_dictionaries)

    """
    rows = []
    for name, value in vars(module).items():
        if not name.startswith("_") and isinstance(value, dict):
            for k, v in value.items():
                rows.append({
                    "Folder": 'data_d_lists.py',
                    'Function':name,
                    "Parameters": k,
                    'Type':'dict',
                    "Definition": v
                })
    return pd.DataFrame(rows)


def dict_to_dataframe(dict_,
                    key_name="KEY",
                    value_name='VALUE'):
    '''
    Function to Simplify the creation of a Dictionary into a Dataframe into a single Command.

    Parameters:
        dict_(dict)
        key_name(str): Name of Column which will include values from Key (Default is KEY)
        value_name(str): Name of Column which will include values from Values (Default is Value)

    Returns:
        Dataframe

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        temp_df = dict_to_dataframe(dict)
    '''
    return pd.DataFrame.from_dict(dict_, orient='index', columns=[value_name]).reset_index().rename(columns={'index': key_name})

def df_to_dict(df,key,value):
    
    '''
    Function to Simply Convert A DF into a Dictionary.
    Takes 2 Arguments, and converts them into a DF of the format {key:value}

    Parameters:
        df (df): Any DataFrame
        key (str): String representing Column Name for Dictionary Key
        value(str): String representing Column Name for Value Key
        
    Returns:
        df

    date_created:12-Dec-25
    date_last_modified: 12-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        from data_d_strings import google_mapping_sheet_csv
        df = pd.read_csv(google_mapping_sheet_csv)
        df_to_dict(df,'Definition','CSV')

    '''

    temp_df = df[[key,value]].copy()

    return df[[key,value]].set_index(key).to_dict()[value]