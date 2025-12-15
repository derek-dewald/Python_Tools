import pandas as pd

def import_d_google_sheet(definition=None):

    '''
    Function Which Extracts a DataFrame from D's Google Sheets, using the Map as provided in Google Mapping Sheet

    Parameters:
        definition(str): Value to be Filtered from D Mapping Sheet, using Column Definition. Which is the name of the sheet.

    Returns:
        Dataframe (Dict is value is not try condition fails).

    date_created:12-Dec-25
    date_last_modified: 12-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        Example Function Call

    '''

    from data_d_strings import google_mapping_sheet_csv
    from dict_processing import df_to_dict

    df = pd.read_csv(google_mapping_sheet_csv)
    df= df[df['CSV'].notnull()].copy()

    try:
        csv_ = df[df['Definition']==definition]['CSV'].item()
        return pd.read_csv(csv_)

    except:
        return df[['Definition','Description']]