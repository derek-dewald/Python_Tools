import pandas as pd
import datetime

def ParamterMapping(Definition=""):
    
    '''
    Function to Google Mapping Sheet, which is used to store Mappings, Links, etc.
    For both simplicity and Organization
    
    Args:
        Definition (Str): Key word used to Access individual elements
        
    Returns:
        Dataframe, unless Definition is defined, in which case it might be Str.
    
    '''

    df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSwDznLz-GKgWFT1uN0XZYm3bsos899I9MS-pSvEoDC-Cjqo9CWeEuSdjitxjqzF3O39LmjJB0_Fg-B/pub?output=csv')
    
    # If user has not included a definition, the return entire DF
    if len(Definition)==0:
        return df
    else:
        try:
            df1 = df[df['Definition']==Definition]
            if len(df1)==1:
                if df1['TYPE'].item()=='csv':
                    return pd.read_csv(df1['VALUE'].item())
                else:
                    return df1['VALUE'].item()
        except:
            return df[df['Definition']==Definition] 


def BackUpGoogleSheets(location='/Users/derekdewald/Documents/Python/Github_Repo/CSV Backup Files/'):
    '''
    Function to Create a Backup of Information Stored in Google Sheets.
    
    Parameters:
        None
        
    Returns:
        CSV Files 
    
    '''
    
    df = ParamterMapping()
    
    for row in range(len(df)):
        try:
            file_name = df['Definition'][row]
            file_location = df['CSV'][row]
            month = datetime.datetime.now().strftime('%b-%y')
            
            temp_df = pd.read_csv(file_location)
            temp_df.to_csv(f'{location}{file_name}_{month}.csv',index=False)
            print(f'Back Up Saved, {location}{file_name}')
        except:
            print(f'Counld Not Print Record {row}')
