import pandas as pd
import webbrowser
import datetime
import requests
import os

def DownloadFilesFromGit(user='derek-dewald',
                        repo='Python_Tools',
                        folder='d_py_functions',
                        output_folder=""):
    '''
    Function to Download Files from Github to a dedicated folder. Specifically used when i DO NOT want to formally link to Github.
    
    Parameters:
        User:
        Repo:
        folder:
        output_folder:
        
    Returns:
        Saves files to Output Folder.
    

    '''
    
    if len(output_folder) == 0:
        output_folder = os.getcwd()
    
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder}"
    response = requests.get(api_url)

    if response.status_code == 200:
        files = response.json()
        py_files = [file for file in files if file['name'].endswith('.py')]

        for file in py_files:
            file_url = file['download_url']
            file_name = file['name']
            file_response = requests.get(file_url)

            if file_response.status_code == 200:
                with open(os.path.join(output_folder, file_name), "w", encoding="utf-8") as f:
                    f.write(file_response.text)
                    
        return True
    else:
        return False

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



def GoogleProcessSheetLinks():
     
    '''
    Function to Google Mapping Sheet, Navigate to Specific Sites.
    Provides Options, Enable Selection based on inputs.
    
    Parameters:
    
        
    Returns:
        
    
    '''

    df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSwDznLz-GKgWFT1uN0XZYm3bsos899I9MS-pSvEoDC-Cjqo9CWeEuSdjitxjqzF3O39LmjJB0_Fg-B/pub?output=csv')
    
    display(df)
    
    p = input('Which Process would You like to review?')
    v = input('What would you like to return?')
    
    df1 = df[df['Definition']==p]
    
    if v.lower() =='link':
        webbrowser.open(df1['Link'].item())
    elif v.lower() == 'csv':
        return pd.read_csv(df1['CSV'].item())
    elif v.lower()=='streamlit':
        webbrowser.open(df1['Streamlit'].item())