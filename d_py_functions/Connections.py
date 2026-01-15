import pandas as pd
import datetime
import requests
import os

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
    

def read_git_folder(owner='derek-dewald',repo='Python_Tools',branch='main',folder='d_py_functions'):

    '''

    Program to Extract .py files from a Git Directory.
    Parameters borrowed from Git Mapping Structure, https://github.com/owner/repo/tree/branch/folder

    Parameters:
        owner(str):  As defined in Git Mapping Structure 
        repo(str): As defined in Git Mapping Structure 
        branch(str): As defined in Git Mapping Structure 
        folder(str): As defined in Git Mapping Structure 

    Returns:
        dictionary

    date_created:15-Dec-25
    date_last_modified: 15-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        py_files_in_git_folder = read_git_folder()

    '''
    
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder}?ref={branch}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        items = resp.json()
    except Exception as e:
        print(f'Failed to read folder {url}')
        return []
        
    py_files = [
        {
            "name": item["name"],
            "download_url": item["download_url"]
        }
        for item in items
        if item.get("type") == "file" and item["name"].endswith(".py")
    ]
    return py_files

def read_git_file(git_url):

    """

     Read the contents of a single public GitHub file.

    Parameters:
        git_url(str): Link of Git URL, can be populated using read_git_folder

    Returns:
        List

    date_created:15-Dec-25
    date_last_modified: 15-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        read_git_file('https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/d_py_functions/Connections.py')
    """
    
    try:
        r = requests.get(git_url, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception:
        return ""
    
def download_file_from_git(user='derek-dewald',
                        repo='Python_Tools',
                        folder='d_py_functions',
                        export_folder='/Users/derekdewald/Documents/Python/Github_Repo/CSV Backup Files/'):
    '''
    Function to Download Files from Github to a dedicated folder. 
    Used for ease of access, and when utilizing Git Directly not readily available.

    Parameters:
        User(str): Git Hub User (as defined by URL) 
        Repo:(str): Git Hub Repo (as defined by URL) 
        folder:(str): Git Hub Folder (as defined by URL) 
        output_folder:(str): Folder as to where Files will be Saved, If not defined it will go to Current Dir.

    Returns:
        .py files to Windows Folder

    date_created:1-Jan-25
    date_last_modified: 30-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        download_file_from_git()
        

    '''
    
    if len(export_folder) == 0:
        export_folder = os.getcwd()
    
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
                with open(os.path.join(export_folder, file_name), "w", encoding="utf-8") as f:
                    f.write(file_response.text)
                    
        return True
    else:
        return False

def backup_google_worksheets(export_folder='/Users/derekdewald/Documents/Python/Github_Repo/CSV Backup Files/'):
    '''
    Definition of Function

    Parameters:
        List of Parameters

    Returns:
        Object Type

    date_created:30-Dec-25
    date_last_modified: 30-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        Example Function Call
    
    '''

    from data_d_dicts import links

    month_str = datetime.datetime.now().strftime('%b-%y')


    df = pd.read_csv(links['google_mapping_sheet_csv'])
    
    google_df_dict = {}

    for row in df[df['CSV'].notnull()].index:
        link = df.iloc[row]['CSV']
        name = df.iloc[row]['Definition']
        google_df_dict[name] = pd.read_csv(link)

        if export_folder:
            google_df_dict[name].to_csv(f"{export_folder}/{name}_{month_str}.csv",index=False)