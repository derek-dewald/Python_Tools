import pandas as pd
import requests

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
