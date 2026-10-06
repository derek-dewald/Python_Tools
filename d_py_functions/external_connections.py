'''
module_word: external_connections
module_definition: Functions which connect to external sites, repos, etc.

'''
import requests


def extract_py_folder_from_git_repo(
    folder_git_url,
    output_location=None,
    file_extension='.py',
    
):
    '''
    Definition:
        Function Used to Extract Files from Git Hub Manually. Where a direct Git Connection can not be utilzied.
    Parameters:
        folder_git_url(str): Link to Git Repo
        output_location(str): Location to Export Files. If Blank, then returns to local directory where file is read from.
        file_extension(str): Filter Applied to limit types of file
    Returns:
        TXT Files
    Date Created:
        05-Oct-26
    Date Last Modified:
        05-Oct-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        extract_py_folder_from_git_repo(
            folder_git_url='https://api.github.com/repos/derek-dewald/Python_Tools/contents/d_py_functions',
            output_location='/Users/derekdewald/Documents/Python/Github_Repo/JupyterNotebooks/test/')
    Notes:
        None
    Required Functions:
        None

    '''
    
    response = requests.get(folder_git_url)
    response.raise_for_status()
    files = response.json()
    
    for file in files:
        if (file['name'].find(file_extension)!=-1)&(file['name'].find('__init__')==-1):
            file_name = file['name']
            file_url = file['download_url']
        
            file_response = requests.get(file_url)
            file_response.raise_for_status()

            if output_location:
                file_name = f"{output_location}{file_name}"
            
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(file_response.text)
                print(file_name,file_url)