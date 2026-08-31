'''
module_word: filesytstem_tools.py
module_definition: Function which stores all functions that deal with the Maintenance and Adminstration of Windows/IOS/Python folders and files. Specifically, includes Python Documentation Functions.

'''
from pathlib import Path
import os

def read_directory(location=None,
                  file_type=None,
                  match_str=None):
                  
    """
    Definition:
        Function which reads reads a directory and returns a list of files included within

    Parameters:
        location (str): The path to the directory. Defaults to the current working directory if not provided.
        file_type (str): The file extension or type to filter by (e.g., '.ipynb'). If empty, returns all files.
        match_str (str): Option to be applied to help filter only wanted files by portion of string condition.

    Returns:
        Dataframe
    Date Created: 
        3-Dec-25
    Date Last Modified: 
        3-Dec-25
    Process: 
        OS Folder Management
    Categorization: 
        Directory Management
    Usage: 
        d_py_function =  '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/'
        read_directory(d_py_function)
    """
    
    # If no folder is provided, use the current working directory
    if location ==None:
        location = os.getcwd() +"\\"
    
    file_list = os.listdir(location)
        
    # If no file type is provided, return all files in the directory
    if file_type:
        file_list = [x for x in file_list if file_type in x]
    
    if match_str:
        file_list = [x for x in file_list if x.find(match_str)!=-1]
    
    return file_list


def move_file_in_folder(folder1,
                        folder2,
                        file_name,
                        overwrite_without_validation=False
                       ):
    
    '''
    Definition:
        Function Created to Help Move Files Between Folders Directly in Python.
        Function will validate that the Folders both exist and there currently isn't a file of the same name, to reduce risk of overwriting
        unexpectedly. (There is a manual override).

    Parameters:
        folder1(str): Folder of First File
        folder2(str): Folder of Second File
        file_name(str): Name of File to be moved, does not matter of file type)
        overwrite_without_validation(bool): Optional Argument allowing user to automate by apply to default overwrite (also meant to help
        reduce risk of losting information due to inadvertent overwriting)

    Returns:
        None

    date_created:09-Feb-26
    date_last_modified: 09-Feb-26
    classification: OS Folder Management
    sub_classification: File Management
    
    usage:
        
        folder1 = '/Users/derekdewald/Documents/Python/Github_Repo/JupyterNotebooks'
        folder2 = '/Users/derekdewald/Documents/Python/Github_Repo/Project Folder/Synthetic Member Dataset'

        move_file_in_folder(folder1,folder2,'Sythentic Member V3.ipynb')

    
    '''

    # Check if Path 1 exists.
    # Check if Path 2 Exists.
    # Check if File Exists in Path 1
    # Check if Files Exists in Path 2 (do not want to overwrite)


    folder1 = Path(folder1)
    folder2 = Path(folder2)

    src_file = folder1 / file_name
    dst_file = folder2 / file_name

    # Check if source folder exists
    if not folder1.exists():
        return f"Source folder does not exist: {folder1}"

    # Check if destination folder exists
    if not folder2.exists():
        return f"Destination folder does not exist: {folder2}"

    # Check if file exists in source
    if not src_file.exists():
        return f"File not found in source folder: {src_file}"

    # Check if file exists in destination
    if dst_file.exists():
        if overwrite_without_validation:
            dst_file.unlink()  # remove existing file
        else:
            return f"File already exists in destination: {dst_file}"

    # Move the file
    shutil.move(str(src_file), str(dst_file))

    return f"File moved successfully: {file_name}"

def txt_to_python(file_name,encoding="utf-8"):

    
    '''
    Definition: 
        Function Used to Import .txt or .py File into Python.
    Parameters: 
        file_name(str): Name of File, including path location for import
        encoding(str): Encoding to be applied by With Open call. Default is utf-8.

    Returns:
        Dataframe
    Date Created:
        3-Dec-25
    Date Last Modified:
        3-Dec-25
    Process:
        OS Folder Management
    Categorization:
        File Management
    usage:
        location = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/DFProcessing.py'
        file = TextFileImport(location)
    
    '''

    with open(file_name, "r", encoding=encoding) as file:
        data = file.read()
    
    return data






