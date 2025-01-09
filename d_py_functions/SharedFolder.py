import os

def ReadDirectory(folder="", file_type=""):
    """
    Reads a directory and returns a list of files.

    Parameters:
    folder (str): The path to the directory. Defaults to the current working directory if not provided.
    file_type (str): The file extension or type to filter by (e.g., '.ipynb'). If empty, returns all files.

    Returns:
    list: A list of files from the directory, optionally filtered by file type.
    """
    
    # If no folder is provided, use the current working directory
    if not folder:
        file_list = os.listdir(os.getcwd())
    else:
        file_list = os.listdir(folder)
    
    # If no file type is provided, return all files in the directory
    if not file_type:
        return file_list
    
    # Return files that match the specified file type
    return [x for x in file_list if file_type in x]
