import os
import shutil


def DuplicateFileorFolder(source_path, destination_path):
    """
    Function to copy a file or folder to another location while handling errors.

    Args:
        source_path (str): Path to the file or folder to copy.
        destination_path (str): Destination path where the file or folder should be stored.
    
    Returns:
        None
    
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path '{source_path}' does not exist.")

    if os.path.isfile(source_path):
        try:
            shutil.copy2(source_path, destination_path)
        except PermissionError:
            print(f"Skipped (Permission Denied): {source_path}")
        except Exception as e:
            print(f"Failed to copy file: {source_path}. Error: {e}")

    elif os.path.isdir(source_path):
        if not os.path.exists(destination_path):
            os.makedirs(destination_path, exist_ok=True)  # Ensure destination directory exists

        for root, dirs, files in os.walk(source_path):
            # **Skip `.git` directories**
            if ".git" in root.split(os.sep):
                print(f" Skipping `.git` folder: {root}")
                continue  

            for dir_name in dirs:
                source_dir = os.path.join(root, dir_name)
                dest_dir = os.path.join(destination_path, os.path.relpath(source_dir, source_path))

                try:
                    os.makedirs(dest_dir, exist_ok=True)
                except PermissionError:
                    print(f"Skipped directory (Permission Denied): {dest_dir}")
                except Exception as e:
                    print(f"Failed to create directory: {dest_dir}. Error: {e}")

            for file_name in files:
                source_file = os.path.join(root, file_name)
                dest_file = os.path.join(destination_path, os.path.relpath(source_file, source_path))

                try:
                    shutil.copy2(source_file, dest_file)
                except PermissionError:
                    print(f"Skipped file (Permission Denied): {source_file}")
                except Exception as e:
                    print(f"Failed to copy file: {source_file}. Error: {e}")

    else:
        raise ValueError(f"Source path '{source_path}' is neither a file nor a folder.")

    print(f"Finished copying '{source_path}' to '{destination_path}'.")


def ReadDirectory(location=None,
                  file_type=None,
                  match_str=None,
                  create_df=0):
                  
    """
    Function which reads reads a directory and returns a list of files included within

    Args:
    folder (str): The path to the directory. Defaults to the current working directory if not provided.
    file_type (str): The file extension or type to filter by (e.g., '.ipynb'). If empty, returns all files.

    Returns:
    list: A list of files from the directory, optionally filtered by file type.
    """
    
    # If no folder is provided, use the current working directory
    if location ==None:
        file_list = os.listdir(os.getcwd())
    else:
        file_list = os.listdir(location)
        
    # If no file type is provided, return all files in the directory
    if file_type !=None:
        file_list = [x for x in file_list if file_type in x]
    
    if match_str !=None:
        file_list = [x for x in file_list if x.find(match_str)!=-1]
        
    if create_df ==0:                  
        # Return files that match the specified file type
        return file_list
    else:
        final_df = pd.DataFrame()
        for file in file_list:
            if file_type=='csv':
                final_df = pd.concat([final_df,pd.read_csv(f"{location}{file}")])
            elif file_type == 'xlsx':
                final_df = pd.concat([final_df,pd.read_excel(f"{location}{file}")])
        return final_df