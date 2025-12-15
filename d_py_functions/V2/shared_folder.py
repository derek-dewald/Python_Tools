# File Description: File related to Foldes and Files on Local Computer.
import pandas as pd
import numpy as np

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from text_processing import PythonStringParser_AST

from concurrent.futures import ThreadPoolExecutor
import shutil
import ast
import os

def DuplicateFileorFolder(source_path, destination_path):
    """
    Function to copy a file or folder to another location while handling errors.

    Parameters:
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
                  create_df=0,
                  number_of_CPUs=1):
                  
    """
    Function which reads reads a directory and returns a list of files included within

    Parameters:
    folder (str): The path to the directory. Defaults to the current working directory if not provided.
    file_type (str): The file extension or type to filter by (e.g., '.ipynb'). If empty, returns all files.

    Returns:
    list: A list of files from the directory, optionally filtered by file type.
    """
    
    # If no folder is provided, use the current working directory
    if location ==None:
        location = os.getcwd() +"\\"
    
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
        if number_of_CPUs==1:
            final_df = pd.DataFrame()
            for file in file_list:
                if file_type=='csv':
                    final_df = pd.concat([final_df,pd.read_csv(f"{location}{file}")])
                elif file_type == 'xlsx':
                    final_df = pd.concat([final_df,pd.read_excel(f"{location}{file}")])
            return final_df
        
        else:
            if file_type=='csv':
                def read_file(file):
                    return pd.read_csv(file)
            else:
                def read_file(file):
                    return pd.read_excel(file)
            
            file_list = [f"{location}{x}" for x in file_list]
            
            with ThreadPoolExecutor(max_workers=number_of_CPUs) as executor:
                dfs = list(executor.map(read_file,file_list))
            
            return pd.concat(dfs,ignore_index=True)  
        


def crawl_directory_with_progress(root_dir, progress_step=5,print_=1):
    """
    Recursively crawl through a directory, track progress every `progress_step` percent.
    
    Parameters:
        root_dir (str): Root directory to start from.
        progress_step (int): Percent steps at which to report progress (e.g. 5 for 5%).

    Returns:
        pd.DataFrame: DataFrame with file path, name, and type.
    """
    start_time = time.time()
    file_records = []

    # Step 1: Count total directories to visit (quick)
    total_dirs = sum(1 for _ in os.walk(root_dir))
    
    print(total_dirs)
    
    if total_dirs == 0:
        print("No directories found.")
        return pd.DataFrame()

    # Progress tracking
    next_progress_mark = progress_step
    visited_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        visited_dirs += 1

        for file in filenames:
            file_path = os.path.join(dirpath, file)
            file_name = os.path.basename(file)
            file_ext = os.path.splitext(file)[1].lower().lstrip('.')  # remove dot
            
            file_records.append({
                'file_path': file_path,
                'file_name': file_name,
                'file_type': file_ext
            })

        # Print progress every 5%
        if print_==1:
            percent_complete = (visited_dirs / total_dirs) * 100
            if percent_complete >= next_progress_mark:
                elapsed = time.time() - start_time
                estimated_total = elapsed / (percent_complete / 100)
                remaining = estimated_total - elapsed

                print(f"[{percent_complete:.1f}%] Done - "
                      f"Elapsed: {elapsed:.1f}s - "
                      f"ETA: {remaining:.1f}s remaining "
                      f"(Total est: {estimated_total:.1f}s)")

                next_progress_mark += progress_step

    total_time = time.time() - start_time
    if print_==1:
        print(f"✅ Done. Total time: {total_time:.2f} seconds. Files found: {len(file_records)}")
    return pd.DataFrame(file_records)


def MakeFolder(folder,
               path_):
    
    location = f"{path_}{folder}\\"
    
    if os.path.exists(f"{location}"):
        print(f"{location} exits")
    else:
        os.makedirs(f"{location}")
        print('New Folder Created')

def ExtractPythonFunctionDetail(file_path, keywords=None):
    """
    Extracts function details from a .py file using AST parsing.
    
    Parameters:
        file_path (str): Path to the Python file.
        keywords (list): List of keywords to look for in docstrings (default: [parameters:, returns:, date created:, date last modified:]).
        
    Returns:
        pd.DataFrame: DataFrame with function details.

    Date Created: August 17, 2025.
    Date Last Modified: 
    """
    if keywords is None:
        keywords = ["parameters:", "returns:", "date created:", "date last modified:"]

    # Normalize keywords (all lowercase, ensure ":" at end)
    keywords = [k.lower() if k.endswith(":") else k.lower() + ":" for k in keywords]

    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    tree = ast.parse(file_content)
    function_data = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            docstring = ast.get_docstring(node) or "No description available"
            function_code = ast.get_source_segment(file_content, node).strip()

            # Parse docstring content
            doc_lines = docstring.split("\n")
            description_text = []
            sections = {k[:-1].capitalize(): [] for k in keywords}  # e.g. "Returns" -> []

            current_section = None
            for line in doc_lines:
                stripped = line.strip()
                low = stripped.lower()
                # Detect new section
                if any(low.startswith(k) for k in keywords):
                    for k in keywords:
                        if low.startswith(k):
                            current_section = k[:-1].capitalize()
                            # ✅ Remove only the keyword prefix
                            content = stripped[len(k):].strip()
                            sections[current_section].append(content)
                            break
                elif stripped:  # inside a section or description
                    if current_section:
                        sections[current_section].append(stripped)
                    else:
                        description_text.append(stripped)

            # Build record
            function_data.append({
                "Function Name": function_name,
                "Description": " ".join(description_text).strip(),
                **{sec: " ".join(val).strip() for sec, val in sections.items()},
                "Code": function_code
            })

    return pd.DataFrame(function_data)


def ExtractPythonFiles(folder='/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/',export_file='D_Python_Functions'):
    
    '''
    Function to Read a Specifically Determine folder, to look for all of the .py files in it and read them, 
    using the function ExtractPythonFunctionDetail.

    Parameters:
        folder (str): Folder location of a series of .py files to be Read. Default Location, '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/'
        export_file (str): Name of Excel File to Be export if included, by default it will not exclude a file.

    Returns:
        pd.DataFrame() with Listing of All functions read, in format, Function Name, Description, Parameters, Returns, Date Created, Date Last Modified, Code

    Date Created: August 17, 20225
    Date Last Modified: 

    '''

    files = ReadDirectory(folder)
    files = [x for x in files if (x.find('.py')!=-1)&(x.find('__')==-1)]

    final_df = pd.DataFrame()

    for file in files:
        temp_df = ExtractPythonFunctionDetail(f"{folder}{file}")
        temp_df['File'] = file
        final_df = pd.concat([final_df,temp_df]).reset_index(drop=True)

    if export_file:
        final_df.to_excel(f"{folder}{export_file}.xlsx",index=False)

    return final_df


def ReadPythonFiles(location='/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/',file_list=None):
    
    '''
    Function which reads a Folder, iterating through a list of Files, specifically looking for .py files utilizing the Extract Function Details AST.
    Function Looks for Parameters, Returns, Date Created and Date Last Modified.
 
    Parameters:
         location (str): Folder
         
    Returns:
        Dataframe of Functions with description.

    Date Created: 
    Date Last Modified: 25-Aug-25

    '''    
    py_file_dict = {}

    if not file_list:
        file_list = [x for x in ReadDirectory(location) if x.find('.py')!=-1 and x.find('__')==-1]
    
    for file_name in file_list:
        with open(f"{location}{file_name}", "r", encoding="utf-8") as file:
            data = file.read()
            py_file_dict.update(PythonStringParser_AST(data,file_name))
            
    return pd.DataFrame(py_file_dict).T