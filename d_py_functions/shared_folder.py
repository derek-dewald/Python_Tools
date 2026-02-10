from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import ast
import os
import re


# Import Location of D Python Functions
import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from dict_processing import dict_to_dataframe
from list_processing import list_to_dataframe
import data_d_dicts, data_d_lists,data_d_strings
from data_d_lists import function_fields
from input_functions_ignore import input1,input2,input3

def read_directory(location=None,
                  file_type=None,
                  match_str=None):
                  
    """
    Function which reads reads a directory and returns a list of files included within

    Parameters:
        location (str): The path to the directory. Defaults to the current working directory if not provided.
        file_type (str): The file extension or type to filter by (e.g., '.ipynb'). If empty, returns all files.
        match_str (str):

    Returns:
        Dataframe containing a listing of selected files.
    
        
    date_created: 3-Dec-25
    date_last_modified: 3-Dec-25
    classification: TBD
    sub_classification: TBD
    usage: 
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

def text_file_import(file_name,encoding="utf-8"):

    
    '''
    Function Used to Import .txt or .py File into Python.

    Parameters:
        Name of File, including Directory.

    Returns:
        Str

    date_created:3-Dec-25
    date_last_modified:3-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        location = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/DFProcessing.py'
        file = TextFileImport(location)
    
    '''

    
    with open(file_name, "r", encoding=encoding) as file:
        data = file.read()
    
    return data

def parse_dot_py_file(file_text):
    """
    Function which reads a Python file (as text) and provides a summary of
    functions and their components, using a structured docstring format.

    Parameters:
        file_text (str): Full text of a .py file.

    Returns:
        function_list (DataFrame): One row per function with metadata.
        function_parameters (DataFrame): One row per parameter per function.

    date_created:4-Dec-25
    date_last_modified:4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        function_list, function_parameters = ParseDDotPYFile(file_text)
    """

    # Case-insensitive mapping: lowercased key -> canonical column name
    # e.g. "usage" -> "usage", "date_created" -> "date_created"
    metadata_map = {key.lower(): key for key in function_fields}

    tree = ast.parse(file_text)

    meta_rows = []   # for df_meta / function_list
    param_rows = []  # for df_params / function_parameters

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            # Skip imports, classes, top-level code, etc.
            continue

        func_name = node.name
        docstring = ast.get_docstring(node) or ""
        arg_names = [a.arg for a in node.args.args]

        # ---------- DF1 base record ----------
        meta_record = {
            "Function": func_name,
            "Purpose": "",
            "Parameters": arg_names,  # list from signature
            "Returns": None,
        }
        # Initialize all known metadata fields to None
        for col in function_fields:
            meta_record[col] = None

        # ---------- DF2 per-function records ----------
        # Start by creating one row per signature arg with blanks
        param_records = {
            arg: {
                "Function": func_name,
                "Parameters": arg,
                "Type": "",
                "Definition": "",
            }
            for arg in arg_names
        }

        # ---------- Parse docstring ----------
        current_section = "description"
        current_metadata_key = None  # for multi-line metadata blocks like 'usage'
        description_lines = []

        for raw_line in docstring.split("\n"):
            # Keep right side whitespace, strip left & right for logic
            line = raw_line.rstrip()
            stripped = line.strip()

            # Blank line: ends any multi-line metadata block; section unchanged
            if not stripped:
                current_metadata_key = None
                continue

            lower = stripped.lower()

            # --- 0) If we're inside a multi-line metadata block, keep capturing ---
            if current_metadata_key is not None:
                # If this looks like a new metadata key or a section header, close the block
                if ":" in stripped:
                    key_part, _ = stripped.split(":", 1)
                    key_norm = key_part.strip().lower()
                    if key_norm in metadata_map or lower in ("parameters:", "returns:"):
                        current_metadata_key = None
                    else:
                        existing = meta_record.get(current_metadata_key) or ""
                        meta_record[current_metadata_key] = (
                            (existing + "\n" if existing else "") + stripped
                        )
                        continue
                else:
                    # No colon: still part of the same metadata block
                    existing = meta_record.get(current_metadata_key) or ""
                    meta_record[current_metadata_key] = (
                        (existing + "\n" if existing else "") + stripped
                    )
                    continue

            # --- 1) Metadata lines: key: value (outside Parameters section) ---
            if ":" in stripped and current_section != "parameters":
                key_part, value_part = stripped.split(":", 1)
                key_norm = key_part.strip().lower()

                if key_norm in metadata_map:
                    col_name = metadata_map[key_norm]
                    value = value_part.strip()

                    # Initialize field with this first line
                    meta_record[col_name] = value

                    # Treat all metadata keys as potentially multi-line (usage, etc.)
                    current_metadata_key = col_name
                    continue

            # --- 2) Section headers ---
            if lower.startswith("parameters:"):
                current_section = "parameters"
                current_metadata_key = None
                continue

            if lower.startswith("returns:"):
                current_section = "returns"
                current_metadata_key = None
                continue

            # --- 3) Parameters section: name(type): description ---
            if current_section == "parameters":
                match = re.match(r"(\w+)\s*\((.*?)\)\s*:\s*(.*)", stripped)
                if match:
                    pname, ptype, pdesc = match.groups()
                    if pname in param_records:
                        param_records[pname]["Type"] = ptype
                        param_records[pname]["Definition"] = pdesc
                else:
                    # continuation line for the last param with a non-empty Definition
                    for p in reversed(arg_names):
                        if param_records[p]["Definition"]:
                            param_records[p]["Definition"] += " " + stripped
                            break
                continue

            # --- 4) Returns section: only the block under "Returns:" ---
            # This will naturally stop when we hit:
            # - a blank line  -> handled at the top
            # - a metadata line (key: value) -> handled before this block next iteration
            # - a new section header ("Parameters:", etc.) -> handled above
            if current_section == "returns":
                if meta_record["Returns"] is None:
                    meta_record["Returns"] = stripped
                else:
                    meta_record["Returns"] += " " + stripped
                continue

            # --- 5) Otherwise: description / purpose ---
            description_lines.append(stripped)

        meta_record["Purpose"] = "\n".join(description_lines).strip()
        meta_rows.append(meta_record)

        # Add this function's parameter rows to the global df2 list
        param_rows.extend(param_records.values())

    function_list = pd.DataFrame(meta_rows)
    function_parameters = pd.DataFrame(param_rows)

    return function_list, function_parameters



def move_file_in_folder(folder1,
                        folder2,
                        file_name,
                        overwrite_without_validation=False
                       ):
    
    '''
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
    classification:TBD
    sub_classification:TBD
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
