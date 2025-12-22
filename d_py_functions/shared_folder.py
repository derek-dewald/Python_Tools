import pandas as pd
import numpy as np
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


def parse_dot_py_folder(location=None,
                        export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/DataDictionary/'):
    '''
    
    Function which Allows for the Quick Review of All Python Functions in a Particular Directory, using the functions 
    Read Directory and ParseDDotPYFile

    Parameters:
        location (str): Windows or Mac OS Folder Directory (defaults to D's Mac Directory)
        export_location(str): Location to where CSV file is to be exported. If left Blank, will not export a CSV.

    Returns:
        DataFrame

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        function_list, function_parameters = parse_dot_py_folder()
    
    
    '''

    # GEnerate List of Files

    function_list = pd.DataFrame()
    function_parameters = pd.DataFrame()
    

    if not location:
        folder = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions'

    func_list = read_directory(folder,file_type='.py')
    func_list = [x for x in func_list if (x.find('init')==-1) | (x not in ['d_lists','d_strings','d_dictionaries'])]

    for file_name in func_list:
        filename = f"{folder}/{file_name}"
        file_ = text_file_import(filename)

        temp_a,temp_b = parse_dot_py_file(file_)
        temp_a['Folder'] = file_name
        temp_b['Folder'] = file_name

        function_list = pd.concat([function_list,temp_a])
        function_parameters = pd.concat([function_parameters,temp_b])


    temp_param = pd.concat([
        input3(data_d_strings),
        input1(data_d_dicts),
        input2(data_d_lists)
    ])

    function_parameters = pd.concat([
        function_parameters,
        temp_param
    ])

    if export_location:
        print(f'python_function_list Saved to {export_location}')
        print(f'python_function_parameters Saved to {export_location}')
        function_list.to_csv(f'{export_location}python_function_list.csv',index=False)
        function_parameters.to_csv(f'{export_locaiton}python_function_parameters.csv',index=False)

    return function_list,function_parameters


def create_py_table_dict(base_location= '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/',
                         export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/DataDictionary/folder_listing.csv'):
    
    '''
    Function which Generates a Dataframe representing a Function Dictionary, sourcing the Functions from a Shared Folder Location, and
    using the definitions sourced from a Python Dictionary

    Parameters:
        base_location (str): Location of Windows Directory containing .py Files.

    Returns:
        DataFrame

    date_created:4-Dec-25
    date_last_modified:4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        python_function_dict_df = create_py_table_dict()
    '''
    from data_d_dicts import function_table_dictionary

    # Get Defined Functions from Dictionary Reference Listing
    temp_ = dict_to_dataframe( function_table_dictionary,key_name='Function Name',value_name='Definition')
    temp_['Type'] = 'Definition'

    py_functions = list_to_dataframe([x for x in read_directory(base_location,file_type='.py') if (x.find('init')==-1)],column_name_list=['File Name'])
    py_functions['Source'] = 'PY File'
    py_functions['Function Name'] = py_functions['File Name'].apply(lambda x:x.replace('.py',''))

    final_df = py_functions.merge(temp_,on='Function Name',how='outer')


    if export_location:
        print(f'folder_listing Saved to {export_location}')
        final_df.to_csv(export_location,index=False)

    return final_df