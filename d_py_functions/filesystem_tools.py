'''
module_name: filesystem_tools
module_purpose:  

default_structure: 
module_guidance: 

'''

import pandas as pd
import numpy as np
import datetime
import os
import ast
import re

from objects_manual import object_dict as object_dict_manual

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
        ETL
    Categorization: 
        File Manipulation/ Management
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
        ETL
    Categorization:
        File Manipulation/ Management
    usage:
        location = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/DFProcessing.py'
        file = TextFileImport(location)
    
    '''

    with open(file_name, "r", encoding=encoding) as file:
        data = file.read()
    
    return data

def print_python_str_template(object_dict=object_dict_manual):
    
    '''
    Definition: 
        Display default python documentation string in a easy to copy and paste format for creation of new functions.
    Parameters: 
        object_dict(list): List of Default Parameters to be iterated through. 
    Returns: 
        Console Text
    Date Created:   
        06-Jul-26
    Date Last Modified:
        06-Jul-26
    Process: 
        Documentation
    Categorization: 
        Python String Documentation
    Usage: 
        print_python_str_template()
    Notes:
    
    '''

    for x in object_dict['python_str_documentation']['python_object']:
        if x in ['Date Created','Date Last Modified']:
            print(f"{x}:\n\t{datetime.datetime.now().strftime('%d-%b-%y')}")
        else:
            print(f"{x}:\n\tDefinition")  


def parse_dot_py_file(
    file_text, 
    function_columns=None
):
    """
    Definition:
    	Function which parses a .py file which has been read into python into a DataFrame categorizing function for ease of articulation and classification
    Parameters:
    	file_text(str): Text, which should be structured someone in format of function_columns, based on prevailing default Python String Documentation
        function_columns(list): List of columns to include in output file and to be parsed on. Default is to import from object_dict which represents default Python String Documentation.
    Returns:
    	Excel File(s)
    Date Created:
    	05-Aug-25
    Date Last Modified:
    	06-Jul-26
    Process:
    	Documentation
    Categorization:
    	TBD
    Usage:
    	location = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/'
        file_text = txt_to_python(f"{location}/filesystem_tools.py")
        a,b = parse_dot_py_file(file_text)
    Notes:
    	Update on 6-Jul-26 included generalization of List, and Formation of Tab/Indent structure.

    """

    if function_columns is None:
        function_columns = object_dict_manual['python_str_documentation']['python_object']

    # Metadata fields are everything except the core fields
    core_fields = {"Function", "Purpose", "Parameters", "Returns"}
    metadata_fields = [c for c in function_columns if c not in core_fields]

    metadata_map = {key.lower(): key for key in metadata_fields}

    tree = ast.parse(file_text)

    meta_rows = []
    param_rows = []

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        func_name = node.name
        docstring = ast.get_docstring(node) or ""
        arg_names = [a.arg for a in node.args.args]

        # Initialize all columns from one master list
        meta_record = {col: None for col in function_columns}

        meta_record["Function"] = func_name
        meta_record["Purpose"] = ""
        meta_record["Parameters"] = arg_names
        meta_record["Returns"] = None

        param_records = {
            arg: {
                "Function": func_name,
                "Parameters": arg,
                "Type": "",
                "Definition": "",
            }
            for arg in arg_names
        }

        current_section = "description"
        current_metadata_key = None
        description_lines = []

        for raw_line in docstring.split("\n"):
            line = raw_line.rstrip()
            stripped = line.strip()

            if not stripped:
                current_metadata_key = None
                continue

            lower = stripped.lower()

            # Continue multi-line metadata block
            if current_metadata_key is not None:
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
                    existing = meta_record.get(current_metadata_key) or ""
                    meta_record[current_metadata_key] = (
                        (existing + "\n" if existing else "") + stripped
                    )
                    continue

            # Metadata lines
            if ":" in stripped and current_section != "parameters":
                key_part, value_part = stripped.split(":", 1)
                key_norm = key_part.strip().lower()

                if key_norm in metadata_map:
                    col_name = metadata_map[key_norm]
                    meta_record[col_name] = value_part.strip()
                    current_metadata_key = col_name
                    continue

            # Section headers
            if lower.startswith("parameters:"):
                current_section = "parameters"
                current_metadata_key = None
                continue

            if lower.startswith("returns:"):
                current_section = "returns"
                current_metadata_key = None
                continue

            # Parameters
            if current_section == "parameters":
                match = re.match(r"(\w+)\s*\((.*?)\)\s*:\s*(.*)", stripped)

                if match:
                    pname, ptype, pdesc = match.groups()

                    if pname in param_records:
                        param_records[pname]["Type"] = ptype
                        param_records[pname]["Definition"] = pdesc
                else:
                    for p in reversed(arg_names):
                        if param_records[p]["Definition"]:
                            param_records[p]["Definition"] += " " + stripped
                            break

                continue

            # Returns
            if current_section == "returns":
                if meta_record["Returns"] is None:
                    meta_record["Returns"] = stripped
                else:
                    meta_record["Returns"] += " " + stripped
                continue

            # Purpose / description
            description_lines.append(stripped)

        meta_record["Purpose"] = "\n".join(description_lines).strip()

        meta_rows.append(meta_record)
        param_rows.extend(param_records.values())

    function_list = pd.DataFrame(meta_rows)
    function_list = function_list.reindex(columns=["Function"] + function_columns)

    function_parameters = pd.DataFrame(param_rows)

    return function_list, function_parameters


def parse_dot_py_folder(location=None,
                        export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/'):
    '''
    Definition:
    	Function which sacles parse_dot_py_file to folder from file.
    Parameters:
    	location(str): Location of Folder to be read for .py files. Default is defined as /Users/derekdewald/Documents/Python/Github_Repo/d_py_functions
        export_location(str): Location to where CSV file is to be exported. If left Blank, will not export a CSV.
    Returns:
    	DataFrame(s)
        
    Date Created:
    	4-Dec-25
    Date Last Modified:
    	06-Jul-26
    Process:
    	Documentation
    Categorization:
    	Python String Documentation	
    Usage:
    	function_list,parameter_list = parse_dot_py_folder(export_location=False)
    Notes:
    	Definition
            
    '''

    # GEnerate List of Files

    function_list = pd.DataFrame()
    function_parameters = pd.DataFrame()
    

    if not location:
        folder = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions'

    func_list = read_directory(folder,file_type='.py')
    
    for file_name in func_list:
        filename = f"{folder}/{file_name}"
        file_ = txt_to_python(filename)

        temp_a,temp_b = parse_dot_py_file(file_)
        temp_a['Folder'] = file_name
        temp_b['Folder'] = file_name

        function_list = pd.concat([function_list,temp_a])
        function_parameters = pd.concat([function_parameters,temp_b])

    if export_location:
        print(f'python_function_list Saved to {export_location}')
        print(f'python_function_parameters Saved to {export_location}')
        function_list.to_csv(f'{export_location}python_function_list.csv',index=False)
        function_parameters.to_csv(f'{export_location}python_function_parameters.csv',index=False)

    return function_list,function_parameters



