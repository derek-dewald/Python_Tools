'''
module_word: taxonomy_admin_functions.py
module_definition: Functions which deal primarily with administration and maintenance of Taxonomy supporting Documentation. Generalized Functions not to be included.

'''


import pandas as pd
import numpy as np
import datetime
import ast
import re

from objects_automated import object_dict
from filesystem_tools import read_directory
from object_rendering import txt_to_python

def create_knowledge_base(
    df=None,
    columns = ['Process','Categorization','Word','Definition','Order'],
    file_name='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx'
):
    '''

    Definition:
        Process Utilized to create Knowledge Base. 
        Knowledge base has had multiple iterations, currently it is Definitions, with Process Steps Merged back into Process to Make a completely and simply filterable file

    Parameters:
        df(dataframe): File, if None it will call Google Sheet.
        columns(list): Default columns to include in extract
        file_name(str): File Name of Export File, If Blank it WILL NOT export a file.
    Returns:
        knowledge_base.xlsx
    Date Created:
        02-Jul-26
    Date Last Modified:
        7-Oct-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        create_knowledge_base()
    Notes:
        22-Jul - Overhauled merge. Attempted to streamline, simplify and reduce duplication. Increase Visability.
        28-Jul - Originally Named - generate_knowledge_base, stream lined to remove usage of Dictionary, which were complex and operationally inefficient.
        29-Jul - Changed Foundational Structure. Simplified with Clarity on Input.
        26-Sep - Tested full functionality.
        10-Oct - With a material restructure of Definitions/Notes. Was able to drastically simplify this process.
        
    Required Functions:
        None
    '''

    if len(df)>1:
        df = pd.read_csv(object_dict['csv_links']['python_object']['google_definition_csv'])

    # Items which are Both Process and Process Steps will have Definition in Process Step Overwriden. As process, it should not be updated here.

    process_def = df[df['Categorization']=='Process'][['Process','Definition']].drop_duplicates('Process').rename(columns={'Process':'Word','Definition':"Definition_"})
    df = df.merge(process_def,on='Word',how='left')
    df['Definition'] = np.where(
        (df['Categorization']=='Process Step')&
        (df['Definition_'].notnull())
        ,df['Definition_'],df['Definition'].fillna(''))
    
    df.drop('Definition_',axis=1,inplace=True)
    
    df['Order'] = df['Order'].fillna(0)
    
    df1 = df[df['Categorization']=='Process Step'][columns].drop('Categorization',axis=1)
    insert_df = df1.drop('Definition',axis=1).merge(df1.rename(columns={'Word':'Word_OG','Process':'Word','Order':'Order_OG'}),on='Word',how='inner')
    
    insert_df['Categorization'] = insert_df['Word']
    insert_df['Word'] = insert_df['Word_OG'].copy()
    insert_df = insert_df[['Process','Categorization','Word','Definition','Order','Order_OG']].copy()
    
    final_df = pd.concat([df[columns],insert_df])
    final_df['Order_OG'] = final_df['Order_OG'].fillna(0)
    final_df = final_df.sort_values(['Process','Order','Order_OG'])
    final_df.drop(['Order','Order_OG'],axis=1,inplace=True)
    
    if file_name:
        try:
            final_df.to_excel(file_name,index=False)
        except:
            pass

def create_blank_function_doc_string():

    '''
    Definition:
        Function which generates Default Documentation for Python Function Documentaiton in consistent format.
    Parameters:
        df(dataframe): Knowledge base Dataframe. If None it will pull from either Local or Git Hub.
    Returns:
        str
    Date Created:
        28-Aug-26
    Date Last Modified:
        28-Aug-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        create_blank_function_doc_string()
    Notes:
        None
    Required Functions:
        None


    '''

    parameters = object_dict['python_string_documentation']['python_object']
    
    text = ""
    for item in parameters:
        if item in ['Date Created','Date Last Modified']:
            text += f"\n{item}:\n    {datetime.datetime.now().strftime('%d-%b-%y')}"
        elif item in ['Notes','Required Functions']:
            text += f"\n{item}:\n    None"

        else:
            text += f"\n{item}:\n    TBD"
    print(text)

###########################################################

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
    	OS Folder Management
    Categorization:
    	File Management
    Usage:
    	location = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/'
        file_text = txt_to_python(f"{location}/filesystem_tools.py")
        a,b = parse_dot_py_file(file_text)
    Notes:
    	Update on 6-Jul-26 included generalization of List, and Formation of Tab/Indent structure.

    """

    if function_columns is None:
        function_columns = object_dict['python_string_documentation']['python_object']

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
                    current_section = "metadata" ## ADDED 26-Sep-26 In attempt to reduce Sentence Run.
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

def create_python_documentation(
        location=None,
        export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/'
        ):
    '''
    Definition:
    	Function which Generates Documentation related to Python Functions, which are annotated in a specific manner.

    Parameters:
    	location(str): Location of Folder to be read for .py files. Default is defined as /Users/derekdewald/Documents/Python/Github_Repo/d_py_functions
        export_location(str): Location to where CSV file is to be exported. If left Blank, will not export a CSV.
    Returns:
    	dataframe
        python_function_list.csv
        python_function_parameters.csv
        python_function_file_definition.csv
        
    Date Created:
    	4-Dec-25
    Date Last Modified:
    	06-Jul-26
    Process:
    	OS Folder Management
    Categorization:
    	File Management
    Usage:
    	function_list,parameter_list,function_file_definition = create_python_documentation(export_location=False)
    Notes:
        None
    Required Functions:
        Read Directory
    	create_df_dot_py_doc_string
        txt_to_python
        parse_dot_py_file
            
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

    # Generate Overall File Definition File
    function_file_definition = create_df_dot_py_doc_string()

    if export_location:
        function_list.to_csv(f'{export_location}python_function_list.csv',index=False)
        function_parameters.to_csv(f'{export_location}python_function_parameters.csv',index=False)
        function_file_definition.to_csv(f'{export_location}python_function_file_definition.csv',index=False)

    return function_list,function_parameters,function_file_definition


def create_df_dot_py_doc_string(location=None):
    
    '''
    Definition:
    	Function which iterates over a directory, looking for .py files which it can read the File Doc String, to understand the .py Purpose.

    Parameters:
    	location(str): Name of Windows/IOS Folder.
    Returns:
    	DataFrame(s)
    Date Created:
    	28-Aug-26
    Date Last Modified:
    	28-Aug-26
    Process:
    	TBD
    Categorization:
    	TBD
    Usage:
    	df = create_df_dot_py_doc_string()
    Notes:
        None
    Required Functions:
    	Read Directory
        read_dot_py_doc_string
    '''

    if location is None:
        location = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions'

    file_list = read_directory(location)
    file_list = [x for x in file_list if (x.find('.py')!=-1) & (x not in ['__init__.py','__pycache__'])]

    file_dict = {}
    
    for file in file_list:
        file_location = f'{location}/{file}'
        file_dict[file] = read_dot_py_doc_string(file_location)
    
    return pd.DataFrame(file_dict).T.drop('Word',axis=1).reset_index().rename(columns={'index':"Function"})

def read_dot_py_doc_string(file):
    '''
    Definition:
    	Function which reads the Doc String of .py files, specifically looking for text which can be used to documents the function Name and Purpose.

    Parameters:
    	file(str): Name of file as documented in local Folder.
        
    Returns:
    	DataFrame(s)
        
    Date Created:
    	28-Aug-26
    Date Last Modified:
    	28-Aug-26
    Process:
    	TBD
    Categorization:
    	TBD
    Usage:
    	df = read_dot_py_function_notes()
    Notes:
    	None
    Required Functions:
        None
    
    '''
    
    text = txt_to_python(file)

    fields = [
        'module_word',
        'module_definition'
    ]

    file_info = {}

    for field in fields:
        pattern = rf'{field}:\s*(.*?)(?=\nmodule_\w+:|\'\'\'|\"\"\"|$)'
        match = re.search(pattern, text, re.DOTALL)

        file_info[field.removeprefix('module_').title()] = (
            ' '.join(match.group(1).split())
            if match else None
        )

    return file_info

def generate_objects_automated_py(
    links_df=None,
    knowledge_base_df=None
):
    '''
    Definition:
        Create Automated Python File
    Parameters:
        links_df(df): Data from Google Sheets with Location of CSV and Links. If Nothing, it will pull from Google.
        consolidated_df (df): Dataset created from extract_consolidated_raw_dataset.py. If nothing, it will pull from Local Source.
        dot_py_documentation (df): Dataset created from _______, representing Python String Documentation. If nothing, it will pull from local Source. 
        definition_df (df): Data from Google Sheets. If nothing will pull from Google.
    Returns:
        Dot Py File
    Date Created:
        29-Jun-26
    Date Last Modified:
        28-Aug-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        generate_objects_automated_py(links_df,consolidated_df)
    Notes:
        20Jul Update included definition_df, specifically because certain values aren't being passed forward to consolidated_df, values where I do not want to manually maintain the 
        list, I want it to be included based on Automated Definition. Based on process this data isn't exposed, so when running this function the only way to access the list is 
        from the direct source. Slightly confusing perhaps could remove consolidated_df, can iterate.
        24Jul. Updated Algo Classification to not print. Added 4 New Lists to Dict.
        28Aug. Changed to meet new structure.
        
    Required Function:
        None
    '''

    # Import Data
    if links_df is None:
        links_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=469651051&single=true&output=csv')
    if knowledge_base_df is None:
        try:
            knowledge_base_df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx')
        except:
            knowledge_base_df = pd.read_excel("https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/knowledge_base.xlsx")


    
    # Generate Files for Dictionary Utilziation

    csv_link_df = links_df[links_df['CSV'].notnull()]
    url_link_df = links_df[links_df['Link'].notnull()]


    py_dict_required_columns = knowledge_base_df[knowledge_base_df['Process']=='Python String Documentation']['Word'].tolist()
    
    ##########
    
    text_ = f"""

'''
module_word: objects_automated
module_definition: Created to serve as a repository for automatically created lists, dictionaries and strings from Google Notes, Dictionaries and other sources as appropriate.  File is created by _____. Whenever run it is automatically overwriden
    
'''
object_dict = {{}}

object_dict['csv_links'] = {{
    'Process':"CSV Links",
    'Categorization':'Reference Dictionary',
    'Word':"CSV Links",
    'Definition':"Dictionary of Links to Google Sheet, Git Hub and other pertinent datasource",
    'publish':0,
    'python_object':{csv_link_df.set_index('COLUMN')[['CSV']].to_dict()['CSV']}
        }}
        
object_dict['url_links'] = {{
    'Process':"URL Links",
    'Categorization':'Reference Dictionary',
    'Word':"URL Links",
    'Definition':"Dictionary of Links to Google Sheet, Git Hub and other pertinent datasource",
    'publish':0,
    'python_object':{url_link_df.set_index('COLUMN')[['Link']].to_dict()['Link']}
        }}

object_dict['python_string_documentation'] = {{
    'Process':"D Organization",
    'Categorization':'Taxonomy',
    'Word':"Python String Documentation",
    'Definition':"Python List of Categories (in order) supporting the prevailing required Python String Documentation Taxonomy",
    'publish':0,
    'python_object':{py_dict_required_columns}
        }}

object_dict['Machine Learning Ontology'] = {{
    'Process':"Machine Learning Ontology",
    'Categorization':'Taxonomy',
    'Word':"ML Model Taxonomy",
    'Definition':"{knowledge_base_df[(knowledge_base_df['Process']=='Machine Learning Ontology')&(knowledge_base_df['Word']=='Taxonomy Definition')].iloc[0]['Definition']}",
    'publish':0,
    'python_object':{knowledge_base_df[(knowledge_base_df['Process']=='Machine Learning Ontology')&(knowledge_base_df['Categorization']=='Taxonomy Node')]['Word'].tolist()}
        }}

object_dict['learning_paradigm'] = {{
    'Process':"Machine Learning Ontology",
    'Categorization':'Taxonomy',
    'Word':"Learning Paradigm",
    'Definition':"{knowledge_base_df[knowledge_base_df['Word']=='Learning Paradigm']["Definition"].iloc[0]}",
    'publish':0,
    'python_object':{knowledge_base_df[knowledge_base_df['Process']=='Learning Paradigm']['Word'].sort_values().tolist()}
        }}

object_dict['learning_objective'] = {{
    'Process':"Machine Learning Ontology",
    'Categorization':'Taxonomy',
    'Word':"Learning Paradigm",
    'Definition':"{knowledge_base_df[knowledge_base_df['Word']=='Learning Objective']["Definition"].iloc[0]}",
    'publish':0,
    'python_object':{knowledge_base_df[knowledge_base_df['Process']=='Learning Objective']['Word'].sort_values().tolist()}
        }}

object_dict['computational_approach'] = {{
    'Process':"Machine Learning Ontology",
    'Categorization':'Taxonomy',
    'Word':"Computational Approach",
    'Definition':"{knowledge_base_df[knowledge_base_df['Word']=='Computational Approach']["Definition"].iloc[0]}",
    'publish':0,
    'python_object':{knowledge_base_df[knowledge_base_df['Process']=='Computational Approach']['Word'].sort_values().tolist()}
        }}

object_dict['analytical_object_type'] = {{
    'Process':"Machine Learning Ontology",
    'Categorization':'Taxonomy',
    'Word':"Analytical Object Type",
    'Definition':"{knowledge_base_df[knowledge_base_df['Word']=='Analytical Object Type']["Definition"].iloc[0]}",
    'publish':0,
    'python_object':{knowledge_base_df[knowledge_base_df['Process']=='Analytical Object Type']['Word'].sort_values().tolist()}
        }}

object_dict['analytical_method'] = {{
    'Process':"Machine Learning Ontology",
    'Categorization':'Taxonomy',
    'Word':"Analytical Method",
    'Definition':"{knowledge_base_df[knowledge_base_df['Word']=='Analytical Method']["Definition"].iloc[0]}",
    'publish':0,
    'python_object':{knowledge_base_df[knowledge_base_df['Process']=='Analytical Method']['Word'].sort_values().tolist()}
        }}
"""
    with open("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/objects_automated.py", "w") as f:
        f.write(text_)

