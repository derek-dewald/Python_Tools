from SharedFolder import ReadDirectory
from Connections import ParamterMapping
#from IPython.display import display, HTML
import pandas as pd
import ast



def extract_function_details_ast(file_content,file_name):
    '''
    Extracts structured function details from a Python script using AST.
    
    Parameters:
    file_content (str): The raw string content of a Python script.

    Returns:
    dict: A dictionary with function names as keys and metadata (description, args, return, code).

    '''
    
    tree = ast.parse(file_content)  # Parse the script into an AST
    function_data = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):  # Identify function definitions
            function_name = node.name
            docstring = ast.get_docstring(node) or "No description available"

            # Extract arguments from function signature
            args = [arg.arg for arg in node.args.args]

            # Extract return type annotation if present
            return_type = ast.unparse(node.returns) if node.returns else "None"

            # Extract function code using AST
            function_code = ast.get_source_segment(file_content, node).strip()

            # Process the docstring to separate description, args, and return
            description_text = []
            args_text = []
            return_text = "None"

            if docstring:
                doc_lines = docstring.split("\n")
                found_args = False

                for line in doc_lines:
                    stripped = line.strip()

                    if stripped.lower().startswith(("args:", "parameters:")):  # Start of args
                        found_args = True
                        continue
                    elif stripped.lower().startswith("returns:"):  # Start of return
                        return_text = stripped.replace("Returns:", "").strip()
                        found_args = False
                        continue

                    if not found_args:
                        description_text.append(stripped)
                    else:
                        args_text.append(stripped)
                        
            function_data[function_name] = {
                "Description": "\n".join(description_text).strip(),
                "Arguments": args_text if args_text else args,
                "Return": return_text,
                "Code":function_code,
                'File':file_name
            }
            
    return function_data


def ReadPythonFiles(location='/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/'):
    
    '''
    Function which reads a Folder, iterating through all files saved, looking for .py files utilizing athe Extract Function Details AST 
 
 
     Args:
         location (str): Folder
         
    Returns:
        Dataframe of Functions with description.

    '''    
    py_file_dict = {}
    
    for file_name in [x for x in ReadDirectory(location) if x.find('.py')!=-1 and x.find('__')==-1]:
        with open(f"{location}{file_name}", "r", encoding="utf-8") as file:
            data = file.read()
            py_file_dict.update(extract_function_details_ast(data,file_name))
            
    return pd.DataFrame(py_file_dict).T

def CreateMarkdown(df, return_value=""):
    '''
    Create a Markdown string from a DataFrame with columns: Title, Header, Description
    '''
    try:
        df1 = df[['Title', 'Header', 'Description']]
    except:
        print('DataFrame must contain: Title, Header, Description')
        return ''

    title = ""
    step_number = 1
    text = ""

    for _, row in df1.iterrows():
        # New section
        if title != row['Title']:
            if title != "":
                text += "\n"  # separate sections
            title = row['Title']
            text += f"### {step_number}. {title}\n"
            step_number += 1

        # Level 2 bullet
        if isinstance(row['Header'], str) and row['Header'].strip():
            text += f"- {row['Header'].strip()}\n"

            # Level 3 bullet (indented)
            if isinstance(row['Description'], str) and row['Description'].strip():
                text += f"    - {row['Description'].strip()}\n"

    return text



def CreateMarkdownfromProcess(process_name=None,return_value=""):
    '''
    Function to call Process Map from Google sheet with a single reference to the name of the process.
    
    Args:
        process_name (str): Process Name as defined in Google Sheet
        return_value (str): Value Desired to be returned, as required input from CreateMarkdown
    
    Returns:
        If return_value == "", then Displayed Markdown
        If return_value == "text", HTML formatted markdown text
        otherwise, returns ""
    '''
    # Call Parameter Mapping function to return DF of Process Sheet

    df = pd.read_csv(ParamterMapping('ProcessSheet')['CSV'].item())
    
    if process_name !=None:
        df1 = df[df['Process']==process_name].drop('Process',axis=1)
    else:
        return df
        
    try:
        if return_value =="":
            return CreateMarkdown(df1)
        elif return_value == "text":
            return CreateMarkdown(df1,'text')
    
    except:
        print("Could Not Format Data")
        return df
