import ast
import pandas as pd
from SharedFolder import ReadDirectory

def CreateMarkdown(df,return_value=""):
    
    '''
    Function to Create a Markdown file from Process DF, which is a data frame of the structure, 
    Title,Header,Description
    
    Args:
        Dataframe( Must be of format, Title, Header, Description)
        return_value (str: "" or text):
            If Blank, will render text in HTML Format. 
            If text, then will return text for rendering in HTML Markdown
    
    Returns:
        Conditional on Return Value. Please read Args.
    
    
    '''
    
    try:
        df1 = df[['Title','Header','Description']]
    
    except:
        
        print('DataFrame does not meet structure requirement, which must include 3 Column: Title, Header, Description')
        return ''
    
    title= ""
    step_number = 1
    text = ""

    l2_bullet = '-'  # Level 2 Bullet
    l3_bullet = '*'  # Level 3 Bullet

    for index, row in df1.iterrows():
        # Ensure previous list is closed before starting a new title
        if title and title != row.iloc[0]:  
            text += "</ul>\n"  # Close the last unordered list before switching to a new title

        # If it's a new title, start a new section
        if title == "" or title != row.iloc[0]:
            text += f"<h4>{step_number}. {row.iloc[0]}</h4>\n<ul>\n"  # Reset indentation
            step_number += 1
            title = row.iloc[0]  # Store the new title

        # Add Level 2 content (Column 2)
        if isinstance(row.iloc[1], str) and row.iloc[1].strip():
            text += f"  <li>{row.iloc[1]}</li>\n"  # L2 starts here

            # Add Level 3 content (Column 3) only if it exists
            if isinstance(row.iloc[2], str) and row.iloc[2].strip():
                text += f"    <ul><li>{row.iloc[2]}</li></ul>\n"  # L3 indented under L2

    text += "</ul>\n"  # Close any remaining lists

    if return_value =="":# Display the formatted HTML output in Jupyter Notebook
        display(HTML(text))
        
    else:
        return text


def CreateMarkdownfromProcess(process_name,return_value=""):
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

    try:
        df = ParamterMapping('ProcessSheet')
        df1 = df[df['Process']==process_name].drop('Process',axis=1)

    except:
        print('Could Not Retrieve Data')
        return ""
        

    try:
        if return_value =="":
            return CreateMarkdown(df1)
        elif return_value == "text":
            return CreateMarkdown(df1,'text')
    
    except:
        print("Could Not Format Data")
        return ""


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

