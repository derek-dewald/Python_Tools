# File Description: Archived Functions is a List of Historically Created Files which are no longer in active use, but stored for the purposes of context. No Active Files should be included, it should bot be linked or referenced. MOVE FILES from this location if referenced.



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

def CreateCalculatedField(df, primary_key, calc_instructions, include_all=1):

    '''
    calc_instructions = [
    {'type': 'sum', 'value1': 'LENDING', 'name': 'TOTAL_LENDING'},
    {'type': 'weighted_average', 'value1': 'LENDING', 'value2': 'INTEREST_RATE', 'name': 'WEIGHTED_INTEREST'},
    {'type': 'ratio', 'value1': 'RENEWED_AMOUNT', 'value2': 'MATURED_AMOUNT', 'name': 'RENEWAL_RATE'}
    ]

    output = CreateCalculatedField(final_df, ['BRANCHNAME', 'CITY', 'LOB', 'DURATION'], calc_instructions)

    

    Create as initially Used in Data Management Dashboard Accumulation. Was not pursed Overtly Complex.

    '''

    base_aggs = {}
    
    # Collect all fields we need
    for calc in calc_instructions:
        if calc['type'] == 'sum':
            base_aggs[calc['name']] = (calc['value1'], 'sum')
        elif calc['type'] == 'weighted_average':
            base_aggs[f"__{calc['name']}_NUM"] = (
                calc['value2'], lambda x, col=calc['value1']: (df.loc[x.index, col] * x).sum()
            )
            base_aggs[f"__{calc['name']}_DEN"] = (calc['value1'], 'sum')
        elif calc['type'] == 'ratio':
            base_aggs[f"__{calc['name']}_NUM"] = (calc['value1'], 'sum')
            base_aggs[f"__{calc['name']}_DEN"] = (calc['value2'], 'sum')

    # Base groupby
    grouped = df.groupby(primary_key, dropna=False).agg(**base_aggs).reset_index()

    # Compute post-aggregates
    for calc in calc_instructions:
        if calc['type'] == 'sum':
            continue
        elif calc['type'] == 'weighted_average':
            num = grouped[f"__{calc['name']}_NUM"]
            den = grouped[f"__{calc['name']}_DEN"]
            grouped[calc['name']] = np.where(den != 0, num / den, np.nan)
            grouped.drop(columns=[f"__{calc['name']}_NUM", f"__{calc['name']}_DEN"], inplace=True)
        elif calc['type'] == 'ratio':
            num = grouped[f"__{calc['name']}_NUM"]
            den = grouped[f"__{calc['name']}_DEN"]
            grouped[calc['name']] = np.where(den != 0, num / den, np.nan)
            grouped.drop(columns=[f"__{calc['name']}_NUM", f"__{calc['name']}_DEN"], inplace=True)

    result_frames = [grouped.copy()]

    # Rollup combinations
    if include_all:
        for r in range(1, len(primary_key)):
            for group_cols in combinations(primary_key, r):
                temp = df.copy()
                for col in primary_key:
                    if col not in group_cols:
                        temp[col] = 'All'
                temp_group = CreateCalculatedField(temp, primary_key, calc_instructions, include_all=0)
                result_frames.append(temp_group)

        # Full 'All' row
        temp = df.copy()
        for col in primary_key:
            temp[col] = 'All'
        temp_group = CreateCalculatedField(temp, primary_key, calc_instructions, include_all=0)
        result_frames.append(temp_group)

    return pd.concat(result_frames, ignore_index=True)



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
