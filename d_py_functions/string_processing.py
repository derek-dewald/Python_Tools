import pandas as pd
import numpy as np

import inspect
import textwrap
import ast

def FunctionToSTR(func, *, normalize=True, strip_docstring=True):
    """
    Function which Converts the Documentation of a Python Function into a List, breaking on White Space. For purposes of development of 
    CompareFunction, which looks for differences.

    Parameters:
        func(function): Python Function
        normalize (bool): Optional Parameters to Standardize the Formating of the function to represent as python processes, opposed to how written. Example, removes impact of Blank Spaces.
        strip_docstring (bool): Optional Parameter to Not include the Doc String using ast to read and parse function.

    Returns:
        list

    date_created:17-Dec-25
    date_last_modified: 17-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        from data_d_strings import template_doc_string_print
        FunctionToSTR(template_doc_string_print)
    
    """
    try:
        src = inspect.getsource(func)
        src = textwrap.dedent(src)
        
    except (OSError, TypeError):  # no source (e.g., builtins, C-extensions, REPL edge cases)
        name = getattr(func, "__name__", repr(func))
        return f"<no source available for {name!r}>"

    if strip_docstring or normalize:
        
        try:
            mod = ast.parse(src)
            fn = next(n for n in mod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
            if strip_docstring and fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) and isinstance(fn.body[0].value.value, str):
                fn.body = fn.body[1:]  # drop leading docstring
            if normalize:
                # Return just the function definition re-emitted from the AST
                return ast.unparse(fn)
        
        except Exception:
            # If AST handling fails, just return the dedented source we already have
            pass

    return src.split()


def compare_function(func1,func2,additional_records=20,strip_docstring=True):
    
    '''
    Function which compares the Text of 2 python function objections, for the purposes of validiting equivalency or easily identify changes.
    Used to reduce impact of manual validation and creation of function in multiple environments without a consistent tool such as Git. 
    Once adopting better practices, this will no longer be necessary.

    Function Utilizes the Input Function FunctionToSTR() as a Base. Currently it Does

    Parameters:
        func1(function): Python Function
        func2(function): Python Function
        additional_records(int): Number of String Characters for each function which will be displayed after the point of identifying difference.
        strip_docstring(bool): Boolean to determine whether you wish to compare the Doc String aswell as the function.
    Returns:
        Object Type

    date_created:17-Dec-25
    date_last_modified: 17-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        Example Function Call

    
    Function which Compares 2 Functions and determines if they are different. Specifically, it can help to easily
    Manage Version control of Functions outside of a More robust environment such as GIT.
    

    '''
    
    list1 = FunctionToSTR(func1,strip_docstring=strip_docstring)
    list2 = FunctionToSTR(func2,strip_docstring=strip_docstring)
    
    length = max(len(list1),len(list2))
    
    for record in range(0,length):
        # Functions should not have the same name. Pass first
        if record==1:
            pass
        elif list1[record]!=list2[record]:
            print(f'Mismatch Identified at Record Number: {record}')
            print(list1[record:record+additional_records])
            print(list2[record:record+additional_records])
            
            break

    print('Reconcilation Complete')


def word_counts_from_column(df,
                            column_name,
                            lower= True,
                            pattern= r"[A-Za-z']+",
                            min_len= 1,
                            dropna=True):
    """
    Vectorized word counting over a DataFrame column.
    
    Parameters:
        df (df): DataFrame
        column_name(str): Column containing text to iterate through.
        lower(bool): Apply Lowercasing for consistency before counting.
        pattern(re): r"[A-Za-z']+". Regex for tokens (words). Adjust for unicode if needed.
        min_len(int): Minimum token length to keep.
        dropna(bool): Whether to drop empty tokens.
        
    Returns:
        pd.Series. Word counts indexed by token (sorted desc).

    date_created:01-Feb-26
    date_last_modified: 01-Feb-26
    classification:TBD
    sub_classification:TBD
    usage:
        Example Function Call

    """
    s = df[column_name]

    # Convert everything to string safely; keep NaN out of the way
    # Using astype(str) turns NaN into 'nan', so instead fillna('') first.
    s = s.fillna("").astype(str)

    if lower:
        s = s.str.lower()

    # Find all tokens per row (vectorized)
    tokens = s.str.findall(pattern).explode()

    if dropna:
        tokens = tokens.dropna()

    if min_len > 1:
        tokens = tokens[tokens.str.len() >= min_len]

    return tokens.value_counts()