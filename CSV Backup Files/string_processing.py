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