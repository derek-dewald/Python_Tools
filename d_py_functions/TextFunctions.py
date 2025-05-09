import pandas as pd
import numpy as np
import inspect 
import re

def TextClean(
    df,
    column_list,
    remove_newlines=False,
    strip_whitespace=False,
    normalize_whitespace=False,
    only_digits=False):
    """
    Applies selected text cleaning operations to specified DataFrame columns.

    Parameters:
        df (pd.DataFrame): DataFrame to clean.
        column_list (list): Columns to clean.
        
        Cleaning Options (all default to False):
            remove_newlines (bool): Remove \r and \n characters.
            strip_whitespace (bool): Trim leading and trailing whitespace.
            normalize_whitespace (bool): Collapse multiple spaces/tabs into one space.
            only_digits (bool): Remove all non-digit characters and convert to numeric.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    for col in column_list:
        if col not in df.columns:
            continue

        series = df[col].astype(str)

        if remove_newlines:
            series = series.str.replace(r'[\r\n]+', '', regex=True)
        if normalize_whitespace:
            series = series.str.replace(r'\s+', ' ', regex=True)
        if strip_whitespace:
            series = series.str.strip()
        if only_digits:
            series = series.str.replace(r'[^\d.]', '', regex=True)
            series = pd.to_numeric(series, errors='coerce')

        df[col] = series

    return df


def IterateThroughListuntilText(list_,text):
    """
    Function which Iterates through a list of python objects, not stopping until it finds an object which shares the same text as it.
    Created to iterate through sklearn model documentation . 

    Parameters:
        lines (list of str): Input lines

    Returns:
        str: Concatenated result up to (but not including) line with '..'
    """
    result = []
    for line in list_:
        if text in line:
            break
        result.append(line.strip())
    return ' '.join(result).strip()


def extract_doc_sections_all(estimator_class, sections=('Parameters', 'Attributes'),model_name=None):
    """
    Function which iterates through text, looking for breakpoints and specific indentation. 
    Created for the purposes of extracting Parameters and Attributes from SKlearn Library.

    Parameters:
        estimator_class: A scikit-learn class (e.g., LogisticRegression)
        sections: A tuple of section names to extract

    Returns:
        pd.DataFrame with columns: ['Section', 'Name', 'Type', 'Default', 'Description']
    """
    doc = inspect.getdoc(estimator_class)
    lines = doc.split('\n')

    results = []
    section = None
    current_name = None
    current_type = ''
    current_default = ''
    current_description = []

    for line in lines:
        stripped = line.strip()

        # Section start
        if stripped in sections:
            section = stripped
            continue
        elif stripped in ['Parameters', 'Attributes', 'Returns', 'Examples', 'See Also', 'Notes', 'References']:
            if section in sections:
                section = None  # exit current section
            continue

        if section is None:
            continue

        # Param/attribute line
        if re.match(r'^\S[^:]*\s*:\s*.+$', line):
            if current_name:
                results.append({
                    'Section': section,
                    'Name': current_name,
                    'Type': current_type,
                    'Default': current_default,
                    'Description': ' '.join(current_description).strip()
                })

            # Start new field
            current_description = []
            colon_index = line.find(':')
            current_name = line[:colon_index].strip()
            rest = line[colon_index+1:].strip()

            current_type = ''
            current_default = ''
            if ',' in rest:
                parts = [p.strip() for p in rest.split(',')]
                current_type = parts[0]
                for p in parts[1:]:
                    if p.startswith('default='):
                        current_default = p.split('=')[1].strip()
            else:
                current_type = rest

        elif line.startswith('    ') and not line.strip().startswith('..'):
            current_description.append(line.strip())

    # Final item
    if current_name:
        results.append({
            'Section': section,
            'Name': current_name,
            'Type': current_type,
            'Default': current_default,
            'Description': ' '.join(current_description).strip()
        })

    df = pd.DataFrame(results)
    if not model_name:
        model_name = lines[0]
    df['Model'] =model_name
    
    df['Estimator'] = estimator_class

    sklearn_desc = pd.DataFrame([[model_name,IterateThroughListuntilText(lines,'..')]],columns=['Model Name','Sklearn Desc'])
    
    return df,sklearn_desc