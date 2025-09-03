import pandas as pd
import numpy as np
import inspect 
import re
from collections import Counter

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



def CountWordsinDFColumn(df,
                         column_name,
                         output_type='both'):
    '''
    Function which reads a Dataframe Column and returns a value count of either, Phrases, Words or Both.
    Used for understanding Text frequency within a Column, utilizes Counter, from Collections Library.
    
    Parameters:
        df (dataframe)
        column_name (str): Name of Column to Iterate through
        output_type (str): phrase, word, both.
            phrase: Returns a count of the entire phrase
            word: Returns a count of each individual word contained within the text.
            both: Returns
    
    Returns:
        df
    

    
    '''
    
    col = df[(df[column_name].notnull())][column_name].astype(str).values.flatten()
    total_col = Counter(col)
    
    if (output_type.lower()=='phrase')|(output_type.lower()=='both'):
        col_count_df = pd.DataFrame(total_col.items(),columns=['Text','Count']).sort_values('Count',ascending=False)
        col_count_df['Type'] = 'Phrase'
    else:
        col_count_df = pd.DataFrame()
        
    if (output_type.lower()=='word')|(output_type.lower()=='both'):
        words = []
        for phrase in total_col:
            words.extend(phrase.split())
        
        total_word = Counter(words)
        word_count_df = pd.DataFrame(total_word.items(),columns=['Text','Count']).sort_values('Count',ascending=False)
        word_count_df['Type'] = 'Word'
    else:
        word_count_df = pd.DataFrame()
    
    if output_type.lower() =='phrase':
        return col_count_df
    
    elif output_type.lower() == 'word':
        return word_count_df
    
    else:
        return pd.concat([col_count_df,word_count_df])
    

def RemovePatternFromDFColumn(df,
                               column,
                               pattern,
                               new_column_name=None,
                               ignore_case=True,
                               clean_whitespace=True):
    """
    Removes text matching a regex pattern from a DataFrame column.
    Optionally logs removed text in a new or existing column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to clean.
        pattern (str): Regex pattern to remove.
        new_column_name (str): Optional column name to store removed values.
        ignore_case (bool): If True, pattern match is case-insensitive.
        clean_whitespace (bool): If True, strips and collapses whitespace.
    
    Returns:
        pd.DataFrame: Modified DataFrame with cleaned and optionally logged matches.
    """
    df = df.copy()
    flags = re.IGNORECASE if ignore_case else 0

    # Safely fill NaNs
    original_text = df[column].fillna('')

    # Find all matches (full pattern matches, not just inner substrings)
    found_matches = original_text.str.findall(pattern, flags=flags)
    match_strings = found_matches.apply(lambda x: ', '.join(x))

    # Add or update the removal log
    if new_column_name:
        if new_column_name in df.columns:
            df[new_column_name] = df[new_column_name].fillna('')
            df[new_column_name] = df.apply(
                lambda row: ', '.join(filter(None, [row[new_column_name], match_strings[row.name]])).strip(', '),
                axis=1
            )
        else:
            df[new_column_name] = match_strings

    # Remove the matched patterns from the text
    cleaned_text = original_text.str.replace(pattern, '', flags=flags, regex=True)
    
    if clean_whitespace:
        cleaned_text = (
            cleaned_text.str.replace(r'\s+', ' ', regex=True).str.strip()
        )
    
    df[column] = cleaned_text

    return df

def RemoveWordfromDFColumn(df,
                           column,
                           word,
                           new_column_name=None):
    
    '''
    
    '''
    
    df = df.copy()
    
    if new_column_name:
        try:
            df[new_column_name] = np.where(df[column].fillna("").str.contains(word,case=False),word,df[new_column_name])
        except:
            df[new_column_name] = np.where(df[column].fillna("").str.contains(word,case=False),word,"")
        
    df[column] = df[column].fillna("").str.replace(word,"",regex=True,case=False).str.strip()
        
    return df


def GenericStrClean(df,
                    column_name,
                    trim=True,
                    lower_case=True,
                    remove_punc=False,
                    remove_letters=False,
                    remove_numbers=False,
                    new_column_name=None):
    '''


    '''
    
    df= df.copy()
    
    if not new_column_name:
        new_column_name = f"{column_name}_CLEAN"
    
    result = df[column_name].astype(str).copy()
    
    
    if lower_case:
        result = result.str.lower()
    
    if remove_letters:
        result = result.str.replace(r'[a-zA-Z]',"",regex=True)
    
    if remove_punc:
        result = result.str.replace(r'[^\w\s]','',regex=True)
        
    if remove_numbers:
        result = result.str.replace(r'\d','',regex=True)
        
    if trim:
        result = result.str.strip()
                
    df[new_column_name] = result
    
    return df

def TextClean(
    df,
    column_list,
    lower_case=False,
    remove_newlines=False,
    strip_whitespace=False,
    normalize_whitespace=False,
    remove_punctuation=False,
    only_digits=False,
    only_letters=False):
    
    
    """
    Applies selected text cleaning operations to specified DataFrame columns.

    Parameters:
        df (pd.DataFrame): DataFrame to clean.
        column_list (list): Columns to clean.
        
        Cleaning Options (all default to False):
            remove_newlines (bool): Remove \r and \n characters.
            strip_whitespace (bool): Trim leading and trailing whitespace.
            normalize_whitespace (bool): Collapse multiple spaces/tabs into one space.
            remove_punctuation (bool): Remove punctuation characters.
            only_digits (bool): Remove all non-digit characters and convert to numeric.
            only_letters (bool): Remove all non-letter characters and convert to string.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    for col in column_list:
        if col not in df.columns:
            continue

        series = df[col].astype(str)
        
        if lower_case:
            series = series.str.lower()
        if remove_newlines:
            series = series.str.replace(r'[\r\n]+', '', regex=True)
        if normalize_whitespace:
            series = series.str.replace(r'\s+', ' ', regex=True)
        if strip_whitespace:
            series = series.str.strip()
        if only_digits:
            series = series.str.replace(r'[^\d.]', '', regex=True)
            series = pd.to_numeric(series, errors='coerce')
        if only_letters:
            series = series.str.replace(r'[^a-zA-Z]', '', regex=True)
        if remove_punctuation:
            series = series.str.translate(str.maketrans('', '', string.punctuation))

        df[col] = series

    return df