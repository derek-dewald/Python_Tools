'''
module_word: eda_functions
module_definition: Repo for functions required to implement Exploratory Data Analysis in Machine Learning Lifecycle

'''

import numpy as np
import pandas as pd

def create_ml_dictionary_template(
    source='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx'
):
    '''
    Definition:
        Function used to create a Data Dictionary for a New Data set using a residual manual definition arepository. 
    Parameters:
        source(str): location of manual Excel File. Can be updated to take multiple input types is necessary
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
        NoneGood
    
    '''
    
    df = pd.read_excel(source)
    #return {x:"" for x in df[(df['Process'].str.contains('Machine Learning Lifecycle'))&(df['Categorization']=='Process Step')]['Word']}

    temp_df = df[
        (df['Process'].str.contains('Machine Learning Lifecycle'))&
        (df['Source']!='LVL2')&
        (df['Word']!='Definition')
    ]
    base_dict = {}
    
    for index,row in temp_df.iterrows():
        if row['Source']=='Knowledge Base':
            word = row['Word']
            definition = row['Definition'] 
            base_dict[word] = {'Process Guidance':definition}
        else:
            # Word Defined above. Use it for reference
            base_dict[word][row['Word']] = row['Definition']

    return base_dict