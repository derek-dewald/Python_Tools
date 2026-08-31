'''
module_name: eda_functions
module_purpose: Repo for functions required to implement Exploratory Data Analysis in Machine Learning Lifecycle

'''

import numpy as np
import pandas as pd

def create_ml_dictionary_template():
    df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx')
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