'''
module_name: eda_functions
module_purpose: Repo for functions required to implement Exploratory Data Analysis in Machine Learning Lifecycle

'''

import numpy as np
import pandas as pd

def ml_dict_to_df(base_dict):

    records = []

    for process, details in base_dict.items():

        process_steps = details.get(
            'Requiured Process Steps', {}
        )

        # If process has required steps
        if process_steps:

            for step, step_details in process_steps.items():

                records.append({
                    'PROCESS': process,
                    'STEP OBJECTIVE': details.get('Step Objective'),
                    'PROJECT REQUIREMENT': details.get('Project Requirement'),
                    'REQUIRED PROCESS STEP': step,
                    'PROCESS STEP OBJECTIVE': step_details.get('Step Objective'),
                    'PROCESS STEP PROJECT REQUIREMENT': step_details.get(
                        'Project Requirements'
                    ),
                    'PROCESS STEP STATUS': step_details.get('Status'),
                    'PROJECT SPECIFIC DELIVERABLE(S)': details.get(
                        'Project Specific Deliverable(s)'
                    ),
                    'OBJECT(S)': details.get('object(s)'),
                    'STATUS': details.get('Status')
                })

        # If no required process steps
        else:

            records.append({
                'PROCESS': process,
                'STEP OBJECTIVE': details.get('Step Objective'),
                'PROJECT REQUIREMENT': details.get('Project Requirement'),
                'REQUIRED PROCESS STEP': None,
                'PROCESS STEP OBJECTIVE': None,
                'PROCESS STEP PROJECT REQUIREMENT': None,
                'PROCESS STEP STATUS': None,
                'PROJECT SPECIFIC DELIVERABLE(S)': details.get(
                    'Project Specific Deliverable(s)'
                ),
                'OBJECT(S)': details.get('object(s)'),
                'STATUS': details.get('Status')
            })

    return pd.DataFrame(records)#.drop(['STEP OBJECTIVE','PROJECT REQUIREMENT'],axis=1)

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
            try:
                process = {}
                for index,row in temp_df[temp_df['Categorization']==word][['Word','Definition']].iterrows():
                    process[row['Word']] = {'Step Objective':row['Definition'],'Project Requirements':'Not Defined','Status':'Not Started'}
            except:
                process = {}
        
            base_dict[word] = {
                'Step Objective':definition,
                'Project Requirement':"Not Defined",
                "Requiured Process Steps":process,
                "Project Specific Deliverable(s)":[],
                'object(s)':[],
                'Status':"Not Started"
            }
        else:
            pass

    base_df = ml_dict_to_df(base_dict)
    
    return base_dict,base_df