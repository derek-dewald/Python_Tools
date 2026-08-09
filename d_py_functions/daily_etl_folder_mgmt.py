'''
module_name: daily_etl_folder_mgmt
module_purpose: Function Repository for all manually created functions with the Automation and Administration associated with managing my Diarization System. 

default_structure: Python Function, Documented Consistent with Prevailing Guidance.
module_guidance: Functions which are used in here are to be Stand Alone, sole purpose for management of this daily/monthly/ad hoc periodic files. Can utilize generic functions outside.

'''

import pandas as pd
import numpy as np

from objects_automated import object_dict

def extract_consolidated_raw_dataset(df_dict,export_location=False):
    
    '''
    Definition:
        Create a Consolidated Dataset of files in Dictionary, which are meant to be of the structure Process, Categorization, Word and Definition. Data set for the purposes, of aggregating totals and _____________. Used as a input for generate_objects_automated_py.
    Parameters:
        df_dict(dict): Dictionary of files to be included. 
        export_location(str): Location to where CSV file is to be exported. If left Blank, will not export a CSV.
    Returns:
        Excel File
    Date Created:
        06-Jul-26
    Date Last Modified:
        06-Jul-26
    Process:
        ETL
    Categorization:
        Excel File Creation
    Usage:
        df_dict = {
        'Notes':notes_df,
        'Definitions':definition_df,
        'Knowledge Base':knowledge_base_df,
        'Manual Objects':manual_object_df
        }
        extract_consolidated_raw_dataset(df_dict)
    Notes:
        Definition

    '''
    df = pd.DataFrame()
    
    for df_name in df_dict.keys():
        try:
            temp = df_dict[df_name][['Process','Categorization','Word','Definition']]
            temp['Location'] = df_name
            df = pd.concat([df,temp])
        except:
            print(f'Could not compute, {df_name}')

    if export_location:
        df.to_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/consolidated_dataset.xlsx',index=False)

    return df

def generate_files_for_streamlit(
    definition_df=pd.DataFrame(),
    notes_df=pd.DataFrame(),
    generate_excel_files=True,
    ignore_words_from_lvl1_lvl2 = []
):

    '''
    Definition:
        Process Utilized to Combine Notes/ Definitions and Logic into Knowledge Base, which is utilized to Create, Processes, Parameters.

        Taxonomy - Process >> Process Step >> Word/Action/Guidance/Etc.

        Process: 
        Process Step:

        Important. To Retain Order in Consolidated Data, or to be included in Process it must be defined as a process.
        Machine Learning - Process. Which is Comprised of Processes. Goal Setting, Data Preparation etc..

    Parameters:
        notes_df (dataframe): Dataframe containing Notes from Google. Default is none and it will pull directly from Google.
        definition_df (dataframe): Dataframe containing Definitions from Google. Default is none and it will pull directly from Google.
        
    Returns:
        Excel File
    Date Created:
        02-Jul-26
    Date Last Modified:
        29-Jul-26
    Process:
        Definition
    Categorization:
        Definition
    Usage:
        knowledge_base_df,process_df,consolidated_data = generate_files_for_streamlit(definition_df,notes_df)
        knowledge_base_df,process_df,consolidated_data = generate_files_for_streamlit()
    Notes:
        22-Jul - Overhauled merge. Attempted to streamline, simplify and reduce duplication. Increase Visability.
        28-Jul - Originally Named - generate_knowledge_base, stream lined to remove usage of Dictionary, which were complex and operationally inefficient.
        29-Jul - Changed Foundational Structure. Simplified with Clarity on Input.

        Version 3. With the Idea that We have 3 Layers. 
        Combined Definition/ Notes.
        Level 1 - Process is also a Word. Merge this in. Machine Learning Lifecycle.
        Level 2 - Categorization is also a Word.
        
    '''
    
    if len(definition_df)==0:
        definition_df = pd.read_csv(object_dict['csv_links']['python_object']['google_definition_csv'])

    if len(notes_df)==0:
        notes_df = pd.read_csv(object_dict['csv_links']['python_object']['google_notes_csv']).fillna('')

    if len(ignore_words_from_lvl1_lvl2)==0:
        ignore_words_from_lvl1_lvl2 = ['Process','Requirement','Definition','Guidance']

    # Before Merging Files. Update Notes to include Definitions from any item which is a Process from Definition.

    definition_df = definition_df[(definition_df['Process'].notnull())&(definition_df['Process']!="")][['Process','Categorization','Word',"Definition"]].fillna('').copy()
    notes_df = notes_df[(notes_df['Process'].notnull())&(notes_df['Process']!="")][['Process','Categorization','Word',"Definition"]].fillna('').copy()

    notes_df1 = notes_df.merge(definition_df[definition_df['Word']=='Definition'][['Process','Definition']].rename(columns={'Process':"Word",'Definition':'Definition_'}),on='Word',how='left').fillna("")
    notes_df1['Definition'] = np.where((notes_df1['Categorization']=='Process Step')&(notes_df1['Definition']==""),notes_df1['Definition_'],notes_df1['Definition'])
    notes_df1.drop('Definition_',axis=1,inplace=True)

    # CREATE KNOWLEDGE File, which is simply everything combined Together
    notes_filter = notes_df1[['Word','Process']].drop_duplicates().reset_index().rename(columns={'index':'notes_order'})
    def_filter = definition_df[['Word','Process']].drop_duplicates().reset_index().rename(columns={'index':'def_order'})
    
    knowledge_base_df = pd.concat([definition_df,notes_df1]).reset_index(drop=True)

    # generate a list of Records where the Value for Word is also a Process, meaning these are Sub Processes 
    process_also_word_list = knowledge_base_df['Word'].unique().tolist()
    # Filter list to get Dataframe
    lvl1_df = knowledge_base_df[(knowledge_base_df['Process'].isin(process_also_word_list))&(knowledge_base_df['Categorization']!='Process')]

    # Merge Filtered list back into dataset to get a list of Items to add as LVL1 Processes
    lv1_insert_df = knowledge_base_df.drop('Definition',axis=1).merge(lvl1_df,left_on='Word',right_on='Process',how='inner',suffixes=('',"_"))
    lv1_insert_df['Source'] = 'LVL1'

    knowledge_base_df['Source'] = "Knowledge Base"
    knowledge_base_df = pd.concat([knowledge_base_df,lv1_insert_df])

    # Merge in Filter List
    knowledge_base_df = knowledge_base_df.merge(notes_filter,on=['Word','Process'],how='left').merge(def_filter,on=['Word','Process'],how='left')
    knowledge_base_df.sort_values(['Process','def_order','notes_order'],inplace=True)

    # Standardize After Filtering.
    knowledge_base_df['Categorization'] = np.where(knowledge_base_df['Process_'].notnull(),knowledge_base_df['Process_'],knowledge_base_df['Categorization'])
    knowledge_base_df['Word'] = np.where(knowledge_base_df['Word_'].notnull(),knowledge_base_df['Word_'],knowledge_base_df['Word'])
    knowledge_base_df.drop(['Process_','Categorization_','Word_','notes_order','def_order'],inplace=True,axis=1)


    order_df = knowledge_base_df[['Process','Categorization','Word']].drop_duplicates().reset_index(drop=True).reset_index(names='order')
    
    # Need to merge in Second Level. Identify items which are Words
    cat_is_word_list = knowledge_base_df['Word'].unique().tolist()

    # Do not want to take Standarized Word
    cat_is_word_list = [x for x in cat_is_word_list if x not in ignore_words_from_lvl1_lvl2]

    # Generate List of Items to Merge Back in
    lvl2_df = knowledge_base_df[(knowledge_base_df['Categorization'].isin(cat_is_word_list))]

    # Merge ONLY where Processs and Categorization from Master Are Equal to Process and Categorization from LVL2 to prevent unitended duplicationk
    lvl2 = knowledge_base_df.drop('Source',axis=1).merge(lvl2_df.drop('Source',axis=1),left_on=['Categorization','Word'],right_on=['Process','Categorization'],how='inner',suffixes=("","_"))
    lvl2['Source'] = "LVL2"

    knowledge_base_df = pd.concat([knowledge_base_df,lvl2])
    knowledge_base_df = knowledge_base_df.merge(order_df,on=['Process','Categorization','Word'],how='left')
    
    # Clean Up residual definitions
    knowledge_base_df['Categorization'] = np.where(knowledge_base_df['Categorization_'].notnull(),knowledge_base_df['Categorization_'],knowledge_base_df['Categorization'])
    knowledge_base_df['Word'] = np.where(knowledge_base_df['Word_'].notnull(),knowledge_base_df['Word_'],knowledge_base_df['Word'])
    knowledge_base_df['Definition'] = np.where(knowledge_base_df['Definition_'].notnull(),knowledge_base_df['Definition_'],knowledge_base_df['Definition'])
    knowledge_base_df.drop(['Process_','Categorization_','Word_','Definition_'],inplace=True,axis=1)

    source_map = {'Knowledge Base':0,'LVL1':1,'LVL2':2}
    knowledge_base_df['source_map'] = knowledge_base_df['Source'].map(source_map)


    knowledge_base_df = knowledge_base_df.sort_values(['order','source_map']).drop(['order','source_map'],axis=1)
    if generate_excel_files:
        knowledge_base_df.to_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx',index=False)

    return knowledge_base_df