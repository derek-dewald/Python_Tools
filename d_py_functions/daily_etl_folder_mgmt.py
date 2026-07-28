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
    generate_excel_files=True
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
        22-Jul-26
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
    '''
    
    if len(definition_df)==0:
        definition_df = pd.read_csv(object_dict['csv_links']['python_object']['google_definition_csv'])

    if len(notes_df)==0:
        notes_df = pd.read_csv(object_dict['csv_links']['python_object']['google_notes_csv']).fillna('')

    # Before Merging Files. Update Notes to include Definitions from any item which is a Process from Definition.

    definition_df = definition_df[['Process','Categorization','Word',"Definition"]].fillna('').copy()
    notes_df = notes_df[['Process','Categorization','Word',"Definition"]].fillna('').copy()

    notes_df1 = notes_df.merge(definition_df[definition_df['Word']=='Definition'][['Process','Definition']].rename(columns={'Process':"Word",'Definition':'Definition_'}),on='Word',how='left').fillna("")
    notes_df1['Definition'] = np.where((notes_df1['Categorization']=='Process Step')&(notes_df1['Definition']==""),notes_df1['Definition_'],notes_df1['Definition'])
    notes_df1.drop('Definition_',axis=1,inplace=True)

    # CREATE KNOWLEDGE File, which is simply everything combined Together
    knowledge_base_df = pd.concat([definition_df,notes_df1]).reset_index(drop=True)
    
    # Create 2 Distinct Files Processes. Not Processes
    processes = knowledge_base_df[knowledge_base_df['Categorization']=='Process'].reset_index(drop=True).reset_index().rename(columns={'index':'PROC_ORDER'})
    not_processes = knowledge_base_df[knowledge_base_df['Categorization']!='Process'].reset_index(drop=True).reset_index().rename(columns={'index':'NP_ORDER'})
    #return knowledge_base_df,processes,not_processes
    
    # Create a Supplemental File which are effectively Items which need to be merged into Processes to move from the Straw Man Process to a more fullsome 
    supplemental = not_processes.merge(knowledge_base_df[['Word','Process']].drop_duplicates().rename(columns={'Process':'Process_','Word':"Process"}),on='Process',how='inner')
    supplemental['Word_'] = supplemental['Word'].copy()
    supplemental['Word'] = supplemental['Process'].copy()
    supplemental['Process'] = supplemental['Process_'].copy()
    supplemental.drop('Process_',axis=1,inplace=True)

    # Before Supplemental can be finalized, needs to merge into Final DF. to ORder BEFORE replacing Word with Word_
    consolidated_data = pd.concat([knowledge_base_df,supplemental])
    
    consolidated_data = consolidated_data.merge(processes[['PROC_ORDER',"Process"]].rename(columns={'Process':"Word"}),on='Word',how='left')
    # For items that dont have a order does this matter?
    consolidated_data['CAT_ORDER'] = np.where(consolidated_data['Categorization']=='Process',0,1)
    
    # Fill as 0 so that Proceses will take higher priorirty, while retain NP Order
    consolidated_data['NP_ORDER'] = consolidated_data['NP_ORDER'].fillna(0)

    consolidated_data = consolidated_data.sort_values(['Process','CAT_ORDER','PROC_ORDER','NP_ORDER'])
    consolidated_data['Categorization'] = np.where(consolidated_data['Word_'].notnull(),consolidated_data['Word'],consolidated_data['Categorization'])
    consolidated_data['Word'] = np.where(consolidated_data['Word_'].notnull(),consolidated_data['Word_'],consolidated_data['Word'])
    
    consolidated_data.drop(['Word_','PROC_ORDER','CAT_ORDER','NP_ORDER'],inplace=True,axis=1)
    
    # Merge in Second ORder items. Examples Regularization. WHich is Machine Learning Lifecycle - Data Preperation - Regularization - Lasso/Ridge
    supp1 = supplemental[['Process','Word','Definition','Word_']].rename(columns={'Process':"Categorization"})
    
    temp = consolidated_data.merge(supp1,on=["Categorization",'Word'],how='inner',suffixes=("","_"))
    temp['Definition'] = np.where(temp['Definition_'].notnull(),temp['Definition_'],temp['Definition'])
    temp['Categorization'] = np.where(temp['Definition_'].notnull(),temp['Word'],temp['Categorization'])
    temp['Word'] = np.where(temp['Word_'].notnull(),temp['Word_'],temp['Word'])
    
    temp.drop(['Definition_','Word_'],axis=1,inplace=True)

    # Need to add the Order. Complex due to second level relationship.
    order_df = consolidated_data[['Process','Categorization','Word']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'order'})
    consolidated_data = consolidated_data.merge(order_df,on=['Process','Categorization','Word'],how='left')
    
    temp = temp.merge(consolidated_data[['Process','Word','order']].rename(columns={'Word':'Categorization'}),on=['Process','Categorization'],how='left')
    consolidated_data['order1']=0
    temp['order1'] = 1
    consolidated_df = pd.concat([consolidated_data,temp]).sort_values(['order','order1']).drop(['order','order1'],axis=1)

    # Create a DataFrame for the Highlevel Process, which is Processes and Process Steps.
    process_df = consolidated_df[consolidated_df['Categorization'].isin(['Process','Process Step'])].copy()

    if generate_excel_files:
        knowledge_base_df.to_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx',index=False)
        process_df.to_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/defined_processes.xlsx',index=False)
        consolidated_df.to_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/consolidated_dataset.xlsx',index=False)
    
    return knowledge_base_df,process_df,consolidated_df
    