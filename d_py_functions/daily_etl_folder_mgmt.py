'''
module_name: daily_etl_folder_mgmt
module_purpose: Function Repository for all manually created functions with the Automation and Administration associated with managing my Diarization System. 

default_structure: Python Function, Documented Consistent with Prevailing Guidance.
module_guidance: Functions which are used in here are to be Stand Alone, sole purpose for management of this daily/monthly/ad hoc periodic files. Can utilize generic functions outside.

'''
import pandas as pd
import numpy as np

from objects_manual import object_dict



def extract_consolidated_raw_dataset(df_dict,export_location=False):
    
    '''
    Create a Consolidated Dataset of files in Dictionary, which are meant to be of the structure Process, Categorization, Word and Definition. Data set for the purposes, 
    of aggregating totals and _____________. Used as a input for generate_objects_automated_py.
    

    Parameters:
        df_dict(dict): Dictionary of files to be included. 

    Returns:
        Dataframe

    date_created:03-Jul-26
    date_last_modified: 03-Jul-26
    classification:TBD
    sub_classification:TBD
    usage:
        df_dict = {
    'Notes':notes_df,
    'Definitions':definition_df,
    'Knowledge Base':knowledge_base_df,
    'Manual Objects':manual_object_df
    }
    
    extract_consolidated_raw_dataset(df_dict)
    
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

def generate_objects_automated_py(
    links_df=pd.DataFrame(),
    consolidated_df=pd.DataFrame()):
    '''

    Create objects_automated.py.

    Parameters:
        links_df(df): Data from Google Sheets with Location of CSV and Links. If nothing included, will pull directly
        consolidated_df (df): Dataset created from extract_consolidated_raw_dataset.py, 

    Returns:
        Object Type

    date_created:29-Jun-26
    date_last_modified: 29-Jun-26
    classification:TBD
    sub_classification:TBD
    usage:
        generate_objects_automated_py(links_df,consolidated_df)
        
    '''
    # Import Data
    if len(links_df)==0:
        links_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=469651051&single=true&output=csv')
    if len(consolidated_df)==0:
        consolidated_df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/consolidated_dataset.xlsx')

    # Generate Dataframes
    # Generate Unique List of Processes and Categories
    process_list = consolidated_df[consolidated_df['Process'].notnull()]['Process'].unique().tolist()
    cat_list = consolidated_df[consolidated_df['Categorization'].notnull()]['Categorization'].unique().tolist()

    # Generate df of CSV and URL
    csv_link_df = links_df[links_df['CSV'].notnull()]
    url_link_df = links_df[links_df['Link'].notnull()]


    text_ = f"""

'''
module_name: objects_automated
module_purpose: Created to serve as a repository for automatically created lists, dictionaries and strings from Google Notes, Dictionaries and other sources as appropriate.  File is created by _____. Whenever run it is automatically overwriden
    
'''
object_dict = {{}}

object_dict['cat_reference_list'] = {{
    'Process':"Categorization Values Currently in Use",
    'Categorization':'Reference List',
    'Word':"Parameter",
    'Definition':"Comprehensive List of all Values utilized in Organizational Taxonomy in the Column Categorization",
    'publish':1,
    'python_object':{cat_list}
    }}

object_dict['process_reference_list'] = {{
    'Process':"Process Values Currently in Use",
    'Categorization':'Reference List',
    'Word':"Parameter",
    'Definition':"Comprehensive List of all Values utilized in Organizational Taxonomy in the Column Process",
    'publish':1,
    'python_object':{process_list}
    }}

object_dict['csv_links'] = {{
    'Process':"Organization",
    'Categorization':'Reference Dictionary',
    'Word':"CSV Links",
    'Definition':"Dictionary of Links to Google Sheet, Git Hub and other pertinent datasource",
    'publish':0,
    'python_object':{csv_link_df.set_index('COLUMN')[['CSV']].to_dict()['CSV']}
        }}
        
object_dict['url_links'] = {{
    'Process':"Organization",
    'Categorization':'Reference Dictionary',
    'Word':"URL Links",
    'Definition':"Dictionary of Links to Google Sheet, Git Hub and other pertinent datasource",
    'publish':0,
    'python_object':{url_link_df.set_index('COLUMN')[['Link']].to_dict()['Link']}
        }}
    """
    
    with open("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/objects_automated.py", "w") as f:
        f.write(text_)


def extract_object_dot_py(
    object_dict,
    export_location):
    
    '''

    Generate an Excel file of all parameters which are manually stored and maintained. File which is generated is used a input component for generate_knowledgebase.
    
    Parameters:
        object_dict (dict): Dictionary Object from either objects_manual or objects_automated
        export_location(str): Location where .xlsx file will be saved. 

    Returns:
        df

    date_created:29-Jun-26
    date_last_modified: 3-Jul-26
    classification:TBD
    sub_classification:TBD
    usage:
        manual_object_df(object_dict_manual,'/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/object_manual_published.xlsx')
    update:
        Added Definition as new record, added column definition.
        To support Automated Object included dict Argument in Required definitions, removed defauly string
    
    '''

    df = pd.DataFrame(object_dict.values())
    df['key'] = object_dict.keys()
    
    final_df = pd.DataFrame()
    
    for item in df[df['publish']==1]['key']:
        temp = object_dict[item]
        temp_df = pd.DataFrame(temp['python_object'],columns=['Word'])
        temp_df['Categorization'] = temp['Categorization']
        temp_df['Process'] = temp['Process']
        temp_df['Definition'] = ""
        temp_df.loc[len(temp_df)] = {
            "Process": temp['Process'],
            "Categorization": "Definition",
            "Word": 'General Definition',
            "Definition": temp['Definition']}
        final_df = pd.concat([final_df,temp_df])    


    #########
    # Do I need to remove Order from Definitions
    #########
    
    final_df['Order'] = final_df.groupby(['Process']).cumcount()+1
    final_df['Categorization'] = np.where(final_df['Categorization']=='Process','Process Step',final_df['Categorization'])

    # is Manual Published, Need Order
    if export_location.find('object_manual_published')!=-1:
        final_df[['Process','Categorization','Word','Definition','Order']].to_excel(export_location,index=False)
    else:
        final_df[['Process','Categorization','Word','Definition']].to_excel(export_location,index=False)


def generate_knowledgebase(
    notes_df=pd.DataFrame(),
    definition_df=pd.DataFrame(),
    manual_object_df=pd.DataFrame(),
    auto_object_df=pd.DataFrame(),
    export_location=None):

    '''
    Process Utilized to Combine Notes/ Definitions and Logic into Knowledge Base, which is utilized to Create, Processes, Parameters.

    Parameters:
        notes_df (dataframe): Dataframe containing Notes from Google. Default is none and it will pull directly from Google.
        definition_df (dataframe): Dataframe containing Definitions from Google. Default is none and it will pull directly from Google.
        manual_object_df (dataframe): Dataframe containining Dataframe of Parameters, generated from Python Process _____, which converts lists in objects_manual.py.
        export_location(str): Name of File to export excel file to. If Blank, returns nothing

    Returns:
        dataframe
        excel_file

    date_created:02-Jul-26
    date_last_modified: 02-Jul-26
    classification:TBD
    sub_classification:TBD
    usage:
        d = generate_knowledgebase(notes_df,definition_df,manual_object_df)
        d = generate_knowledgebase()
 
    '''
    if len(notes_df)==0:
        notes_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSQF2lNc4WPeTRQ_VzWPkqSZp4RODFkbap8AqmolWp5bKoMaslP2oRVVG21x2POu_JcbF1tGRcBgodu/pub?output=csv')

    if len(definition_df)==0:
        definition_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv')

    if len(auto_object_df)==0:
        auto_object_df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/object_auto_published.xlsx')

    if len(manual_object_df)==0:
        manual_object_df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/object_manual_published.xlsx')

    
    # Combine Notes from Google and Notes Extract from Manual List
    consolidated_notes = pd.concat([notes_df,manual_object_df.drop('Order',axis=1)])

    # Take all Processes from Notes, so we can extract Definitions
    process_list = consolidated_notes['Process'].unique().tolist()

    # Also, include any Item in the Consolidated sheet that is a Process Step. 
    process_list.extend(consolidated_notes[consolidated_notes['Categorization']=='Process Step']['Word'].unique().tolist())
    process_list = list(set(process_list))

    # Extract Definitions to include into Consolidated Notes.
    notes_from_def = definition_df[definition_df['Process'].isin(process_list)][['Process','Categorization','Word','Definition']]
    
    # Create a Consolidated notes list (Before Incorporating Definitions).
    
    final_df = pd.concat([
        consolidated_notes,
        notes_from_def])

    # Merge in Sub Processes
    word_list = final_df['Word'].unique().tolist()
    sub_processes = final_df[(final_df['Process'].isin(word_list))]
    #sub_processes = sub_processes.drop(['Categorization'],axis=1).rename(columns={'Word':"Categorization",'Process':'Word','Definition':"Definition"})
    # Include Process.
    
    sub_processes['Definition'] = sub_processes['Word'].fillna("") + ": " + sub_processes['Definition'].fillna('')
    sub_processes['Word'] = sub_processes['Process']
    sub_processes.drop('Process',axis=1,inplace=True)
    sub_processes = sub_processes.merge(final_df[['Word','Process']],on='Word',how='left')


    # Add Automated Dictionary. Here, do not want it causing duplication/ Issue.
    final_df = pd.concat([
        final_df,
        sub_processes,
        auto_object_df
    ])
    
    final_df = final_df.merge(definition_df[['Word','Definition']].rename(columns={'Definition':'Definition1'}),on='Word',how='left')
    final_df['Definition'] = np.where(final_df['Definition'].notnull(),final_df['Definition'],final_df['Definition1'])
    final_df.drop('Definition1',axis=1,inplace=True)
    
    # Sorting
    # Process - Has to be Alphabetical. Do not actually Care Order.
    # Categorization - Based on Manually defined Order, imported from Objects_Manual.
    # Words Matter when they are part of a process, generally when not then alphabetically preferred. 
    
    # Merge in Process  Filter
    proc_filter = manual_object_df[(manual_object_df['Process']=='Notes')&(manual_object_df['Categorization']=='Filter Order')][['Word','Order']].reset_index(drop=True).rename(columns={'Word':"Categorization",'Order':"proc_order"})
    final_df = final_df.merge(proc_filter,on='Categorization',how='left')
    
    # Merge in Word Order.
    final_df = final_df.merge(manual_object_df.drop(['Categorization','Definition'],axis=1).rename(columns={'Order':'word_order'}),on=['Process','Word'],how='left')
    
    final_df.sort_values(['Process','word_order','proc_order','Word'],inplace=True)
    final_df.drop(['word_order','proc_order'],axis=1,inplace=True)

    if export_location:
        final_df.to_excel(export_location,index=False)
       
    return final_df
