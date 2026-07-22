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


def extract_object_dot_py(
    object_dict,
    export_location):
    
    '''
    Definition:
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
        To support Automated Object included dict Argument in Required definitions, removed default string
    
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


def ETL(print_=False):
    '''
    Definition:
        Function which refreshes Organizational Processes and Files. 

    Parameters:
        print_(bool): To Be Determined Utilization.

    Returns:
        objects_automated.py 
        object_manual_published.xlsx
        object_auto_published.xlsx
        knowledge_base.xlsx
        defined_processes.xlsx
        consolidated_dataset.xlsx
        python_function_list.csv
        python_function_parameters.csv
    

    date_created:16-Jul-26
    date_last_modified: 16-Jul-26
    classification:ETL
    sub_classification: ETL
    usage:
        Example Function Call
    '''

    #Specifically Do Not Bulk Call Functions to Ensure Visability as to location and purpose.
    
    # Generate Github_Repo/d_py_functions/objects_automated.py
    try:
        generate_objects_automated_py()
    except:
        pass
    # Export Excel Files for Github_Repo/Streamlit/Data/object_auto_published.xlsx / object_manaul_published.xlsx
    # Function
    from daily_etl_folder_mgmt import extract_object_dot_py
    
    # Data Dictionaries from respective .py Files.
    from objects_automated import object_dict as object_dict_auto
    from objects_manual import object_dict as object_dict_man
    
    extract_object_dot_py(object_dict_man,'/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/object_manual_published.xlsx')
    extract_object_dot_py(object_dict_auto,'/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/object_auto_published.xlsx')
    
    # Import Data from Google and Local Files.
    notes_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSQF2lNc4WPeTRQ_VzWPkqSZp4RODFkbap8AqmolWp5bKoMaslP2oRVVG21x2POu_JcbF1tGRcBgodu/pub?output=csv')
    definition_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv')
    manual_object_df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/object_manual_published.xlsx')
    auto_object_df =   pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/object_auto_published.xlsx')
    
    from daily_etl_folder_mgmt import generate_knowledgebase
    
    # Generate Updated Knowledge Base
    knowledge_base_df = generate_knowledgebase(
        notes_df=notes_df,
        definition_df=definition_df,
        manual_object_df= manual_object_df,
        auto_object_df=auto_object_df,
        export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx')
    
    # Generate Defined Process Listing from Stream Lit from Knowledge Base.
    knowledge_base_df[knowledge_base_df['Categorization']=='Process Step'].to_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/defined_processes.xlsx',index=False)
    
    #########
    # Does not Included Automated Objects Because Automated Objects already Included. 
    #Test this theory.
    ##########
    
    df_dict = {
        'Notes':notes_df,
        'Definitions':definition_df,
        'Knowledge Base':knowledge_base_df,
        'Manual Objects':manual_object_df
    }
    
    from daily_etl_folder_mgmt import extract_consolidated_raw_dataset
    
    # Generates file to /Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/consolidated_dataset.xlsx
    extract_consolidated_raw_dataset(df_dict,True)
    
    from filesystem_tools import parse_dot_py_folder
    # Update Python Function and Parameter List  /Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/python_function_list and python_function_parameters
    function_list,parameter_list = parse_dot_py_folder()

def generate_objects_automated_py(
    links_df=pd.DataFrame(),
    consolidated_df=pd.DataFrame(),
    dot_py_documentation=pd.DataFrame(),
    definition_df=pd.DataFrame(),
    method='local'
):
    '''
    Definition:
        Create Automated Python File
    Parameters:
        links_df(df): Data from Google Sheets with Location of CSV and Links. If Nothing, it will pull from Google.
        consolidated_df (df): Dataset created from extract_consolidated_raw_dataset.py. If nothing, it will pull from Local Source.
        dot_py_documentation (df): Dataset created from _______, representing Python String Documentation. If nothing, it will pull from local Source. 
        definition_df (df): Data from Google Sheets. If nothing will pull from Google.
    Returns:
        Dot Py File
    Date Created:
        29-Jun-26
    Date Last Modified:
        20-Jul-26
    Process:
        Documentation
    Categorization:
        Manual File Creator
    Usage:
        generate_objects_automated_py(links_df,consolidated_df)
    Notes:
        20Jul Update included definition_df, specifically because certain values aren't being passed forward to consolidated_df, values where I do not want to manually maintain the 
        list, I want it to be included based on Automated Definition. Based on process this data isn't exposed, so when running this function the only way to access the list is 
        from the direct source. Slightly confusing perhaps could remove consolidated_df, can iterate.
       
    '''

    if method=='local':
        consolidated_xlsx = '/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/consolidated_dataset.xlsx'
        dot_py_csv = '/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/python_function_list.csv'
    
    else:
        consolidated_xlsx = 'https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/consolidated_dataset.xlsx'
        dot_py_csv = 'https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/python_function_list.csv'
    
    # Import Data
    if len(links_df)==0:
        links_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=469651051&single=true&output=csv')
    if len(consolidated_df)==0:
        consolidated_df = pd.read_excel(consolidated_xlsx)
    if len(dot_py_documentation)==0:
        dot_py_df = pd.read_csv(dot_py_csv)
    if len(definition_df)==0:
        definition_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv')

    # Generate Information Required to Populate Automated Dot Py. 

    # Generate Lists for Consolidated Dataframe - Process and Categorization
    process_list = consolidated_df[consolidated_df['Process'].notnull()]['Process'].unique().tolist()
    process_list.sort()
    cat_list = consolidated_df[consolidated_df['Categorization'].notnull()]['Categorization'].unique().tolist()
    cat_list.sort()
   
    # Generate Definitions from definition_df
    algo_class_list = definition_df[definition_df['Process']=='Algorithm Categorization']['Word'].unique().tolist()
    algo_class_list.sort()

    # Generate Lists for Dot Py String Dataframe - Process and Categorization
    dot_py_proc = dot_py_df[dot_py_df['Process'].notnull()]['Process'].unique().tolist()
    dot_py_cat = dot_py_df[dot_py_df['Categorization'].notnull()]['Categorization'].unique().tolist()
    

    # Generate df of CSV and URL
    csv_link_df = links_df[links_df['CSV'].notnull()]
    url_link_df = links_df[links_df['Link'].notnull()]

    text_ = f"""

'''
module_name: objects_automated
module_purpose: Created to serve as a repository for automatically created lists, dictionaries and strings from Google Notes, Dictionaries and other sources as appropriate.  File is created by _____. Whenever run it is automatically overwriden
    
'''
object_dict = {{}}

object_dict['process_reference_list'] = {{
    'Process':"Documentation Taxonomy Categorization",
    'Categorization':'Reference List',
    'Word':"Documentation Taxonomy Process",
    'Definition':"Comprehensive List of all Values utilized in Organizational Taxonomy in the Column Process",
    'publish':1,
    'python_object':{process_list}
    }}

object_dict['cat_reference_list'] = {{
    'Process':"Documentation Taxonomy Categorization",
    'Categorization':'Reference List',
    'Word':"Documentation Taxonomy Categorization",
    'Definition':"Comprehensive List of all Values utilized in Organizational Taxonomy in the Column Categorization",
    'publish':1,
    'python_object':{cat_list}
    }}

object_dict['algo_class_ref_list'] = {{
    'Process':"Algorithm Categorization",
    'Categorization':'Reference List',
    'Word':"Algorithm Categorization Reference List",
    'Definition':"Comprehensive List of all Values utilized in Algorithm Categorization, used to Categorize ML Methods in Algorithm Taxonomy",
    'publish':1,
    'python_object':{algo_class_list}
    }}

object_dict['dot_py_reference_proc'] = {{
    'Process':"Dot Py String Taxonomy Process",
    'Categorization':'Reference List',
    'Word':"Dot Py String Taxonomy Process",
    'Definition':"Comprehensive List of all Values utilized in Organizational Taxonomy in the Column Process",
    'publish':1,
    'python_object':{dot_py_proc}
    }}

object_dict['dot_py_reference_cat'] = {{
    'Process':"Dot Py String Taxonomy Categorization",
    'Categorization':'Reference List',
    'Word':"Dot Py String Taxonomy Categorization",
    'Definition':"Comprehensive List of all Values utilized in Organizational Taxonomy in the Column Categorization",
    'publish':1,
    'python_object':{dot_py_cat}
    }}
    
object_dict['csv_links'] = {{
    'Process':"CSV Links",
    'Categorization':'Reference Dictionary',
    'Word':"CSV Links",
    'Definition':"Dictionary of Links to Google Sheet, Git Hub and other pertinent datasource",
    'publish':0,
    'python_object':{csv_link_df.set_index('COLUMN')[['CSV']].to_dict()['CSV']}
        }}
        
object_dict['url_links'] = {{
    'Process':"URL Links",
    'Categorization':'Reference Dictionary',
    'Word':"URL Links",
    'Definition':"Dictionary of Links to Google Sheet, Git Hub and other pertinent datasource",
    'publish':0,
    'python_object':{url_link_df.set_index('COLUMN')[['Link']].to_dict()['Link']}
        }}
"""

    with open("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/objects_automated.py", "w") as f:
        f.write(text_)


def generate_knowledgebase(
    notes_df=pd.DataFrame(),
    definition_df=pd.DataFrame(),
    manual_object_df=pd.DataFrame(),
    auto_object_df=pd.DataFrame(),
    export_location=None):

    '''
    Definition:
        Process Utilized to Combine Notes/ Definitions and Logic into Knowledge Base, which is utilized to Create, Processes, Parameters.
    Parameters:
        notes_df (dataframe): Dataframe containing Notes from Google. Default is none and it will pull directly from Google.
        definition_df (dataframe): Dataframe containing Definitions from Google. Default is none and it will pull directly from Google.
        manual_object_df (dataframe): Dataframe containining Dataframe of Parameters, generated from Python Process _____, which converts lists in objects_manual.py.
        export_location(str): Name of File to export excel file to. If Blank, returns nothing
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
        d = generate_knowledgebase(notes_df,definition_df,manual_object_df)
        d = generate_knowledgebase()
    Notes:
        22-Jul - Overhauled merge. Attempted to streamline, simplify and reduce duplication. Increase Visability.
        
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

    # Merge in Definitions.
    final_df1 = final_df.merge(definition_df[['Word','Definition']].drop_duplicates('Word'),on='Word',how='left',suffixes=("","_"))
    final_df1['Definition'] = np.where(final_df1['Definition'].isnull(),final_df1['Definition_'],final_df1['Definition'])
    final_df1.drop('Definition_',axis=1,inplace=True)
    
    # Merge in Processes into themself. IE. In Machine Learning Lifecycle add Goal Setting Steps to makea  complete process
    temp = final_df1[final_df1['Process'].isin(final_df1['Word'].tolist())].copy()
    temp['Definition'] = temp['Word'] +': '+ temp['Definition']
    temp['Word'] = temp['Process']
    temp['Categorization'] = np.where(temp['Categorization']=='Process Step','Guidance',temp['Categorization'])
    temp.drop('Process',axis=1,inplace=True)
    
    final_df1 = final_df1.merge(temp,on='Word',how='left',suffixes=("","_"))
    final_df1['Categorization'] = np.where(final_df1['Categorization_'].notnull(),final_df1['Categorization_'],final_df1['Categorization'])
    final_df1['Definition'] = np.where(final_df1['Definition_'].notnull(),final_df1['Definition_'],final_df1['Definition'])
    
    final_df1.drop(['Definition_','Categorization_'],axis=1,inplace=True)

    
    auto_object_df = auto_object_df.merge(definition_df[['Word','Definition']].drop_duplicates('Word'),on='Word',how='left',suffixes=("","_"))
    auto_object_df['Definition'] = np.where(auto_object_df['Definition'].isnull(),auto_object_df['Definition_'],auto_object_df['Definition'])
    auto_object_df.drop('Definition_',inplace=True,axis=1)
    
    # Add Automated Dictionary. Here, do not want it causing duplication/ Issue.
    final_df1 = pd.concat([
        final_df1,
        auto_object_df
    ])

    # Sorting
    # Process - Has to be Alphabetical. Do not actually Care Order.
    # Categorization - Based on Manually defined Order, imported from Objects_Manual.
    # Words Matter when they are part of a process, generally when not then alphabetically preferred. 
    
    # Merge in Process  Filter
    cat_filter = cat_filter = manual_object_df[manual_object_df['Process']=='Categorization Filter Order'][["Word",'Order']].reset_index(drop=True).rename(columns={'Word':"Categorization",'Order':"proc_order"})
    final_df1 = final_df1.merge(cat_filter,on='Categorization',how='left')
    
    # Merge in Word Order.
    final_df1 = final_df1.merge(manual_object_df.drop(['Categorization','Definition'],axis=1).rename(columns={'Order':'word_order'}),on=['Process','Word'],how='left')
    
    final_df1.sort_values(['Process','word_order','proc_order','Word'],inplace=True)
    final_df1.drop(['word_order','proc_order'],axis=1,inplace=True)    
    final_df1.reset_index(drop=True,inplace=True)

    if export_location:
        final_df1.to_excel(export_location,index=False)
       
    return final_df1