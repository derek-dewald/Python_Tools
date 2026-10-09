'''
module_word: project_process_tools
module_definition: Repo for functions which support the administration, documentation and guidance of Projects.

'''
from object_rendering import visualize_dataframe_in_notebook,print_dict
from IPython.display import Markdown,display
from objects_automated import object_dict
import pandas as pd
import numpy as np


def generate_process_dictionary_from_knowledge_base(df=None,process='Machine Learning Lifecycle'):
    '''
    Definition:
        Function Used to created a Dataframe and Dictionary which Extracts the Expected Processes for a specific activity from Knowledge Base. To be used to help
        guidance project requirements and activities required.
        
    Parameters:
        df(Dataframe): Can import Knowledge Base Directly to save from calling Git.
        process(str): Desired Process to be manually documented. Default is Machine Learning Lifecycle
    Returns:
        df,text
    Date Created:
        09-Oct-26
    Date Last Modified:
        09-Oct-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        project_df = generate_process(knowledge_base) 
    Notes:
        None
    Required Functions:
        print_dict
        object_dict
        
    '''

    if len(df)==0:
        try:
            df = pd.read_excel("/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx")
        except:
            df = pd.read_excel(object_dict['csv_links']['python_object']['d_knowledge_base_url'])

    df = df.copy()

    # filter for Only desired PROCESS
    df = df[df['Process']==process].reset_index(drop=True)

    # Do not want to include the Definition for the selected process
    df = df[(df['Categorization']!='Process')].copy()

    if len(df)==0:
        print('Process Not Identified')
        return df
        
    process_dict = {}
    progress_list = []

    for index_,row in df.iterrows():
        # If it is process step, then it is a New Process and needs a new entry in the Dict.
        if row['Categorization']=='Process Step':
            process = row['Word']
            process_dict[process] = {
                'Definition':row['Definition'],
                'Step Activities':{}
            }

            if len(df[df['Categorization']==process])==0:
                progress_list.append(process)
        else:
            process_dict[process]['Step Activities'][row['Word']] = row['Definition']
            progress_list.append(f"{process} - {row['Word']}")

    print_dict(process_dict)

    break_text = '''
##########################################################################
Generate Manual Dictionary    
##########################################################################
'''
    print(break_text)
    print('project_status_dict = {')
    for item in progress_list:
        text = f"""'{item}':
        {{
        'Comments':[],
        'Pending Items':[],
        'Assumptions':[],
        'Is Completed':0
        }},
        """
        print(text)
    print('}')

    return df
def update_process_df_status(
    df,
    project_status_dict
):

    '''
    Definition:
        Function Used to Quickly Visualize and Calculate the Status of a Project which has utilized the Manual Dictionary Maintenance approach as exported from generate_process_dictionary_from_knowledge_base
    Parameters:
        df(dataframe): Dataframe
        project_status_dict(dict): Dictionary. Manually created from generate_process_dictionary_from_knowledge_base
    Returns:
        Text
    Date Created:
        09-Oct-26
    Date Last Modified:
        09-Oct-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        update_process_df_status(df,project_status_dict)
    Notes:
        None
    Required Functions:
        generate_process_dictionary_from_knowledge_base
        visualize_dataframe_in_notebook
        Markdown
        display
    
    '''
    
    project_status_completion_dict = {x:project_status_dict[x]['Is Completed'] for x in project_status_dict.keys()}
    df['MATCH_COL'] = df['Categorization'] + ' - ' + df['Word']

    df["T"] = df['MATCH_COL'].map(project_status_completion_dict)
    df["T1"] = df['Word'].map(project_status_completion_dict)
    df["IS_COMPLETE"] = np.where(df['T'].notnull(),df['T'],df['T1'])

    df = df[df['IS_COMPLETE'].notnull()]
    df['IS_COMPLETE'] = df['IS_COMPLETE'].astype(int)
    
    df.drop(['T','T1','MATCH_COL'],axis=1,inplace=True)
    
    complete = df['IS_COMPLETE'].sum()
    remaining = len(df['IS_COMPLETE'].notnull()) - complete
    complete_perc = (complete/len(df))*100

    print(f'You have currently completed {complete} for a straight line approximated completion percentage of {complete_perc:.2f}%, there are {remaining} remaining')
    
    display(Markdown(f"## Completed items"))
    visualize_dataframe_in_notebook(df[df['IS_COMPLETE']==1])

    display(Markdown(f"## Completed Not Completed"))
    visualize_dataframe_in_notebook(df[df['IS_COMPLETE']==0])
