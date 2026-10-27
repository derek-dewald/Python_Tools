'''
module_word: project_process_tools
module_definition: Repo for functions which support the administration, documentation and guidance of Projects.

'''
from object_rendering import visualize_dataframe_in_notebook,print_dict
from IPython.display import Markdown,display
from objects_automated import object_dict
import pandas as pd
import numpy as np



def generate_process_dictionary_from_knowledge_base(
    df=None,
    process='Machine Learning Lifecycle',
    return_value='project_dict'
):
    '''
    Definition:
        Function Used to created a Dataframe and Dictionary which Extracts the Expected Processes for a specific activity from Knowledge Base. To be used to help
        guidance project requirements and activities required.
        
    Parameters:
        df(Dataframe): Can import Knowledge Base Directly to save from calling Git.
        process(str): Desired Process to be manually documented. Default is Machine Learning Lifecycle
        return_value(str): Three Options. print_dict/project_dict
            project_dict: Returns a Python Dictionary, which can be used to manage the project
            print_dict: Returns Text, Explaining Steps as Outlined in Knowledge Base
                
    Returns:
        dict_, rendered object
    Date Created:
        09-Oct-26
    Date Last Modified:
        26-Oct-26
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

    if df is None or df.empty:
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

    break_text = '''
##########################################################################
Generate Manual Dictionary    
##########################################################################
'''

    if return_value=='project_dict':
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
    else:
        print_dict(process_dict)


def update_process_df_status(
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
    score_dict = {}
    project_step = []
    for item in project_status_dict.keys():
        score_dict[item] = project_status_dict[item]['Is Completed']
        project_step.append(item[:item.find(' - ')])

    df = pd.DataFrame(score_dict.values(),index=score_dict.keys(),columns=['IS_COMPLETE']).reset_index().rename(columns={'index':"Project Step"})
    df['Step'] = project_step

    complete = df['IS_COMPLETE'].sum()
    remaining = len(df['IS_COMPLETE'].notnull()) - complete
    complete_perc = (complete/len(df))*100

    print(f'You have currently completed {complete} for a straight line approximated completion percentage of {complete_perc:.2f}%, there are {remaining} remaining')

    display(Markdown(f"## Step Status"))
    step_df = (df[['Step','IS_COMPLETE']]
        .groupby('Step')
        .agg(
            Completed=('IS_COMPLETE', 'sum'),
            Total=('IS_COMPLETE', 'count')
        )
        .reset_index().rename(columns={'sum':"",'count':'Total Steps'})
              )
    step_df['Percentage Completed'] = step_df['Completed']*100/step_df['Total']

    visualize_dataframe_in_notebook(
        step_df,
        show_as_percentage=['Percentage Completed']
    )

    display(Markdown(f"## Completed items"))
    visualize_dataframe_in_notebook(df[df['IS_COMPLETE']==1])

    display(Markdown(f"## Completed Not Completed"))
    visualize_dataframe_in_notebook(df[df['IS_COMPLETE']==0])