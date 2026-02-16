import pandas as pd
import numpy as np
import datetime

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from shared_folder import read_directory,text_file_import,parse_dot_py_file
from input_functions_ignore import input1,input2,input3
import data_d_dicts,data_d_lists,data_d_strings
from data_d_dicts import function_table_dictionary,links
from dict_processing import dict_to_dataframe
from list_processing import list_to_dataframe
from data_validation import column_segmenter

def generate_dictionary(export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/DataDictionary/d_learning_notes.csv'):

    '''
    
    Function Used to Generate D Learning Notes, which is a consolidated view of D Notes and D Definitions.
    Used as a crtical input to D Streamlit Dashboard.

    V2. Material Update. With cleaned Data Fields, can simplify this process.
    1) Create Unique list from all processes in Notes, if Value in definition is seen in notes, direclty merge. 
    2) 

    
        
    Parameters:
        export_Location (str): If populated, location where .csv file will be saved.

    Returns:
        Object Type

    date_created:12-Jan-26
    date_last_modified: 8-Feb-26
    classification:TBD
    sub_classification:TBD
    usage:
        notes_df = generate_dictionary()
        
    '''
    
    from data_d_dicts import links

    list_ = ['Process','Categorization','Word','Definition']

    # Download Data
    definition_df = pd.read_csv(links['google_definition_csv'])
    notes_df = pd.read_csv(links['google_notes_csv'])

    definition_df = definition_df[list_].copy()
    notes_df = notes_df[list_].copy()

    # Step 1. Create a list of unique Processes from Notes.
    process_list = notes_df['Process'].unique().tolist()
    process_list.extend([x for x in definition_df['Process'].unique() if (x not in process_list)&(pd.notna(x))])
    process_map = {x:count+0 for count,x in enumerate(process_list)}

    # Step 2 Create a Unique Classification List
    categorization_list = [
        'Definition','Guiding Principle','Consideration','Process Step','Procedure','Expected Outcomes','Parameter','Algorithm']
    
    categorization_list.extend([x for x in notes_df['Categorization'].unique() if (x not in categorization_list)&(pd.notna(x))])
    categorization_list.extend([x for x in definition_df['Categorization'].unique() if (x not in categorization_list)&(pd.notna(x))])
    categorization_map = {x:count+0 for count,x in enumerate(categorization_list)}

    final_df = pd.concat([notes_df,definition_df])
    final_df['PRO_ORDER'] = final_df['Process'].map(process_map)
    final_df['CAT_ORDER'] = final_df['Categorization'].map(categorization_map)

    final_df = final_df.sort_values(['PRO_ORDER','CAT_ORDER'])
    final_df.drop(['PRO_ORDER','CAT_ORDER'],axis=1,inplace=True)

    if export_location:
        final_df.to_csv(export_location,index=False)

    return final_df

def create_py_table_dict(base_location= '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/',
                         export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/DataDictionary/folder_listing.csv'):
    
    '''
    Function which Generates a Dataframe representing a Function Dictionary, sourcing the Functions from a Shared Folder Location, and
    using the definitions sourced from a Python Dictionary

    Parameters:
        base_location (str): Location of Windows Directory containing .py Files.

    Returns:
        DataFrame

    date_created:4-Dec-25
    date_last_modified:4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        python_function_dict_df = create_py_table_dict()
    '''

    # Get Defined Functions from Dictionary Reference Listing
    temp_ = dict_to_dataframe( function_table_dictionary,key_name='Function Name',value_name='Definition')
    temp_['Type'] = 'Definition'

    py_functions = list_to_dataframe([x for x in read_directory(base_location,file_type='.py') if (x.find('init')==-1)],column_name_list=['File Name'])
    py_functions['Source'] = 'PY File'
    py_functions['Function Name'] = py_functions['File Name'].apply(lambda x:x.replace('.py',''))

    final_df = py_functions.merge(temp_,on='Function Name',how='outer')


    if export_location:
        print(f'folder_listing Saved to {export_location}')
        final_df.to_csv(export_location,index=False)

    return final_df



def parse_dot_py_folder(location=None,
                        export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/DataDictionary/'):
    '''
    
    Function which Allows for the Quick Review of All Python Functions in a Particular Directory, using the functions 
    Read Directory and ParseDDotPYFile

    Parameters:
        location (str): Windows or Mac OS Folder Directory (defaults to D's Mac Directory)
        export_location(str): Location to where CSV file is to be exported. If left Blank, will not export a CSV.

    Returns:
        DataFrame

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        function_list, function_parameters = parse_dot_py_folder()
    
    
    '''

    # GEnerate List of Files

    function_list = pd.DataFrame()
    function_parameters = pd.DataFrame()
    

    if not location:
        folder = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions'

    func_list = read_directory(folder,file_type='.py')
    func_list = [x for x in func_list if (x.find('init')==-1) | (x not in ['d_lists','d_strings','d_dictionaries'])]

    for file_name in func_list:
        filename = f"{folder}/{file_name}"
        file_ = text_file_import(filename)

        temp_a,temp_b = parse_dot_py_file(file_)
        temp_a['Folder'] = file_name
        temp_b['Folder'] = file_name

        function_list = pd.concat([function_list,temp_a])
        function_parameters = pd.concat([function_parameters,temp_b])

    temp_param = pd.concat([
        input3(data_d_strings),
        input1(data_d_dicts),
        input2(data_d_lists)
    ])

    function_parameters = pd.concat([
        function_parameters,
        temp_param
    ])

    if export_location:
        print(f'python_function_list Saved to {export_location}')
        print(f'python_function_parameters Saved to {export_location}')
        function_list.to_csv(f'{export_location}python_function_list.csv',index=False)
        function_parameters.to_csv(f'{export_location}python_function_parameters.csv',index=False)

    return function_list,function_parameters

def daily_test(observations=5,file_location='/Users/derekdewald/Documents/Python/Github_Repo/Data/daily_test_results.csv'):
    '''
    
    '''
    # Set Definitions
    primary_key = ['Process','Categorization','Word']
    
    # Source Required Data
    
    # Definitions from Google
    definitions = pd.read_csv(links['google_definition_csv'])
    historical_results = pd.read_csv(links['d_daily_test_score'])

    historical_results["Date"] = pd.to_datetime(historical_results["Date"], format='%Y-%m-%d')
    historical_results["Date"] = historical_results["Date"].apply(lambda x:x.date())

    test_base = definitions.merge(historical_results[primary_key],on=primary_key, how='outer')
    
    # Import Daily Results Tracker 

    # Update Results DF to Include Newest Information
    final_df = historical_results[['Date','Historical_Score','Word','Process','Categorization']].merge(definitions,on=primary_key,how='left')
    
    # Sample New Results and Test on Historical Learning Opportunities.
    today_test = test_base.sample(observations)
    today_test['Date'] = datetime.datetime.now().date()
    today_test.reset_index(drop=True,inplace=True)
    today_test = today_test.fillna('')
    
    result_dict = {}
    
    for count in range(len(today_test)):
        cat,cat1,word = today_test[primary_key].iloc[count]
        print(f'Word {count+1}')
        print(f"Category: {cat}\nClassification: {cat1}\nWord: {word}\n")
    
        if today_test.iloc[count]['Definition']=="":
            result_dict[word] = -2
            input(f'Update Google Sheet with Definition\n')
        else:
            input("######################### ANSWER QUESTION #########################")
            df, nt,lk,md,ds, lt,ac,mt = today_test.iloc[count][['Definition','Notes','Link','Markdown Equation','Dataset Size','Learning Type',"Algorithm Classification",'Model Type']]
            print(f"Definition: {df}\nNotes: {nt}\nLink: {lk}\nMarkdown: {md}\nDataset Size: {ds}\nLearning Type: {lt}\nAlogrithm Class: {ac}\nModel Type: {mt}")
            result = input(f'What was the result? (P/F)\n')
            result_dict[word] = [1 if result.lower() =='p' else -3][0]

    today_test['Historical_Score'] = today_test['Word'].map(result_dict)
    
    # Score Data
    final_df = pd.concat([final_df,today_test]).reset_index(drop=True)
    
    try:
        final_df.to_csv(file_location,index=False)
    except:
        pass

    return final_df

def review_test_results(file_location=None,
                        sample_records=8):

    '''
    Function to Facilitate a Daily Review of Historically Created Words. 
    Function has a scoring Component, a Correct Answer (Pass is worth 1 Point), A Incorrect Answer (Fail is worth -2 points), if cummulative score is not postive
    then user is expected to Answer, with expectation of making score positive, if less than 10 examples with a negative score, it randomly samples from 
    positive scores.

    
    '''
    # Pull from Here so we don't include todays file.
    if not file_location:
        file_location= '/Users/derekdewald/Documents/Python/Github_Repo/Data/daily_test_results.csv'
    
    primary_key = ['Process','Categorization','Word','Historical_Score']
    
    results_df = pd.read_csv(file_location)
    
    results_df["Date"] = pd.to_datetime(results_df["Date"], format='%Y-%m-%d')
    results_df["Date"] = results_df["Date"].apply(lambda x:x.date())
    
    # Generate Data Set to Test. 
    # Test Everything where Review_Score is 0.
    review_df = results_df[(results_df['Historical_Score']<0)].copy()
    passing_concepts = results_df[(results_df['Historical_Score']>=0)].copy()
    
    column_segmenter(results_df[['Historical_Score']].copy(),'Historical_Score',"",bin_list=[-10,-5,-3,0,3,5])
    
    sample_records = min(sample_records,len(review_df))
    additional_records = max(0,10-sample_records)

    # If Less than 10 Items to review then sample historical items 
    if len(review_df)<sample_records:
        review_df = pd.concat([
            review_df.sample(sample_records),
            passing_concepts.sample(additional_records)
        ])
    else:
        review_df = review_df.sample(sample_records)
        
    review_df = review_df.reset_index(drop=True)

    results_dict = {}
    for count in range(len(review_df)):
        cat,cat1,word,score_ = review_df[primary_key].iloc[count]    
        print(f"Process: {cat}\nClassification: {cat1}\nWord: {word}\n")
        print("#############################################################################################################################")
        result = input('Press Enter for Answer.')
        df, nt,lk,md,ds, lt,ac = review_df.iloc[count][['Definition','Notes','Link','Markdown Equation','Dataset Size','Learning Type',"Algorithm Classification"]]
        print(f"Definition: {df}\nNotes: {nt}\nLink: {lk}\nMarkdown: {md}\nDataset Size: {ds}\nLearning Type: {lt}\nAlogrithm Class: {ac}\n")
        
        # Record Result
        result = input('Did you Pass or Fail?')
        results_df.loc[results_df['Word']==word, "Historical_Score"] +=  + [1 if result.lower() =='p' else -2][0] 
        
    results_df.to_csv(file_location,index=False)
    
    return results_df


def generate_streamlit_definition_summary(file_name):
    
    '''
    
    Generate a series of Group By Statements for visualization of the use of D Google Definitions.
    Generates a CSV file which is saved in shared folder.

    Parameters:
        file_name(str): Specific Name from Dictionary Links, which has the link to 1 of 3 files as default, to provide summary analysis.

    Returns:
        None

    date_created:08-Feb-26
    date_last_modified: 15-Feb-26
    classification:TBD
    sub_classification:TBD
    usage:
        generate_streamlit_definition_summary('google_definition_csv')
        generate_streamlit_definition_summary('d_learning_notes_url')
        generate_streamlit_definition_summary('google_notes_csv')

    update:
        15-Feb-16: Added file_name as parameter, enabling the generation of Notes, Definition or Learning Notes as combinations.

    '''

    output_file_name = f'{file_name}_NUMERIC_SUMMARY'
    
    try:
        df = pd.read_csv(links[file_name])

        df = df[['Process','Categorization','Word','Definition']].copy() 
        df["Process_Count"] = df.groupby("Process")["Process"].transform("count")
        df["CAT_Count"] = df.groupby("Categorization")["Categorization"].transform("count")
        df["Word_Count"] = df.groupby("Word")["Word"].transform("count")
        df['ProcessCAT_Count'] = df.groupby(['Process','Categorization'])["Word"].transform("count")
    
        final_df = df.sort_values(['Process_Count','CAT_Count','Word_Count','ProcessCAT_Count'],ascending=False)
    
        final_df.to_csv(f'/Users/derekdewald/Documents/Python/Github_Repo/Data/{output_file_name}.csv',index=False)
    except:
        pass
    
    return final_df