import pandas as pd
import numpy as np
import datetime

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from data_d_dicts import links

def generate_dictionary(notes_df=None,
                        definition_df=None,
                        export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/DataDictionary/d_learning_notes.csv'):

    '''
    Function Used to Generate D Learning Notes, which is a consolidated view of D Notes and D Definitions.
    Used as a crtical input to D Streamlit Dashboard.

    CSV Generated from this process saved to Github and Used for Streamlit Dashboard.

    Approach takes D Notes as the Primary Source and then merges in Definitions, based on a 4 step integration process.

    Approach uses some Filtering Logic
    
    Step1: 
        When Word in Both Sheets is Identical, then utilize word as definition.
        If Word Does not have a Definition in the D Notes, then replace blank with the definition. 
        If word has a definition, then add a new record.
        
    Step2:
        Utilzing Function notes_word_equals_word_definition_categorization, integration where notes.word=definitions.categorization
        It also begins to replace Word.
        
    Step3:
        Utilizing Function notes_word_equals_definition_process, integration where notes.word=definitions.process
        
    Step4: 
        Merge all other Definitions into Notes Sheet. 

    Parameters:
        notes_df (df): If Blank, then Function will generate call
        definitions_df(df): If Blank Function will call.
        export_Location (str): If populated, location where .csv file will be saved.

    Returns:
        Object Type

    date_created:12-Jan-26
    date_last_modified: 12-Jan-26
    classification:TBD
    sub_classification:TBD
    usage:
        notes_df,examine_further= generate_dictionary()
        notes_df,insert_records,insert_records2,insert_records3,examine_further= generate_dictionary()


    
    '''

    d_notes_categorization_order = ['Definition',
                                    'Guiding Principle',
                                    'Algorithm',
                                    'Consideration',
                                    'Process Step']

    def notes_word_equals_word_definition_categorization(df,notes_df):
        df = df.copy()
        # Create Definition where Definition Categorization Equals Word.
        df['WORD_IS_ZATION'] = np.where(df['Categorization_DEF'].isin(notes_df['Word'].unique().tolist()),1,0)
        
        merge_ = df[df['WORD_IS_ZATION']==1].copy()
        residual_values= df[df['WORD_IS_ZATION']!=1].copy()
        
        merge_ = merge_[['Categorization_DEF','Word','Definition_DEF']].copy()
        merge_['Definition'] = merge_['Word'] + ": " + merge_['Definition_DEF']
        merge_ = merge_.drop(['Definition_DEF','Word'],axis=1).rename(columns={'Categorization_DEF':"Word"})
        merge_['Categorization'] = 'Definition'
    
        merge_ = merge_.merge(notes_df.drop_duplicates('Word')[['Word','Process']],on='Word',how='left')

        residual_values = residual_values[['Process_DEF','Categorization_DEF','Word','Definition_DEF']].rename(columns={'Process_DEF':'Process',
                                                                                             'Categorization_DEF':'Categorization',
                                                                                             'Definition_DEF':'Definition'})
        return merge_,residual_values

    def notes_word_equals_definition_process(examine_further,notes_df):
        '''
            
        '''
    
        df = examine_further.copy()
        notes_df = notes_df.copy()
    
        # Make List
        notes_word_list = notes_df['Word'].unique().tolist()
    
        examine_further = df[~df['Process'].isin(notes_word_list)].copy()
        insert_df = df[df['Process'].isin(notes_word_list)].copy()
    
        # Clean Up Insert DF so it meets Notes DF Structure, Not Definition DF Structure.
        # Definition Updated to Include Word
        # Word is Updated to Include Process.
        # Categorization Does not Change
        # Process Takes on Whatever Value in Notes is, as this is now a input into that Process.
        
        # Word is Updated to include Categorization (needs to happen After Definition is Change)
        
        insert_df['Definition'] = insert_df['Word'] + ": " +  insert_df['Definition'] 
        insert_df['Word'] = insert_df['Process'].copy()
        insert_df.drop(['Process'],axis=1,inplace=True)
        insert_df = insert_df.merge(notes_df.drop_duplicates('Word')[['Word','Process']],on='Word',how='left')
    
        return examine_further,insert_df

        
    from data_d_dicts import links

    try:
        notes_df = notes.copy()
    except:
        notes_df = pd.read_csv(links['google_notes_csv'])
        
    try:
        definition_df = definition_df.copy()
        definition_df.rename(columns={'Process':'Process_DEF','Categorization':"Categorization_DEF",'Definition':"Definition_DEF"},inplace=True)
    except:
        definition_df = pd.read_csv(links['google_definition_csv']).rename(columns={'Process':'Process_DEF','Categorization':"Categorization_DEF",'Definition':"Definition_DEF"})
        definition_df.sort_values(['Process_DEF','Categorization_DEF','Word'],inplace=True)
        
    # Step 1: Merge Definitions into Words where they explicitly Match. No logic required.
    # Identify Where there is a Record. 
        # Example 1: ML Project >> Process Step >> Problem Definition
            # This is a Definition to a process which has Steps. Need to Merge a NEW RECORD.
        # Example 2: Best Linear Unbiased Estimator
            # This is a Example which needs Definitions Merged in, some of which arent direct Definitions. 
    
    # We will create a modified Definition DF. To identify where direct matches exist.
    # Need to distribute information from temp_def until Empty. Need Data Quality View Steps.

    temp_def = definition_df[['Process_DEF','Categorization_DEF','Word','Definition_DEF']].merge(notes_df[['Process','Categorization','Word','Definition']].drop_duplicates('Word'),on='Word',how='left',indicator=True)
    merge = temp_def[temp_def['_merge']=='both'].drop('_merge',axis=1)
    examine_further = temp_def[temp_def['_merge']!='both'].drop('_merge',axis=1)
    # Naming for trouble shooting

    # Two Types of Merge
    # Inserting New Records, specifically Where the existing Note has a Definition, this means there is an existing Process. So this
    # Value Represents a Definition, and there is a New Record Insert

    # Merging Definition. WHere the existing note has No Definition, then we will incorporate a definition. This technically could be 
    # Ignored, but by doing this, I can indirectly influence Order without direct assignment by assuming the Notes Order is Reference Point.

    # Insert Record
    insert_records = merge[merge['Definition'].notnull()].drop(['Process_DEF','Categorization_DEF','Definition','Categorization'],axis=1).rename(columns={'Definition_DEF':"Definition"})
    insert_records['Categorization'] = 'Definition'
    insert_records['Source'] = 'Insert Records'
    
    merge_records  = merge[merge['Definition'].isnull()]
    
    # WAIT TO MERGE UNTIL END IF POSSIBLE. THAT WAY I ONLY NEED TO RANK ONCE.
    
    notes_df =  notes_df.merge(merge_records[['Word','Definition_DEF']],on='Word',how='left')
    notes_df['Definition'] = notes_df['Definition'].fillna(notes_df['Definition_DEF'])
    notes_df.drop('Definition_DEF',axis=1,inplace=True)
    notes_df['Source'] = 'Notes DF'
    #### Step 2: 

    # Insert Instances where Word In Notes is Equal to Categorization in Definition. 
    # When a Process as Steps which are defined and need to be included, but Also Represent a Definition unto themselves.
    # Example Bias, Bias Variance Trade Off, they are STEPS to in the Feature Selection Process, but also key terms which 
    # Deserve their own definition and explanation.

    insert_records2, examine_further = notes_word_equals_word_definition_categorization(examine_further,notes_df)
    insert_records2['Source'] = 'Insert Records2'
    # Start To Bring Ordering to the Data Set.

    c1_df = notes_df[['Process']].drop_duplicates().reset_index(drop=True).assign(C1_ORDER=lambda df: df.index+1)
    c12_df = notes_df[['Process','Categorization']].drop_duplicates().reset_index(drop=True).assign(C12_ORDER=lambda df: df.index+1)
    c3_df = notes_df[['Word']].drop_duplicates().reset_index(drop=True).assign(C3_ORDER=lambda df: df.index+1)

    examine_further,insert_records3 = notes_word_equals_definition_process(examine_further,notes_df)
    insert_records3['Source'] = 'Insert Records3'
    examine_further['Source'] = 'Examine Further'
    
    # Determine Order for Sorting.
    notes_df = notes_df.merge(c1_df,on='Process',how='left')
    notes_df = notes_df.merge(c3_df,on=['Word'],how='left')

    insert_records = pd.concat([
        insert_records,
        insert_records2,
        insert_records3,
        examine_further
    ])

    # Determine COL1 and COL3 Sort Order
    insert_records = insert_records.merge(c1_df,on='Process',how='left')
    insert_records = insert_records.merge(c3_df,on='Word',how='left')

    # Combine ALl DF
    notes_df = pd.concat([
        notes_df,
        insert_records
    ])

    # Determine COL2 Sort Order, which is based on Mapping and Can be done once when consolidated.
    col2_order_dict = {d_notes_categorization_order[x]:x+1 for x in range(len(d_notes_categorization_order))}
    notes_df['C2_ORDER'] = notes_df['Categorization'].map(col2_order_dict)
    notes_df['C2_ORDER'] = notes_df['C2_ORDER'].fillna(1000)   

    notes_df =  notes_df.sort_values(['C1_ORDER','C3_ORDER','C2_ORDER'])

    # Need to insure that every record 
    
    #return notes_df,insert_records,insert_records2,insert_records3,examine_further
  
    notes_df = notes_df.drop(['C1_ORDER','C3_ORDER','C2_ORDER'],axis=1)

    if export_location:
        notes_df.to_csv(export_location,index=False)

    return notes_df,examine_further

def review_test_results(
    file_location='/Users/derekdewald/Documents/Python/Github_Repo/Data/daily_test_results.csv'):

    '''
    Function to Facilitate a Daily Review of Historically Created Words. 
    Function has a scoring Component, a Correct Answer (Pass is worth 1 Point), A Incorrect Answer (Fail is worth -2 points), if cummulative score is not postive
    then user is expected to Answer, with expectation of making score positive, if less than 10 examples with a negative score, it randomly samples from 
    positive scores.

    
    '''
    
    primary_key = ['Process','Categorization','Word']
    
    results_df = pd.read_csv(file_location)

    results_df.loc[results_df["Word"] == 'SGDClassifier', "Historical_Score"] = 2
    
    results_df["Date"] = pd.to_datetime(results_df["Date"], format='%Y-%m-%d')
    results_df["Date"] = results_df["Date"].apply(lambda x:x.date())
    
    # Generate Data Set to Test. 
    # Test Everything where Review_Score is 0.
    review_df = results_df[(results_df['Historical_Score']<0)]

    # If Less than 10 Items to review then sample historical items 
    if len(review_df)<10:
        review_df = pd.concat([
            review_df,
            results_df[(results_df['Historical_Score']>0)].sample(10-len(review_df))
        ])
        
    review_df = review_df.reset_index(drop=True)

    results_dict = {}
    for count in range(len(review_df)):
        cat,cat1,word = review_df[primary_key].iloc[count]    
        print(f"Process: {cat}\nClassification: {cat1}\nWord: {word}\n")
        print("#############################################################################################################################")
        result = input('Did you Pass or Fail?')
        results_dict[word] = [1 if result.lower() =='p' else -2][0] 
        df, nt,lk,md,ds, lt,ac = review_df.iloc[count][['Definition','Notes','Link','Markdown Equation','Dataset Size','Learning Type',"Algorithm Class"]]
        print(f"Definition: {df}\nNotes: {nt}\nLink: {lk}\nMarkdown: {md}\nDataset Size: {ds}\nLearning Type: {lt}\nAlogrithm Class: {ac}\n")

    results_df['Temp_Score'] = results_df['Word'].map(dict_).fillna(0)
    results_df['Historical_Score'] = results_df['Historical_Score'] + results_df['Temp_Score'] 
    results_df['Temp_Score'] = 0

    result_df.to_csv(file_location,index=False)
    
    return results_df


def daily_test(observations=5,
               file_location='/Users/derekdewald/Documents/Python/Github_Repo/Data/daily_test_results.csv'):
    '''
    
    '''
    # Set Definitions
    primary_key = ['Process','Categorization','Word']
    
    # Source Required Data
    
    # Definitions from Google
    definitions = pd.read_csv(links['google_definition_csv'])
    
    # Import Daily Results Tracker 
    results_df = pd.read_csv(file_location)
    results_df["Date"] = pd.to_datetime(results_df["Date"], format='%Y-%m-%d')
    results_df["Date"] = results_df["Date"].apply(lambda x:x.date())

    # Update Results DF to Include Newest Information
    final_df = results_df[['Date','Temp_Score','Historical_Score','Word']].merge(definitions,on='Word',how='left')
    
    # Sample New Results and Test on Historical Learning Opportunities.
    today_test = definitions[~definitions['Word'].isin(final_df['Word'])].sample(observations)
    today_test = today_test.reset_index(drop=True)
    today_test = today_test.fillna('')
    
    today_test['Date'] = datetime.datetime.now().date()

    result_score = {}
    for count in range(len(today_test)):
        cat,cat1,word = today_test[primary_key].iloc[count]
        print(f'Word {count+1}')
        print(f"Category: {cat}\nClassification: {cat1}\nWord: {word}\n")
    
        if today_test.iloc[count]['Definition']=="":
            result_dict[word] = -2
            input(f'Update Google Sheet with Definition\n')
        else:
            input("######################### ANSWER QUESTION #########################")
            df, nt,lk,md,ds, lt,ac = today_test.iloc[count][['Definition','Notes','Link','Markdown Equation','Dataset Size','Learning Type',"Algorithm Class"]]
            print(f"Definition: {df}\nNotes: {nt}\nLink: {lk}\nMarkdown: {md}\nDataset Size: {ds}\nLearning Type: {lt}\nAlogrithm Class: {ac}\n")
            score = input(f'What was the result? (P/F)\n')
            result_dict[word] = [1 if result.lower() =='p' else -3][0]

    today_test['Historical_Score'] = today_test['Word'].map(result_dict)
    today_test['Temp_Score']= 0
    
    # Score Data
    final_df = pd.concat([final_df,today_test]).reset_index(drop=True)
    final_df.to_csv(file_location,index=False)

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
    from data_d_dicts import function_table_dictionary
    from dict_processing import dict_to_dataframe

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


