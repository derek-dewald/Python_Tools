import pandas as pd
import numpy as np
import datetime

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")


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
