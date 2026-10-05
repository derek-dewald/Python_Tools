'''
module_word: d_daily_test
module_definition: Repo for functions related to Testing Aqcquired Knowledge.

'''

import pandas as pd
import numpy as np
import datetime
import random
import time
from IPython.display import Markdown,display

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")
from objects_automated import object_dict
from object_rendering import visualize_dataframe_in_notebook
from utility_functions import gpt_question

def sample_knowledge_test_single_df(
    base_df,
    pause=3,
    test_name='Concepts and Definitions'
):
    '''
    final_results_df = d_daily_test(definition_df,historical_df,wqs_df)

    '''
    base_df = base_df.copy()
    test_words = base_df['Word'].tolist() 
    new_words  = []
    score = []
  
    display(Markdown(f"## Time to Test your Knowledge on {test_name}!"))

    for word in test_words:
        temp_df = base_df[base_df['Word']==word]
        print(f'What is the definition and Classification of: {word}')
        time.sleep(pause)
        display(temp_df.reset_index(drop=True).T.to_dict()[0])
        if input('Did you correctly identify the word? If Yes. 1/0)')=='1':
            score.append(1)
        else:
            score.append(0)
    temp_df = pd.DataFrame(test_words,columns=['Word'])
    temp_df = temp_df.merge(base_df[['Word','Definition']].drop_duplicates('Word'),on='Word',how='left')    
    temp_df['Score'] = score
    temp_df['Date'] = datetime.datetime.now().strftime('%d-%b-%y')
    temp_df['Last Tested'] = datetime.datetime.now().strftime('%d-%b-%y')
    temp_df['Definition'] = np.where(temp_df['Definition'].isnull(),"",temp_df['Definition'])
    temp_df['Classification'] = test_name
    
    return temp_df


def d_daily_test(
    actions=5,
    sample=3,
    pause=1
):
    '''
    Have a decision, do I merge in and count if its on historical then drop, or do I count if its coorect. If its not correct, then it should
    have the potential to be added on, and if I cant ger it correct. Have to be careful due to duplication, if there are multiple instances.
    
    
    '''
    # Download Data 
    historical_df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/d_testing_folder/d_historical_test_results1.xlsx')
        
    definition_df = pd.read_csv(object_dict['csv_links']['python_object']['google_definition_csv'])
    definition_df.to_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/definition.xlsx',index=False)
    definition_df = definition_df[definition_df['Word']!='Definition'].copy()
    def_df = definition_df.merge(historical_df[['Word','Score']].rename(columns={'Score':'HISTORICAL_SCORE'}),on='Word',how='left')
    
    
    wqs_df = pd.read_csv(object_dict['csv_links']['python_object']['google_word_quote'])
    wqs_df = wqs_df.merge(historical_df[['Word','Score']],on='Word',how='left')
    wqs_df = wqs_df[wqs_df['Score'].isnull()].copy()
    
    
    # Start by testing More ML Words
    ml_words = definition_df[definition_df['Process']=='Analytical Method']['Word'].tolist()
    all_other_words = definition_df[definition_df['Process']!='Analytical Method']['Word'].tolist()

    new_word_list = []

    while actions>0:
        # Select ML Words when Available. Except for Last Word, to create a distrubition and Include a Non ML Word.
        if (len(ml_words)>0)&(actions>1):
            w = ml_words.pop(random.randrange(len(ml_words)))
            #print(f'ML Word Selected:{w}')
            new_word_list.append(w)
            actions+= -1
        elif len(all_other_words)>0:
            w = all_other_words.pop(random.randrange(len(all_other_words)))
            #print(f'Non ML Word Selected:{w}')
            new_word_list.append(w)
            actions+= -1

    test_df = definition_df[definition_df['Word'].isin(new_word_list)].copy()

    results_df = sample_knowledge_test_single_df(test_df,pause=pause)

    # Select A Word, Quote, Scripture and Proverb.
    
    word_dict = {}
    word_dict['Word'] = wqs_df[wqs_df['Classification']=='Word'].sample(1)['Word'].item()
    word_dict['Quote'] = wqs_df[wqs_df['Classification']=='Quote'].sample(1)['Word'].item()
    word_dict['Scripture'] = wqs_df[wqs_df['Classification']=='Scripture'].sample(1)['Word'].item()
    word_dict['Proverb'] = wqs_df[wqs_df['Classification']=='Proverb'].sample(1)['Word'].item()
    
    word_list2 = list(word_dict.values())
    
    results_df1 = wqs_df[wqs_df['Word'].isin(word_list2)].drop_duplicates('Word')
    
    for word in word_list2:
        display(Markdown(f"#### {word}"))
        display(Markdown(f"#### {results_df1[results_df1['Word']==word]['Definition'].item()}"))
    
    results_df1['Score'] = 0
    results_df1['Date'] = datetime.datetime.now().strftime('%d-%b-%y')
    results_df1['Last Tested'] = datetime.datetime.now().strftime('%d-%b-%y')

    today_score = pd.concat([results_df,results_df1])
    display(Markdown(f"### Your Score today on New Items was: {100*(today_score['Score'].sum()/len(today_score)):,.2f}%"))
    
    final_results_df = pd.concat([historical_df,today_score])

    gpt_question(new_word_list)
    
    visualize_dataframe_in_notebook(today_score)
    
    final_results_df.to_excel('/Users/derekdewald/Documents/Python/Github_Repo/d_testing_folder/d_historical_test_results1.xlsx',index=False)
    final_results_df.to_excel(f"/Users/derekdewald/Documents/Python/Github_Repo/d_testing_folder/archive/d_historical_test_results_{datetime.datetime.now().strftime('%d-%b-%y')}.xlsx",index=False)   
    
def test_previous_errors(
    min_correct=3
):
    '''

    All Pull new dataset to avoid ____
    
    '''

    df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/d_testing_folder/d_historical_test_results1.xlsx')
    def_df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/definition.xlsx')
    df = df.merge(def_df[['Word','Definition']].drop_duplicates('Word').rename(columns={'Definition':'Definition_'}),on='Word',how='left')
    df.drop_duplicates('Word',inplace=True)
    df['Definition'] = np.where(df['Definition_'].notnull(),df['Definition_'],df['Definition'])
    df = df[['Word','Definition','Score','Date','Last Tested','Classification']]
    df['Date'] = pd.to_datetime(df['Date'],errors='coerce').dt.date
    df['Last Tested'] = pd.to_datetime(df['Last Tested'],errors='coerce').dt.date
    display(Markdown(f"## Time to Test Your Recall. Show me what you've learnt"))

    tested_df = pd.DataFrame()
    for topic in df['Classification'].unique():
        topic_correct_remaining = min_correct
        attempts=0
        temp_df = df[(df['Classification']== topic)&(df['Last Tested']<(datetime.datetime.now()-datetime.timedelta(days=1)).date())].copy()
        topic_correct_remaining = min(topic_correct_remaining,len(temp_df))

        while (topic_correct_remaining >=1)|(attempts<5):
            display(Markdown(f"### Current Topic: {topic}, you have {topic_correct_remaining} Remaining until this topic is Solved"))
            example = temp_df.sample(1)
            tested_df = pd.concat([tested_df,example])
            word = example['Word'].item()
            definition_ = example['Definition'].item()
            print(f'What is the definition of {word}')
            print(f'{word}:\n{definition_}')
            if topic == 'Concepts and Definition':
                print(def_df[def_df['Word']==word].drop(['Word','Definition','Notes','Link','Image','Markdown Equation'],axis=1).iloc[0])
            try:
                val = int(input('Did you get it correct? If Yes, 1 else 0)'))
            except:
                val = 0
            if val==1:
                df['Score'] = np.where(df['Word']==word,val,df['Score'])
                df['Last Tested'] = np.where(df['Word']==word,datetime.datetime.now().date(),df['Last Tested'])
                temp_df = temp_df[temp_df['Word']!=word]
                topic_correct_remaining += -int(val)
                display(Markdown(f"#### That is Correct!"))
            else:
                display(Markdown(f"#### That is incorrect!"))
            attempts +=1
        display(Markdown(f"### Congratulations, you have finished your reivew of {topic}"))

    visualize_dataframe_in_notebook(tested_df)
    
    df.to_excel('/Users/derekdewald/Documents/Python/Github_Repo/d_testing_folder/d_historical_test_results1.xlsx',index=False)