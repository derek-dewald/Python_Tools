import pandas as pd
import numpy as np
import datetime

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

def review_test_results(days=7,
    file_location='/Users/derekdewald/Documents/Python/Github_Repo/Data/daily_test_results.csv'):

    '''
    
    '''
    primary_key = ['Category','Categorization','Word']
    
    results_df = pd.read_csv(file_location)
    results_df["Date"] = pd.to_datetime(results_df["Date"], format='%Y-%m-%d')
    results_df["Date"] = results_df["Date"].apply(lambda x:x.date())

    cut_off = datetime.datetime.now().date() - datetime.timedelta(days=days)
    print(cut_off)

    review_df = results_df[(results_df['Date']>cut_off)&(results_df['Score']==-1)]
    review_df = review_df.reset_index(drop=True)

    for count in range(len(review_df)):
        cat,cat1,word = review_df[primary_key].iloc[count]    
        print(f"Category: {cat}\nClassification: {cat1}\nWord: {word}\n")

        input("######################### THINK ABOUT IT!  #########################")
        df, nt,lk,md,ds, lt,ac = review_df.iloc[count][['Definition','Notes','Link','Markdown Equation','Dataset Size','Learning Type',"Algorithm Class"]]
        print(f"Definition: {df}\nNotes: {nt}\nLink: {lk}\nMarkdown: {md}\nDataset Size: {ds}\nLearning Type: {lt}\nAlogrithm Class: {ac}\n")
    
    return review_df

def daily_test(observations=5,
               file_location='/Users/derekdewald/Documents/Python/Github_Repo/Data/daily_test_results.csv'):
    '''
    
    '''
    # Set Definitions
    primary_key = ['Category','Categorization','Word']
    
    # Source Required Data
    
    # Definitions from Google
    definitons = pd.read_csv(links['google_definition_csv'])
    
    # Import Daily Results Tracker 
    results_df = pd.read_csv(file_location)
    results_df["Date"] = pd.to_datetime(results_df["Date"], format='%Y-%m-%d')
    results_df["Date"] = results_df["Date"].apply(lambda x:x.date())
    
    # Update Results DF to Include Newest Information
    final_df = results_df[['Date','Action','Result','Score','Category','Categorization','Word']].merge(definitons,on=primary_key,how='left')
    
    # Sample New Results and Test on Historical Learning Opportunities.
    today_test = definitons.sample(observations)
    today_test = today_test.reset_index(drop=True)
    today_test = today_test.fillna('')
    
    today_test['Date'] = datetime.datetime.now().date()
    today_test['Action'] = ""
    today_test['Result'] = ""
    
    for count in range(len(today_test)):
        cat,cat1,word = today_test[primary_key].iloc[count]
        print(f'Word {count+1}')
        print(f"Category: {cat}\nClassification: {cat1}\nWord: {word}\n")
    
        if today_test.iloc[count]['Definition']=="":
            today_test.iloc[count, today_test.columns.get_loc('Action')] = 'Test'
            today_test.iloc[count, today_test.columns.get_loc('Result')] = 'Push'
            input(f'Update Google Sheet with Definition\n')
        else:
            today_test.iloc[count, today_test.columns.get_loc('Action')] = 'Update' 
            input("######################### ANSWER QUESTION #########################")
            df, nt,lk,md,ds, lt,ac = today_test.iloc[count][['Definition','Notes','Link','Markdown Equation','Dataset Size','Learning Type',"Algorithm Class"]]
            print(f"Definition: {df}\nNotes: {nt}\nLink: {lk}\nMarkdown: {md}\nDataset Size: {ds}\nLearning Type: {lt}\nAlogrithm Class: {ac}\n")
            score = input(f'What was the result? (Pass/Fail)\n')
    
    final_df = pd.concat([final_df,today_test]).reset_index(drop=True)
    final_df['Result'] = np.where((final_df['Result'].isnull())|(final_df['Result']==""),'Fail',final_df['Result'])

    condition = [
        final_df['Result'] == 'Fail',
        final_df['Result'] == 'Push',
        final_df['Result'] == 'Pass']

    final_df['Score'] = np.select(condition,[-1,0,1],-1)

    final_df.to_csv(file_location,index=False)

    return final_df