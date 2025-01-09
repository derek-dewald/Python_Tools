import pandas as pd
import numpy as np
import datetime

current_date = datetime.datetime.now().strftime('%d-%b-%y')

def DailyTask(dictionary,date=current_date,**kwargs):

    df = pd.DataFrame()
    
    for key,value in dictionary.items():
        if key in kwargs:
            value = kwargs[key]
        else:
            value = input(value)

        temp_df = pd.DataFrame([value],index=[date],columns=[key])
        try:
            df = pd.concat([df,temp_df],axis=1)
        except:
            pass

    print(df['Text'].item())
            
    return df.reset_index().rename(columns={'index':'Date'})
