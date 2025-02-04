import pandas as pd
import numpy as np

def BinaryColumnCreator(df,
                        column_name,
                        new_column_name,
                        value,
                        calculation):
  
  if calculation=='>':
    df[new_column_name] = np.where(df[column_name]>value,1,0)
  elif calculation =='=':
    df[new_column_name] = np.where(df[column_name]==value,1,0)
  elif calculation =='<':
    df[new_column_name] = np.where(df[column_name]<value,1,0)
  elif calculation =='isin':
    df[new_column_name] = np.where(df[column_name].isin(value),1,0)
  elif calculation =='contains':
    df[new_column_name] = np.where(df[column_name].str.contains(value),1,0)
  elif calculation == 'dict':
    for key,value in value.items():
        try:
          df[new_column_name] = np.where((df[key]==value)&(df[new_column_name]==1),1,0)
        except:
          df[new_column_name] = np.where(df[key]==value,1,0)


def TextClean(df, 
              column_list, 
              clean_type='only_digits'):

    if clean_type == 'only_digits':
        for column in column_list:
            df[column] = df[column].astype(str).str.replace(r'[^\d.]', "", regex=True)
            df[column] = pd.to_numeric(df[column], errors='coerce')

    return df