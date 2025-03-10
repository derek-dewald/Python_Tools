import pandas as pd


def ConvertDate(df,
                column_name,
                new_column_name="",
                normalize=0):
    '''
    Function to Convert Str to Datetime for Dataframe Column
    
    Parameters:
        column_name (str): Name of Column to Convert
        new_column_name (str): If populated, it will create a new column Name, otherwise it will replace column_name
        normalize (int): Binary Flag, if 0 then no normalization, if 1 then .dt.normalize applied.
        
    Returns:
        Nil 
    
    
    
    '''
    
    if new_column_name =="":
        new_column_name = column_name
    
    if normalize==1:
        df[new_column_name] = pd.to_datetime(df[column_name],errors='coerce').dt.normalize()
    else:
        df[new_column_name] = pd.to_datetime(df[column_name],errors='coerce')
        
def ConvertDateColumns(df,normalize=0):
    
    '''
    Function which applies the ConverDate function to all Columns in DF with word DATE. It overwrites Existing Data.
    
    Parameters:
        Nil
        
    Returns:
        Nil 
    
    '''
    
    
    for column in df.columns:
        if column.upper().find('DATE')!=-1:
            ConvertDate(df,column,normalize=normalize)
            