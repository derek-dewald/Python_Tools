'''
module_word: utility_functions
module_definition: Location for functions which do not fall within the scope of any other classification and are Simple, High Generalized, General Purpose functions.

'''
import pandas as pd
import numpy as np
import inspect

def InspectFunction(function_name):
    print(inspect.getsource(function_name))


def extract_tables_from_db(
    db_name='Analytics.INFORMATION_SCHEMA.COLUMNS',
    table_dictionary_source= {'location':'C:\\Users\\ddewald\\desktop\\D.xlsx',"sheet_name":'Tables'},
    include_columns=True,
    exclude_schemas=['Cyder','Datamart','dbo','TEST']
    
):
      
    if include_columns:
        sql = f'''  
SELECT
    TABLE_SCHEMA,
    TABLE_NAME,
    COLUMN_NAME,
    DATA_TYPE,
    CHARACTER_MAXIMUM_LENGTH
FROM {db_name}
ORDER BY
    TABLE_SCHEMA,
    TABLE_NAME,
    ORDINAL_POSITION'''
        
    else:
        sql = '''
SELECT TABLE_SCHEMA, TABLE_NAME
FROM Analytics.INFORMATION_SCHEMA.TABLES
ORDER BY TABLE_SCHEMA, TABLE_NAME'''
        
    df = TIME_SQL(sql)
    
    try:
        dd = pd.read_excel(table_dictionary_source['location'],sheet_name=table_dictionary_source['sheet_name'])
        dd = dd.rename(columns={'Table':"TABLE_NAME",'Schema':'TABLE_SCHEMA'})
        
        hist_df = dd[dd['Historical']==1].copy()
        hist_df['TABLE_SCHEMA'] = 'HISTORICAL'
        dd = pd.concat([dd,hist_df])
        df = df.merge(dd,on=['TABLE_NAME','TABLE_SCHEMA'],how='left')
        
    except:
        pass
        
    df = df[~df['TABLE_SCHEMA'].isin(exclude_schemas)]
        
    return df