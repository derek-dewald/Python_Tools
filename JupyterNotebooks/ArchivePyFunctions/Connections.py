# File Description: Connections are functions which are used to Connect to difference External Sources. 


import pandas as pd
import webbrowser
import datetime
import requests
import os

def DownloadFilesFromGit(user='derek-dewald',
                        repo='Python_Tools',
                        folder='d_py_functions',
                        output_folder=""):
    '''
    Function to Download Files from Github to a dedicated folder. Specifically used when i DO NOT want to formally link to Github.
    
    Parameters:
        User:
        Repo:
        folder:
        output_folder:
        
    Returns:
        Saves files to Output Folder.
    

    '''
    
    if len(output_folder) == 0:
        output_folder = os.getcwd()
    
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder}"
    response = requests.get(api_url)

    if response.status_code == 200:
        files = response.json()
        py_files = [file for file in files if file['name'].endswith('.py')]

        for file in py_files:
            file_url = file['download_url']
            file_name = file['name']
            file_response = requests.get(file_url)

            if file_response.status_code == 200:
                with open(os.path.join(output_folder, file_name), "w", encoding="utf-8") as f:
                    f.write(file_response.text)
                    
        return True
    else:
        return False

def ParamterMapping(Definition=""):
    
    '''
    Function to Google Mapping Sheet, which is used to store Mappings, Links, etc.
    For both simplicity and Organization
    
    Args:
        Definition (Str): Key word used to Access individual elements
        
    Returns:
        Dataframe, unless Definition is defined, in which case it might be Str.
    
    '''

    df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSwDznLz-GKgWFT1uN0XZYm3bsos899I9MS-pSvEoDC-Cjqo9CWeEuSdjitxjqzF3O39LmjJB0_Fg-B/pub?output=csv')
    
    # If user has not included a definition, the return entire DF
    if len(Definition)==0:
        return df
    else:
        try:
            df1 = df[df['Definition']==Definition]
            if len(df1)==1:
                if df1['TYPE'].item()=='csv':
                    return pd.read_csv(df1['VALUE'].item())
                else:
                    return df1['VALUE'].item()
        except:
            return df[df['Definition']==Definition] 


def BackUpGoogleSheets(location='/Users/derekdewald/Documents/Python/Github_Repo/CSV Backup Files/'):
    '''
    Function to Create a Backup of Information Stored in Google Sheets.
    
    Parameters:
        None
        
    Returns:
        CSV Files 
    
    '''
    
    df = ParamterMapping()
    
    for row in range(len(df)):
        try:
            file_name = df['Definition'][row]
            file_location = df['CSV'][row]
            month = datetime.datetime.now().strftime('%b-%y')
            
            temp_df = pd.read_csv(file_location)
            temp_df.to_csv(f'{location}{file_name}_{month}.csv',index=False)
            print(f'Back Up Saved, {location}{file_name}')
        except:
            print(f'Counld Not Print Record {row}')



def GoogleProcessSheetLinks():
     
    '''
    Function to Google Mapping Sheet, Navigate to Specific Sites.
    Provides Options, Enable Selection based on inputs.
    
    Parameters:
    
        
    Returns:
        
    
    '''

    df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSwDznLz-GKgWFT1uN0XZYm3bsos899I9MS-pSvEoDC-Cjqo9CWeEuSdjitxjqzF3O39LmjJB0_Fg-B/pub?output=csv')
    
    display(df)
    
    p = input('Which Process would You like to review?')
    v = input('What would you like to return?')
    
    df1 = df[df['Definition']==p]
    
    if v.lower() =='link':
        webbrowser.open(df1['Link'].item())
    elif v.lower() == 'csv':
        return pd.read_csv(df1['CSV'].item())
    elif v.lower()=='streamlit':
        webbrowser.open(df1['Streamlit'].item())

def NavigateUsingDMap():
     
    '''
    Function to Google Mapping Sheet, Navigate to Specific Sites.
    Provides Options, Enable Selection based on inputs.
    
    Parameters:
    
        
    Returns:
        
    
    '''

    df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSwDznLz-GKgWFT1uN0XZYm3bsos899I9MS-pSvEoDC-Cjqo9CWeEuSdjitxjqzF3O39LmjJB0_Fg-B/pub?output=csv')
    
    display(df)
    
    p = input('Which Process would You like to review?')
    v = input('What would you like to return?')
    
    df1 = df[df['Definition']==p]
    
    if v.lower() =='link':
        webbrowser.open(df1['Link'].item())
    elif v.lower() == 'csv':
        return pd.read_csv(df1['CSV'].item())
    elif v.lower()=='streamlit':
        webbrowser.open(df1['Streamlit'].item())



def CreateTableAnalytics(sql):
    '''
    Function utilized to Run Script Created from Function: generate_create_table_sql
    
    Because the formating in generate_create_table_sql is not beyond reproach, it is a good practice to keep these 
    items distinct and ensure a manual step before Table is created.
    
    Parameters:
        sql(text): SQL Text, generated primarily exclusively by generate_create_table_sql
    
    Returns:
        Creates a New table in Analytics Database
    
    
    '''
    cnxn = Analytics()
    cursor = cnxn.cursor()
    
    cursor.execute(sql)
    cnxn.commit()
    
    cursor.close()
    cnxn.close()

def DeleteAnalyticalTable(table_name,
                          condition=None,
                          by_pass_validation=0,
                          remove_table=0):
    
    cnxn = Analytics()
    cursor = cnxn.cursor()
    
    if remove_table==1:
        proceed = input('Do you want to Delete Table. Only Way forward is Yes')
        if proceed !='Yes':
            print("User Defined Escape")
            return
        delete_sql = f'DROP TABLE {table_name}'
    
    else:
        delete_sql = f'Delete from {table_name}'

        if by_pass_validation==0:
            proceed = input('Do you REALLY want to delete Data from this table. Only Way forward is Yes')

            if proceed !='Yes':
                print("User Defined Escape")
                return
        if condition:
            delete_sql +=  f" {condition}" 
    try:
        cursor.execute(delete_sql)
        cnxn.commit()
    except Exception as e:
        print('Error during Insert',e)
        cnxn.rollback()
    finally:
        cursor.close()
        cnxn.close()


def IterativeTestInsertColumnByColumn(df, table_name,total_rows=5):
    """
    Attempts to insert each column from the first row individually into a test table
    with one column. Used to pinpoint formatting/casting issues.
    """
    

    columns = []
    
    df = df[:total_rows].copy()
            
    for cols in df.columns:
        columns.append(cols)
        print(f"Attempting to Insert:{cols}")
        INSERT_STATEMENT(df[columns],table_name)
        print(f'Successfully Inserted:{cols}')


def UpdateAnalyticsTable(df, table_name, batch_size=0):
    '''
    Function to insert data into a SQL Server table in batches or all at once.
    Handles nulls and datetime values.
    '''
    start_time = timeit.default_timer()

    # Replace NaN and NaT with None
    df = df.where(pd.notnull(df), None).copy()

    if batch_size == 0:
        # Insert all rows at once
        try:
            INSERT_STATEMENT(df, table_name)
            print(f"Inserted all {len(df)} records. Elapsed Time: {timeit.default_timer() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Could not insert data: {e}")
    else:
        # Insert in batches
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            try:
                INSERT_STATEMENT(df.iloc[start:end], table_name)
                print(f"Inserted records {start} to {end - 1}. Elapsed Time: {timeit.default_timer() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Could not insert records {start} to {end - 1}: {e}")




def INSERT_STATEMENT(df,table_name):
    
    '''
    Function Used to Insert Data Into a Table in MS SQL.
    
    
    
    '''
    cnxn = Analytics()
    cursor = cnxn.cursor()
    
    cursor.fast_executemany=True
    
    columns = ', '.join(f"[{col}]" for col in df.columns)
    placeholders = ', '.join(['?'] * len(df.columns))
    insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    data = list(df.itertuples(index=False,name=None))
    
    try:
        cursor.executemany(insert_sql,data)
        cnxn.commit()
    except Exception as e:
        print('Error during Insert',e)
        cnxn.rollback()
    finally:
        cursor.close()
        cnxn.close()


def get_varchar_bucket(length):
    '''
    Function Used for generate_create_table_sql to round VARCHAR VALUES
    '''
    thresholds = [8, 16, 32, 64, 128, 255]
    for t in thresholds:
        if length <= t:
            return t
    return 255 

def TableColumnCleaner(df,clean_df=0):
    df = df.copy()

    # Auto-detect if not provided
    inferred_text = []
    inferred_number = []
    inferred_date = []
    failed_cols = []
    
    for col in df.columns:
        dtype = df[col].dtype

        if col == "MEMBERNBR":
            inferred_number.append(col)
        elif col == "ACCTNBR":
            inferred_number.append(col)
        elif col.lower().find('date')!=-1:
            inferred_date.append(col)
        elif col.lower().find('flag')!=-1:
            inferred_number.append(col)
        elif pd.api.types.is_string_dtype(dtype):
            inferred_text.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            inferred_number.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            inferred_date.append(col)
        else:
            failed_cols.append(col)
            
    print(f"Text Columns:{inferred_text}")
    print(f"Numeric Columns:{inferred_number}")
    print(f"Date Columns:{inferred_date}")
    print(f"Failed Columns:{failed_cols}")

    if clean_df==1:

        # Clean text
        for col in inferred_text:
            df[col] = df[col].fillna("")

        # Clean numeric
        for col in inferred_number:
            df[col] = df[col].fillna(0)

        # Clean date
        for col in inferred_date:
            try:
                ConvertDate(df, col, col, 1)
            except Exception as e:
                print(f"Failed to convert date column {col}: {e}")
                failed_cols.append(col)
    return df
