import pandas as pd
import numpy as np
import datetime
import calendar

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
            


def CalculateDaysFromtoday(df,
                           column_name,
                           new_column_name='DaysSinceDate',
                           month_int=0):
    
    '''
    Function to Simply Calculate the Number of Days in Years between 2 points in time. 
    First Date is populated via Datafarme, Second date is Calculated by Refence of Month_int.
    
    Parameters:
        df (dataframe)
        column_name (str): Name of Column in Dataframe. Should be datetime format before utilization.
        new_column_name (str): Name of Column which will be created (Defaults to DaysSinceDate
        month_int: Reference point of time, Please refer to MonthSelector for additional insight as to how this is 
        calculated.
        
    Returns:
        Dataframe, with new column. 
    
    
    '''
    
    reference_date = MonthSelector(month_int)
    
    df[new_column_name] = (reference_date-df[column_name])/datetime.timedelta(365.25)


def MonthSelector(month_int=1,
                  return_value='month_dt',
                  end_date_yesterday=1):
    
        
    '''
    Function to 
    
    
    Parameters:
        
    
    Return:
        
    '''
    
    return CreateMonthList(month_int=month_int,months=1,return_value=return_value,end_date_yesterday=end_date_yesterday)[0]

def CreateMonthList(month_int=0,
                    months=36,
                    end_date_yesterday=1,
                    sort_ascending=0,
                    return_value="month_dt"):
    
    '''
    Function to 
    
    
    Parameters:
        
    
    Return:
        
    '''
    
    
    
    final_months = month_int + months
    month_list = pd.date_range(end=pd.Timestamp.today(),periods=final_months,freq='M').normalize()
    month_list = month_list.union([pd.Timestamp.today().normalize()-datetime.timedelta(days=end_date_yesterday)])
    
    if return_value =='month_str':
        month_list = [x.strftime('%b-%y') for x in month_list]
    elif return_value =='month':
        month_list = [x.strftime('%d-%b-%y') for x in month_list]
    elif return_value =='month_dt':
        month_list = [x.to_pydatetime() for x in month_list]
    if sort_ascending==0:
        month_list = month_list[::-1]
    
    return list(month_list[month_int:final_months])




def MonthEndDate(x,og_format):
    
        
    '''
    Function to take a Random Date in a specific format and convert it into dt. Primarily used with Str Jan-25.
    Included import function in code as this is a rarely used file and preventing the need to import in library of commonly used functions.
    
    
    Parameters:
        
    
    Return:
        
    '''

    x = datetime.datetime.strptime(x,og_format)
    day = calendar.monthrange(x.year,x.month)[1]
    return datetime.datetime(x.year,x.month,day)
    

def ManualConvertDate(df,
                    column_name,
                    new_column_name="",
                    og_format='month_str',
                    new_format='month_dt'):
    
    '''
    Function to Take a Existing Dataframe Data Column and Convert it to a New Column
    
    Parameters:
        df:
        column_name:
        og_format:
        new_format:
            
    Returns:
        Dataframe with New Column (or if not included, then Updated Format of column_name)
        
    '''
    
    if new_column_name=='':
        new_column_name = column_name
        
    og_list = df[column_name].unique().tolist()
    
    if og_format =='month_str':
        if new_format=='month_dt':
            new_list = [datetime.datetime.strptime(x,"%b-%y") for x in og_list]
        elif new_format =='month':
            new_list = [MonthEndDate(x,'%b-%y').strftime('%d-%b-%y') for x in og_list]
        else:
            new_list = og_list
    
    elif og_format =='month':
        if new_format=='month_dt':
            new_list = [MonthEndDate(x,"%d-%b-%y") for x in og_list]
        elif new_format =='month_str':
            new_list = [datetime.datetime.strptime(x,"%d-%b-%y").strftime('%b-%y') for x in og_list]
        else:
            new_list = og_list
            
    elif og_format =='month_dt':
        og_list = pd.to_datetime(df[column_name].unique())
        if new_format=='month':
            new_list = [datetime.datetime.strftime(x,"%d-%b-%y") for x in og_list]
        elif new_format =='month_str':
            new_list = [datetime.datetime.strftime(x,"%b-%y") for x in og_list]
        else:
            new_list = new_list = df[column_name].unique().tolist()
        
    temp_df = pd.DataFrame(new_list,columns=[new_format]).merge(pd.DataFrame(og_list,columns=[og_format]),left_index=True,right_index=True,how='left').rename(columns={new:new_column_name})
    
    return df.merge(temp_df,on=column_name,how='left')
