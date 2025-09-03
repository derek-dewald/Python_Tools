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
                      new_column_name="ORG_ADD_DATE", 
                      og_format='month_str', 
                      new_format='month_dt'):
    """
    Convert date columns in a DataFrame to a new format.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to convert.
        new_column_name (str): Name of the new column with converted dates.
        og_format (str): Original date format ('month_str', 'month', 'month_dt').
        new_format (str): Desired new date format ('month_str', 'month', 'month_dt').

    Returns:
        pd.DataFrame: DataFrame with the new date column.
    """
    df = df.copy()

    # Define date format mappings
    format_mappings = {
        'month_str': '%b-%y',
        'month': '%d-%b-%y',
        'month_dt': '%Y-%m-%d'
    }

    # Check if the original format is valid
    if og_format not in format_mappings:
        raise ValueError(f"Invalid original format: {og_format}. Choose from {list(format_mappings.keys())}.")

    # Convert the original column to datetime
    df[column_name] = pd.to_datetime(df[column_name], format=format_mappings[og_format], errors='coerce')

    # Handle invalid dates
    if df[column_name].isnull().any():
        print("Warning: Some dates could not be converted and will be set to NaT.")

    # Convert to the new format if necessary
    if new_format == 'month_str':
        df[new_column_name] = df[column_name].dt.strftime('%b-%y')
    elif new_format == 'month':
        df[new_column_name] = df[column_name].dt.strftime('%d-%b-%y')
    elif new_format == 'month_dt':
        df[new_column_name] = df[column_name]
    else:
        raise ValueError(f"Invalid new format: {new_format}. Choose from {list(format_mappings.keys())}.")

    return df

def generate_day_list(start_date=datetime.datetime(2025,1,1),end_date= None):
    if end_date is None:
        end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    
    # Generate the list of dates
    date_list = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    return date_list

def FIRST_DAY_OF_MONTH(x):
    try:
        x = pd.to_datetime(x,errors='coerce').replace(day=1)
    except:
        pass
    return x.date()

def LAST_DAY_PREVIOUS_MONTH(date_dt=None,
                            return_value='month_dt'):
    if date_dt==None:
        date_dt = datetime.datetime.now()
    
    new_date = date_dt.replace(day=1,hour=0,minute=0,second=0,microsecond=0)-datetime.timedelta(days=1)
    
    if return_value=='month_dt':
        return new_date
    
    elif return_value == 'month_str':
        return new_date.strftime('%d-%b-%y')


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

def MonthSelector(month_int=1,
                  return_value='month_dt',
                  end_date_yesterday=1):


    '''
    Function to return the Date based on Binary Int. 1 Represents month end last month, 0 Today (can change to yesterday with default binary flag)


    Parameters:
        month_int (int): Representative of the desired month 0 (today), 1 (last month end), 2 (2 Months, If in FebX it would be 31DecX)
        end_date_yesterday (int): OPtional function to overwrite today to be yesterday when 0 for SQL db queries.

    Return:

    '''

    if (month_int==0) & (end_date_yesterday==1):
        value_ = (datetime.datetime.now()-datetime.timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0)
        if return_value=='month_dt':
            return value_
        elif return_value =='month_str':
            return value_.strftime('%b-%y')
        elif return_value =='month':
            return value_.strftime('%d-%b-%y')
    else:
        return CreateMonthList(month_int=month_int,months=1,return_value=return_value)[0].replace(hour=0,minute=0,second=0,microsecond=0)
        

def MonthIntManualConvert(value):
    '''
    Function to support Integration of Month Int and Datetime.
    Written because several legacy functions utilized Month Int, which is no longer the practice, but 
    do not want to re-write the enitre function.
    
    
    
    '''
    
    if isinstance(value,int):
        return CreateMonthList(months=value+1)[value]
    else:
        try:
            return CreateMonthList().index(value)
        except:
            return CreateMonthList(months=100).index(value)


def IsMonthEnd(dt_date):
    '''
    Function to Take a Date and Determine if it is the last day of the month, if Yes, then Return True, Else False
    Function Used to determine when automating Table Creation whether Table should be Stored in HIST aswell as Current.
    
    
    '''
    try:
        tomorrow = dt_date + datetime.timedelta(days=1)
    except:
        tomorrow = dt_date.date() + datetime.timedelta(days=1)
        
    if tomorrow.month != dt_date.month:
        return True
    else:
        return False


def DaysSince(df,column_name,new_column_name='DAYS_SINCE',date_dt=None):
    
    '''
    Function which Calculates the number of days between a DataFrame Column and a selected Static Date.
    
    Used as a input to LengthOfTimeSegment
    
    Parameters:
        df (dataFrame)
        column_name (str): Name of Column which stores a Date Value
        new_column_name (str): Name of New Column which will be returned, if blank DAYS_SINCE
        date_dt (datetime.datetime): Value which will be compared against, as a default it is the time the function
        was run
        
    Returns:
        Dataframe (With New Column new_column_name)
    
    Date Created: August 22,2025
    Date Last Modified: 
    
    '''
    
    if not date_dt:
        date_dt = datetime.datetime.now()
        
    df[f"CLEAN_DATE"] = pd.to_datetime(df[column_name],errors='coerce')
    date = pd.Timestamp(date_dt)
    df[new_column_name] = (date - df['CLEAN_DATE']).dt.days

    return df.drop("CLEAN_DATE",axis=1)    
    
def LengthOfTimeSegment(df,
                        column_name,
                        new_column_name='MEMBER_DURATION',
                        date_dt=None,
                        dict_=None,
                        keep_all=None):
    
    '''
    Function which segments the amount of time between a Date and a Static Reference Point. Segments to be provided
    From a Dictionary. Default Dictionary is DataMaps.duration_dict.
    
    Parameters:
        df (dataFrame)
        column_name (str): Name of Column which stores a Date Value
        new_column_name (str): Name of New Column which will be returned, if blank DAYS_SINCE
        date_dt (datetime.datetime): Value which will be compared against, as a default it is the time the function
        was run
        dict_ (dict): Dictionary used to Calculate Segment. Default stored in DataMaps.duration_dict.
        keep_all (Bool): True/ False to determine whether user would like to keep values created through processing,
        primarily for DQ and Troubleshooting, but sometime could be used for other purposes. 
    
    
    Returns:
        Dataframe (With New Column new_column_name)
    
    Date Created: August 22,2025
    Date Last Modified: 
    
    
    '''
    
    if not date_dt:
        date_dt = datetime.datetime.now()
    
    temp_df = DaysSince(df,column_name)

    
    if not dict_:
        from DataMaps import duration_dict
        dict_ = duration_dict
        
    bins = sorted(dict_.items())
    edges,labels = zip(*bins)
    
    df[new_column_name] = pd.cut(temp_df['DAYS_SINCE'],
                                 bins=[-np.inf]+ list(edges),
                                 labels=labels,
                                 right=True)
    if keep_all:
        return df
    else:
        return df.drop(['CLEAN_DATE','DAYS_SINCE'],axis=1)






