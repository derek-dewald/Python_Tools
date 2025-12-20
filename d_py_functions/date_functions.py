import pandas as pd
import datetime
import numpy as np

def historical_month_end_list(end_dt=None,
                              total_months=12,
                              sort_ascending=True,
                              format_="datetime"):
    
    '''
    Function which Creates a list of Month End Dates. 

    Parameters:
        end_dt (datetime): Last date in period. Remember it is month end, so technically it will not return this month unless it is a month end date
        total_months (int): Total number of records in list, by default it will be 12.
        sort_ascending(bool): True/False to Determine whether end_dt will be the First, or Last value in list. By default (True) it is the Last
        format_(str): Optional Argument to change the format of the values in list, 5 options incude (if valid option not selected, datetime returned).
        Valid options include: '%d-%b-%y', '%b-%y', timestamp, dt_date.

    
    Returns:
        List

    date_created:19-Dec-25
    date_last_modified: 19-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        df = historical_month_end_list()
        
    '''

    if not end_dt:
        end_dt = datetime.datetime.now()

    month_list = pd.date_range(end=end_dt,periods=total_months,freq='ME').normalize()
        
    if format_ =='%d-%b-%y':
        month_list = [x.strftime(format_) for x in month_list]
    
    elif format_ =='%b-%y':
        month_list = [x.strftime(format_) for x in month_list]

    elif format_ == 'timestamp':
        pass

    elif format_ == 'dt_date':
        month_list = [x.to_pydatetime().date() for x in month_list]
    
    else:
        month_list = [x.to_pydatetime() for x in month_list]
        
    if sort_ascending:
        month_list = month_list[::-1]
    
    return list(month_list)