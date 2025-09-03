from IPython.display import display
import inspect
import pandas as pd
import numpy as np
import math

def ViewDF(df,global_=1):
    '''
    Function to Assist in the Viewing of a Dataframe in a User Friendly Manner, showing all rows, columns and Numbers.

    Parameters:
        df (dataframe)
        global (int): Binary Flag, if 1 it applies the preferences to the Workbook, if 0 it displays the single dataframe, not exporting the perferences to the workbook
    
    
    '''

    if global_ ==1:
        pd.set_option('display.max_colwidth',None)
        pd.set_option('display.max_columns',None,)
        pd.set_option('display.expand_frame_repr',False)
        pd.set_option('display.float_format','{:.2f}'.format)
    else:
        with pd.option_context(
            'display.max_colwidth',None,
            'display.max_columns',None,
            'display.float_format', '{:.2f}'.format,
            'display.expand_frame_repr',False):
            display(df)


def InspectFunction(function_name):
    print(inspect.getsource(function_name))

def TextClean(
    df,
    column_list,
    lower_case=False,
    remove_newlines=False,
    strip_whitespace=False,
    normalize_whitespace=False,
    remove_punctuation=False,
    only_digits=False,
    only_letters=False):
    
    
    """
    Applies selected text cleaning operations to specified DataFrame columns.

    Parameters:
        df (pd.DataFrame): DataFrame to clean.
        column_list (list): Columns to clean.
        
        Cleaning Options (all default to False):
            remove_newlines (bool): Remove \r and \n characters.
            strip_whitespace (bool): Trim leading and trailing whitespace.
            normalize_whitespace (bool): Collapse multiple spaces/tabs into one space.
            remove_punctuation (bool): Remove punctuation characters.
            only_digits (bool): Remove all non-digit characters and convert to numeric.
            only_letters (bool): Remove all non-letter characters and convert to string.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    for col in column_list:
        if col not in df.columns:
            continue

        series = df[col].astype(str)
        
        if lower_case:
            series = series.str.lower()
        if remove_newlines:
            series = series.str.replace(r'[\r\n]+', '', regex=True)
        if normalize_whitespace:
            series = series.str.replace(r'\s+', ' ', regex=True)
        if strip_whitespace:
            series = series.str.strip()
        if only_digits:
            series = series.str.replace(r'[^\d.]', '', regex=True)
            series = pd.to_numeric(series, errors='coerce')
        if only_letters:
            series = series.str.replace(r'[^a-zA-Z]', '', regex=True)
        if remove_punctuation:
            series = series.str.translate(str.maketrans('', '', string.punctuation))

        df[col] = series

    return df

def calculate_remaining_payments(loan_amount, annual_interest_rate, payment_amount, payments_per_year):
    """
    Calculate the number of remaining loan payments.

    Parameters:
    - loan_amount: float, the remaining principal
    - annual_interest_rate: float, annual interest rate as a decimal (e.g., 0.06 for 6%)
    - payment_amount: float, payment made each period
    - payments_per_year: int, number of payments per year (e.g., 12 for monthly)

    Returns:
    - int, number of remaining payments
    """
    try:
        rate_per_period = annual_interest_rate / payments_per_year
        if payment_amount <= rate_per_period * loan_amount:
            #raise ValueError("Payment amount too low to cover interest. Loan will never be repaid.")
            return 0
    except:
        return 0
    
    try:
        numerator = math.log(payment_amount / (payment_amount - rate_per_period * loan_amount))
        denominator = math.log(1 + rate_per_period)
        return math.ceil(numerator / denominator)/(payments_per_year/12)
    except:
        return 0


def ViewEntireRecord(df,index_=0):
    with pd.option_context('display.max_columns',None,
                          'display.max_rows',None,
                          'display.width',None):
        display(df.iloc[index_])


def PauseProcess(time,time_check,text):
    if time>time_check:
        print(f'Elapsed time to process {text}:{time:,.2f}')
        resume_ = input(f'Would you like to continue? 1 Column is taking more than {time_check} Seconds. (Yes/Y)')
        if (resume_.lower()!='yes')|(resume_.lower()!='y'):
            time_check += 5
    return time_check




