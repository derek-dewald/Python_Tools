from IPython.display import display
import inspect
import pandas as pd

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
    else:
        with pd.option_context(
            'display.max_colwidth',None,
            'display.max_columns',None,
            'display.expand_frame_repr',False):
            display(df)

def ViewDecimals(df=None, decimal_points=2, global_=1):
    '''
    Function to Assist in Setting Decimal Display Precision for DataFrames.

    Parameters:
        df (dataframe, optional): DataFrame to display. Required if global_=0.
        decimal_points (int): Number of decimal places to show.
        global_ (int): Binary Flag, if 1 applies to whole session (Workbook),
                       if 0 applies only for displaying the single DataFrame.
    '''
    if global_ == 1:
        pd.set_option('display.precision', decimal_points)
        pd.set_option('display.float_format', f'{{:.{decimal_points}f}}'.format)
    else:
        if df is None:
            print("⚠️  DataFrame must be provided when global_=0")
            return
        with pd.option_context(
            'display.precision', decimal_points,
            'display.float_format', f'{{:.{decimal_points}f}}'.format
        ):
            display(df)


def InspectFunction(function_name):
    print(inspect.getsource(function_name))