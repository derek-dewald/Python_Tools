from IPython.display import display

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