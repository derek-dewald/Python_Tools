from IPython.display import display
import inspect
import pandas as pd
import random

def inspect_function(function_name):
    '''
    Function which Reads the Document String of a Python Function

    Parameters:
        function_name(str): Name of Function which is Loaded into Current Python Session Memory

    Returns:
        Object Type

    date_created:15-Dec-25
    date_last_modified: 15-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        from utility_functions import InspectFunction
        InspectFunction(InspectFunction)
    
    '''
    print(inspect.getsource(function_name))


def view_df(df=None,update_decimal=2):
    '''
    Function which helps to set default Visualization View in Pandads. Can be applied to a dataframe specifically or the current Session

    Parameters:
        df(dataframe): Any Dataframe
        update_decimal (int): Number of Default Decimal Places to Show in Notebook

    Returns:
        Object Type

    date_created:15-Dec-25
    date_last_modified: 15-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        ViewDf(df)

    '''
    if update_decimal:
        pd.options.display.float_format = '{:,.2f}'.format

    if df:
      with pd.option_context(
            'display.max_colwidth',None,
            'display.max_columns',None,
            'display.expand_frame_repr',False):
            display(df)
    else:
        pd.set_option('display.max_colwidth',None)
        pd.set_option('display.max_columns',None,)
        pd.set_option('display.expand_frame_repr',False)


def password_generator(minimum=8,maximum=10):
    
    '''

    Function to Create a Randomly Generated Password.

    Parameters:
        minimum(int): Minimum Number of Required Characters
        maximum(int): Maximum Number of Required Characters

    Returns:
        Object Type

    date_created:15-Dec-25
    date_last_modified: 15-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        password_generator()
 
    '''

    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    char = ['!','@','#','$','%','&','*',')',"("]
    num = ['1','2','3','4','5','6','7','8','9','0']
    
    # select a random number between 8 and 12
    total_char  = random.randint(minimum,maximum)
    #print(total_char)

    # Select a random Number of Special Characters, always at the End for simplicity
    special_char = random.randint(1,3)
        
    alpha_num = total_char - special_char 
    
    # ALways start with a Capital
    password = random.choice(letter).upper()
    
    for i in range(0,alpha_num-1):
        temp = random.choice([letter,num])
        password += random.choice(temp)
    
    for i in range(0,special_char):
        password += random.choice(char)

    p_list = list(password)
    random.shuffle(p_list)  # shuffle in place
    return "".join(p_list)


def view_entire_df_column(df,index_=0):
    
    '''
    Function to Support with the visualization of a specific dataframe column within Jupyter Notebook. Prints the Entire column (Transposed).

    Parameters:
        df(df): Dataframe
        index_ (int): Index Row Position of record you would like to be printed in Notebook, default is 0.

    Returns:
        None

    date_created:01-Feb-26
    date_last_modified: 01-Feb-26
    classification:TBD
    sub_classification:TBD
    usage:
        view_entire_df_column(df)
    
    '''
    with pd.option_context('display.max_columns',None,
                           'display.max_rows',None,
                           'display.width',None):
        
        display(df.iloc[index_])

