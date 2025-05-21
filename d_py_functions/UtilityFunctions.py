from IPython.display import display
import inspect
import pandas as pd
import random

def ViewDF(df=None,update_decimal=1):
    '''
    Function to Assist in the Viewing of a Dataframe in a User Friendly Manner, showing all rows, columns and Numbers.

    Parameters:
        df (dataframe)
        global (int): Binary Flag, if 1 it applies the preferences to the Workbook, if 0 it displays the single dataframe, not exporting the perferences to the workbook
    
    
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


def InspectFunction(function_name):
    print(inspect.getsource(function_name))

def password_generator(minimum=8,maximum=10):
    
    '''
    Purpose: Create a Randomly Generated Password of a predetermined number of CHaracters. 
    
    Input: 
    
    Minimum: Number of Minimum Characters. 
    Maximum: Number of Maximum Characters.
    
    Default:
    Minimum: 8 Characters
    Maximum: 10 Characters.
    
    Notes:
    First Character is always a Letter which is captilized.
    The last 1-3 letters are special Characters, randomly assinged. 
    
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

    #print(len(password))
    return password
