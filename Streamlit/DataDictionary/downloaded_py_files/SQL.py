def ConvertListtoSQLText(list_):
    '''
    Function to take a returned list from Python and convert it into SQL format for Quert.


    Parameters:
        List_ (list): List of str objects to be included in SQL text
    
    Returns:
        Str
    
    
    
    
    '''

    return ', '.join(f"'{str(x)}'" for x in list_)