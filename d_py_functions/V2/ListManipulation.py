import pandas as pd
import numpy as np

def CombineLists(list_,
                 combo=1,
                 r=2):
        
    '''
    Function to 
    
    
    Parameters:
        
    
    Return:
        
    '''
    
    items = sum([1 if isinstance(x,list) else 0 for x in list_])

    if items==0:
        if combo==1:
            return list(map(list,combinations(list_,r)))
        else:
            return list(map(list,permutations(list_,r)))
    
    return list(map(list,product(*list_)))