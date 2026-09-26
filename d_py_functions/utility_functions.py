'''
module_word: utility_functions
module_definition: Location for functions which do not fall within the scope of any other classification and are Simple, High Generalized, General Purpose functions.

'''
import pandas as pd
import numpy as np
import inspect

def InspectFunction(function_name):
    print(inspect.getsource(function_name))