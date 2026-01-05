import pandas as pd
import numpy as np
import random

try:
    rng = np.random.default_rng(seed)
except:
    rng = np.random.default_rng()

def list_to_dataframe(list_,
                      column_name_list=None):
    '''
    Function to Simplify the creation of a Dictionary into a Dataframe into a single Command.

    Parameters:
        list_ (list): List of Values to be iterated into Row.
        column_name_list (list): Name of Column to be added, add as List. 

    Returns:
        Object Type

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        temp_df = list_to_dataframe(dict)
    '''
    if not column_name_list:
        return pd.DataFrame(list_)
    else:
        return pd.DataFrame(list_,columns=column_name_list)
    
def random_uniform_normalized_list(n, skew=1):

    """
    Function to create a list of RNG numbers for the purposes of creating a distribution.
    Values equal 1.

    Parameters:
        n(int): Number of Values to Return in list.
        skew(int): Skew to include in data, Values Greater than 0 will create 

    Returns:
        list

    date_created:29-Dec-25
    date_last_modified: 29-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        create_distribution_weight(5)


    """  
    # Generate random positive numbers
    raw = rng.random(n) ** skew  # apply skew
    weights = raw / raw.sum()    # normalize to sum to 1
    return [float(w) for w in weights]

def random_choice_from_uniform_list(total_records,
                                    name="Example",
                                    distinct_entities=0,
                                    list_distribution=[],
                                    return_value=None,
                                    skew=1):
    '''
    Create a random generate list from provided inputs. List is of length as defined in total records, the name of the records is defined in name. 
    The distribution of values is conditionally determined by either distinct entities, or the distribution as provided in list_distribution.

    Parameters:
        total_records(int): Number of records to be returned in list.
        name(str): Name of Random Records.
        distinct_entities(int): If populated, it will be used to generate a random distribution of defined values, also used as the number of reocrds
        list_distribution(list): Distribution to be used for random sampling.
        return_value(str): Default to None, and will return a list. Can input 'df' to return a dataframe
        skew(float): Skew to include in random distribution.
        
    Returns:
        list
        if return_value is 'df' then DataFrame

    date_created:29-Dec-25
    date_last_modified: 29-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        random_choice_from_uniform_list(unique_records=40,name='BRANCHNAME',LEGACY=[.5,.15,.3,.05])
    '''
    
    if (distinct_entities==0)&(list_distribution==[]):
        raise TypeError('User must select either Number of Distinct Entries or Provide a Distribution')
    
    if distinct_entities==0:
        distinct_entities = len(list_distribution)
        
    if len(list_distribution)==0:
        list_distribution = random_uniform_normalized_list(distinct_entities,skew=skew)
        
    name_list = [f"{name} {x+1}" for x in range(0,distinct_entities)]
    
    final_list = [random.choices(name_list,weights=list_distribution)[0] for x in range(0,total_records)]
    
    if return_value=='df':
        return pd.DataFrame(final_list,columns=[name])
    else:
        return final_list