import pandas as pd
import numpy as np

def ConvertDicttoDF(dict_,
                    key_name="KEY",
                    value_name='VALUE'):
    
    return pd.DataFrame.from_dict(dict_, orient='index', columns=[value_name]).reset_index().rename(columns={'index': key_name})


def MapDicttoDF(df,
                dict_,
                column_name,
                new_column_name):
    '''
    
    
    
    '''
    
    bins = sorted(dict_.items())
    edges,labels = zip(*bins)
    
    df[new_column_name] = pd.cut(df[column_name],
                                 bins=[-np.inf]+ list(edges),
                                 labels=labels,
                                 right=True)
