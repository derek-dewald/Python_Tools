'''
module_word: df_creation_maint
module_definition: Repo for functions which Create, Materially Alter, Modify or Restructure DF.

'''

import numpy as np
import pandas as pd

def dataframe_to_dict(df):
    """
    Convert a DataFrame into a dictionary while preserving
    column names, index values, and row order.
    """
    return {
        "columns": df.columns.tolist(),
        "index": df.index.tolist(),
        "data": df.astype(object).where(pd.notna(df), None).values.tolist()
    }

def dict_to_dataframe(data):
    """
    Reconstruct a DataFrame from a dictionary.
    """
    return pd.DataFrame(
        data=data["data"],
        columns=data["columns"],
        index=data["index"]
    )
