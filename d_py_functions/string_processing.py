import pandas as pd
import numpy as np

def convert_str_to_parameters(module) -> pd.DataFrame:
    """
    Convert a .py file containing Strings into a Dataframe which can is used to populate Function Parameters

    Parameters:
        module (module): Py file containing Strings.

    Returns:
        Dataframe

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        import d_strings`aaax
        temp = convertlisttoparameters(d_strings)

    """
    rows = []
    for name, value in vars(module).items():
        if not name.startswith("_") and isinstance(value, str):
            rows.append({
                'Folder':'data_d_strings.py',
                'Function':"Reference String",
                "Parameters": name,
                'Type':'str',
                "Definition": value,  # DO NOT strip or clean
            })
    return pd.DataFrame(rows)
