import pandas as pd

def input1(module):
    """
    Originally Named convert_dict_to_parameters
    Convert a .py file containing Dict into a Dataframe which can is used to populate Function Parameters

    Parameters:
        module (module): Py file containing Dict.

    Returns:
        Dataframe

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        import d_dictionaries
        temp = convertlisttoparameters(d_dictionaries)

    """
    rows = []
    for name, value in vars(module).items():
        if not name.startswith("_") and isinstance(value, dict):
            for k, v in value.items():
                rows.append({
                    "Folder": 'data_d_lists.py',
                    'Function':name,
                    "Parameters": k,
                    'Type':'dict',
                    "Definition": v
                })
    return pd.DataFrame(rows)


def input2(module):
    """
    originally named convert_list_to_parameters
    Convert a .py file containing Lists into a Dataframe which can is used to populate Function Parameters

    Parameters:
        module (module): Py file containing Lists.

    Returns:
        Dataframe

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        import d_lists
        temp = convertlisttoparameters(d_lists)

    """
    rows = []
    for name, value in vars(module).items():
        if not name.startswith("_") and isinstance(value, list):
            for i, item in enumerate(value):
                rows.append({
                    'Folder':'data_d_lists.py',
                    "Function": name,
                    "Parameters": item,
                    "Type":'str',
                    "Definition":''
                })
    return pd.DataFrame(rows)

def input3(module):
    """
    originally named convert_str_to_parameters
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
