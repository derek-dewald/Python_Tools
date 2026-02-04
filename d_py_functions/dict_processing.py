from collections.abc import Mapping, Sequence
import pandas as pd
import numpy as np

def dict_to_dataframe(dict_,
                    key_name="KEY",
                    value_name='VALUE'):
    '''
    Function to Simplify the creation of a Dictionary into a Dataframe into a single Command.

    Parameters:
        dict_(dict)
        key_name(str): Name of Column which will include values from Key (Default is KEY)
        value_name(str): Name of Column which will include values from Values (Default is Value)

    Returns:
        Dataframe

    date_created:4-Dec-25
    date_last_modified: 4-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        temp_df = dict_to_dataframe(dict)
    '''
    return pd.DataFrame.from_dict(dict_, orient='index', columns=[value_name]).reset_index().rename(columns={'index': key_name})


def flatten_clean_dict(dict_,
                       index_name='INDEX',
                       clean=True,
                       apply_new_lvl=False,
                       high_low_list_fix=True):
    
    '''
    Function which takes a Nested Dictionary (Dictionary, which references dictionary, and converts it into a DataFrame, works best when
    Dictionary ultimate Values are List.

    Parameters:
        dict_(dict): Nested Dictionary
        index_name(str): Default Name of Column to be applied to First Level Dictionary in DataFrame.
        clean(bool): Used to convert a single flat DF to a Matrix, which each new column
        apply_new_level(bool): Optional Argument to support the application to a Single Dictionary
        high_low_list_fix(bool): Optional Arguement, used for resolution of list [0,1] dict extraction.
        
    Returns:
        DataFrame

    date_created:30-Dec-25
    date_last_modified: 30-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        from synthetic_member import mbr_profile_dict
        flatten_clean_dict (mbr_profile_dict)
    
    '''

    if apply_new_lvl:
        dict_ = {'ADDED_LEVEL':dict_}
    
    from collections.abc import Mapping, Sequence

    def flatten_dict(obj, parent_key="", sep="."):
        """
        Recursively flattens a nested dict/list into a flat dict of {column_name: value}.
        - Dict keys are joined with `sep`.
        - List elements are addressed with [index].
        - Scalars become leaf values.

        Examples:
          {"a": {"b": 1}} -> {"a.b": 1}
          {"a": [ {"x": 1}, {"x": 2} ]} -> {"a[0].x": 1, "a[1].x": 2}
        """
        items = {}

        # Dict-like
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                items.update(flatten_dict(v, new_key, sep=sep))

        # List/tuple-like (but not str/bytes)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                items.update(flatten_dict(v, new_key, sep=sep))

        # Scalar (leaf)
        else:
            items[parent_key] = obj

        return items


    def flatten_to_df(data, sep="."):
        """
        Accepts:
          - a single dict -> returns a 1-row DataFrame with one column per leaf.
          - a list/iterable of dicts -> returns a DataFrame with one row per item.
          - a mix (list containing dicts and scalars) -> flattens each element.

        Column names reflect the nested paths using `sep` and [index] for lists.
        """
        # Single dict -> one row
        if isinstance(data, Mapping):
            flat = flatten_dict(data, sep=sep)
            return pd.DataFrame([flat])

        # Iterable -> one row per element
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            rows = []
            for elem in data:
                if isinstance(elem, Mapping) or (isinstance(elem, Sequence) and not isinstance(elem, (str, bytes, bytearray))):
                    rows.append(flatten_dict(elem, sep=sep))
                else:
                    # scalar element becomes a single column named by its position
                    rows.append({ "value": elem })
            return pd.DataFrame(rows)

        # Scalar -> single column/value
        return pd.DataFrame([{"value": data}])
    
    flat_df = flatten_to_df(dict_)
    
    if not clean:
        return flat_df
    
    else:
        final_df = pd.DataFrame()
        for classification in dict_.keys():
            # Select Columns with Classification in it:
            name = f'{classification}.'
            cols = [c for c in flat_df.columns if c.startswith(name)]
            temp_df = flat_df[cols].copy()
            temp_df = temp_df.rename(columns={x:x.replace(name,"") for x in cols})
            temp_df.rename(index={0:classification},inplace=True)
            final_df = pd.concat([final_df,temp_df])
        final_df = final_df.reset_index().rename(columns={'index':index_name})
        
        # Update for Usage of Bracketing for High/ Low
        if high_low_list_fix:
             final_df.rename(columns={x:x.replace('[0]','_low').replace('[1]','_high') for x in final_df.columns},inplace=True)
        
        #final_df = final_df.reset_index().rename(columns={'index':index_name})
        return final_df
