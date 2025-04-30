import pandas as pd
import numpy as np

def TextClean(
    df,
    column_list,
    remove_newlines=False,
    strip_whitespace=False,
    normalize_whitespace=False,
    only_digits=False):
    """
    Applies selected text cleaning operations to specified DataFrame columns.

    Parameters:
        df (pd.DataFrame): DataFrame to clean.
        column_list (list): Columns to clean.
        
        Cleaning Options (all default to False):
            remove_newlines (bool): Remove \r and \n characters.
            strip_whitespace (bool): Trim leading and trailing whitespace.
            normalize_whitespace (bool): Collapse multiple spaces/tabs into one space.
            only_digits (bool): Remove all non-digit characters and convert to numeric.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    for col in column_list:
        if col not in df.columns:
            continue

        series = df[col].astype(str)

        if remove_newlines:
            series = series.str.replace(r'[\r\n]+', '', regex=True)
        if normalize_whitespace:
            series = series.str.replace(r'\s+', ' ', regex=True)
        if strip_whitespace:
            series = series.str.strip()
        if only_digits:
            series = series.str.replace(r'[^\d.]', '', regex=True)
            series = pd.to_numeric(series, errors='coerce')

        df[col] = series

    return df