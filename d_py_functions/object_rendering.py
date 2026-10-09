'''
module_word: object_rendering
module_definition: Functions that transform Python objects or structured data into human-readable, formatted representations or documents without changing the underlying meaning of the data. Rendering is the process of converting data or an object from its internal structure into a visual or formatted output suitable for viewing, interpretation, or sharing.

'''
import pandas as pd

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import textwrap

def txt_to_python(file_name,encoding="utf-8"):

    
    '''
    Definition: 
        Function Used to Import .txt or .py File into Python.
    Parameters: 
        file_name(str): Name of File, including path location for import
        encoding(str): Encoding to be applied by With Open call. Default is utf-8.

    Returns:
        Dataframe
    Date Created:
        3-Dec-25
    Date Last Modified:
        3-Dec-25
    Process:
        OS Folder Management
    Categorization:
        File Management
    usage:
        location = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/DFProcessing.py'
        file = TextFileImport(location)
    
    '''

    with open(file_name, "r", encoding=encoding) as file:
        data = file.read()
    
    return data

def export_formatted_df_to_single_xlsx(
    df,
    file_name,
    sheet_name="Sheet1",
    index=False,
    freeze_header=True,
    column_formats=None,  # dict: {col_name: format_type}
):
    '''
    Definition:
        Function Used to Simplify the required formating to a excel output from Python to Excel
    Parameters:
        df(dataframe): Any Dataframe
        file_name(str): Name of Excel Output File
        sheet_name(str): Name of Sheet to be generated in Excel
        index(bool): Include DF index in output as default action
        freeze_header(bool): Freeze Excel File Column Header as default action
        column_foratas(dict): Dictionary to format specific columns based on desired type, options include 
    Returns:
        TBD
    Date Created:
        28-Aug-26
    Date Last Modified:
        28-Aug-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        TBD
    Notes:
        None
    Required Functions:
        None
    
    
    '''
    
    if not file_name.endswith(".xlsx"):
            file_name += ".xlsx"
            
    with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=index, sheet_name=sheet_name)

        workbook = writer.book
        ws = writer.sheets[sheet_name]

        if freeze_header:
            ws.freeze_panes(1, 0)
        
        # ---- Base format (default for all cells) ----
        base_format = workbook.add_format({
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter'
        })

        # ---- Predefined special formats ----
        format_map = {
            "acct": workbook.add_format({'num_format': '0'}),
            "date": workbook.add_format({'num_format': 'yyyy-mm-dd'}),
            "money": workbook.add_format({'num_format': '$#,##0.00'}),
        }

        # ---- Auto-size columns ----
        for col_idx, col in enumerate(df.columns):
            max_len = min(
                80,
                max(df[col].astype(str).map(len).max(), len(col)) + 2
            )

            # Check if column has a special format
            if column_formats and col in column_formats:
                fmt_key = column_formats[col]
                fmt = format_map.get(fmt_key, base_format)
            else:
                fmt = base_format

            ws.set_column(col_idx, col_idx, max_len, fmt)

def visualize_dataframe_in_notebook(
    df,
    show_as_number=[],
    show_as_currency=[],
    show_as_currency_without_dec=[],
    show_as_percentage=[],
):
    
    '''
    
    Definition:
        Function which applies some modest formating to a Dataframe to increase visual astetic.

    Parameters:
        df (dataframe): Any DataFrame
        show_as_number(list): List of Columns which are to be displayed as format :,.2f
        show_as_currency(list): List of Columns which are to be displayed as format $:,.2f
        show_as_currency_without_dec(list): List of Columns which are to be displayed as format $:,.0f
        show_as_percentage(list): List of Columns which are to be displayed as format :.2f%        
        

    Returns:
        Object Type

    date_created: 27-Aug-26
    date_last_modified: 27-Aug-26
    classification:TBD
    sub_classification:TBD
    usage:
        visualize_dataframe_in_notebook(df)

    '''

    formats = {}
    
    for val in show_as_number:
        formats[val] = '{:,.2f}'
    
    for val in show_as_currency_without_dec:
        formats[val] = '${:,.0f}'

    for val in show_as_currency:
        formats[val] = '${:,.2f}'

    for val in show_as_percentage:
        formats[val] = '{:.2f}%'

    
    #df = df.replace(r'\$', r'\\$', regex=True)
 
    styled_df = (
        df.style
        .hide(axis='index')
        .format(formats,escape='html')
        .set_table_styles([
              {'selector': 'table',
               'props': [('border-collapse', 'collapse')]},
              {'selector': 'th',
               'props': [('border', '1px solid black'),
                         ('padding', '5px'),
                         ('text-align', 'center'),
                         ('vertical-align', 'middle'),
                         ('white-space', 'normal')]},
              {'selector': 'td',
               'props': [('border', '1px solid black'),
                         ('padding', '5px'),
                        ('text-align', 'center'),
                         ('vertical-align', 'middle'),
                         ('white-space', 'normal')]}
          ])
    )

    display(styled_df)

def print_dict(d, indent=0, width=100):
    '''
    Definition:
        Print Text from Dictionary into a structured output in Console.
    
    Parameters:
        d(dict): Any Dictionary.
        indent(float): Indent Required
    
    Returns:
        None. Prints Structured Text in Console.
    
    Date Created:
        28-Aug-26
        
    Date Last Modified:
        28-Aug-26
    
    Process:
        TBD
    
    Categorization:
        TBD
    
    Usage:
        TBD
    
    Notes:
        Developed for usage with Machine Learning Project. 
    
    Required Functions:
        None
    
    '''
    for key, value in d.items():
        prefix = "    " * indent

        if isinstance(value, dict):
            print(f"\n{prefix}{key}")
            print_dict(value, indent + 1, width)

        elif isinstance(value, list):
            print(f"\n{prefix}{key}")
            for item in value:
                text = str(item).strip()
                print(textwrap.fill(
                    text,
                    width=width,
                    initial_indent=f"{prefix}    - ",
                    subsequent_indent=f"{prefix}      "
                ))

        else:
            print(f"\n{prefix}{key}")
            text = str(value).strip()
            print(textwrap.fill(
                text,
                width=width,
                initial_indent=f"{prefix}    ",
                subsequent_indent=f"{prefix}    "
            ))

def generate_formated_excel_file(
    df,
    filename,
    sheet_name='Sheet1',
    wrap=True,
    align='center',
    valign='vcenter',
    currency_columns=None,
    balance_columns=None,
    date_columns=None,
    int_columns=None,
    decimal_columns=None
):

    # Default empty lists
    currency_columns = currency_columns or []
    balance_columns = balance_columns or []
    date_columns = date_columns or []
    int_columns = int_columns or []
    decimal_columns = decimal_columns or []

    with pd.ExcelWriter(
        filename,
        engine='xlsxwriter'
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name=sheet_name
        )

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # --------------------------------------------------
        # Formats
        # --------------------------------------------------

        general_format = workbook.add_format({
            'text_wrap': wrap,
            'align': align,
            'valign': valign
        })

        currency_format = workbook.add_format({
            'num_format': '$#,##0.00',
            'text_wrap': wrap,
            'align': align,
            'valign': valign
        })

        balance_format = workbook.add_format({
            'num_format': '$#,##0.00;[Red]-$#,##0.00',
            'text_wrap': wrap,
            'align': align,
            'valign': valign
        })

        date_format = workbook.add_format({
            'num_format': 'yyyy-mm-dd',
            'text_wrap': wrap,
            'align': align,
            'valign': valign
        })

        int_format = workbook.add_format({
            'num_format': '#,##0',
            'text_wrap': wrap,
            'align': align,
            'valign': valign
        })

        decimal_format = workbook.add_format({
            'num_format': '#,##0.00',
            'text_wrap': wrap,
            'align': align,
            'valign': valign
        })

        # --------------------------------------------------
        # Columns
        # --------------------------------------------------

        for idx, col in enumerate(df.columns):

            max_len = min(
                140,
                max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
            )

            # Default
            column_format = general_format

            # Overrides
            if col in currency_columns:
                column_format = currency_format

            elif col in balance_columns:
                column_format = balance_format

            elif col in date_columns:
                column_format = date_format

            elif col in int_columns:
                column_format = int_format

            elif col in decimal_columns:
                column_format = decimal_format

            worksheet.set_column(
                idx,
                idx,
                max_len,
                column_format
            )