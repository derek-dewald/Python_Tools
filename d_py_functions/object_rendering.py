'''
module_word: object_rendering
module_definition: Functions that transform Python objects or structured data into human-readable, formatted representations or documents without changing the underlying meaning of the data. Rendering is the process of converting data or an object from its internal structure into a visual or formatted output suitable for viewing, interpretation, or sharing.

'''


COMMON_TABLE_STYLE = [
    {
        'selector': 'table',
        'props': [
            ('border-collapse', 'collapse'),
            ('border', '1px solid #D3D3D3'),
            ('font-family', 'Arial, sans-serif'),
            ('font-size', '10pt')
        ]
    },
    {
        'selector': 'thead th',
        'props': [
            ('background-color', '#003366'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('vertical-align', 'middle'),
            ('padding', '9px 14px'),
            ('border', '1px solid #D3D3D3')
        ]
    },
    {
        'selector': 'td',
        'props': [
            ('text-align', 'center'),
            ('vertical-align', 'middle'),
            ('padding', '7px 14px'),
            ('border', '1px solid #D3D3D3'),
            ('font-variant-numeric', 'tabular-nums')
        ]
    },
    {
        'selector': 'tbody tr:nth-child(even)',
        'props': [
            ('background-color', '#F5F8FC')
        ]
    },
    {
        'selector': 'tbody tr:hover',
        'props': [
            ('background-color', '#EAF2FA')
        ]
    },
    {
        'selector': '.row_heading',
        'props': [
            ('text-align', 'center'),
            ('vertical-align', 'middle'),
            ('font-weight', '500'),
            ('min-width', '150px'),
            ('max-width', '200px'),
            ('white-space', 'normal'),
            ('overflow-wrap', 'break-word'),
            ('padding', '7px 14px'),
            ('border', '1px solid #D3D3D3')
        ]
    }
]


from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import seaborn as sns


from IPython.display import HTML, display, Markdown
import html

import pandas as pd
import numpy as np

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

def generate_data_header_box(
    value_dict,
    leading_text=None,
    trailing_text=None,
    width=1200):
    
    """

Definition:
    Function Used to Create a Psedu-Dashboard header. Used to quickly visualize information in header box with leading and trailing text.
    
Parameters:
    value_dict(dict): Dictionary in the format  dict_ = {'item1': {'Text': 'Total Transactions', 'Value': f"{20000:,.0f}"}}
    leading_text(str): HTML formatted Text. Displayed before Visual
    trailing_text(str): HTML formatted Text. Displayed After Visual
    
Returns:
    Rendered Output
    
Date Created:
    21-Oct-26
    
Date Last Modified:
    21-Oct-26

Process:
    TBD

Categorization:
    TBD

Usage:
    generate_data_header_box(example,leading_text,trailing_text)

Notes:
    None
    
Required Functions:
    html
    Markdown
    display
    
    """

    cards = ""

    for item in value_dict.values():

        value = item["Value"]
        label = html.escape(str(item["Text"]))

        formatted = (
            f"{value:,.0f}"
            if isinstance(value, (int, float))
            else str(value)
        )

        cards += f"""
        <div class="kpi-card">
            <div class="kpi-value">{html.escape(formatted)}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """

    output = f"""
    <div style="
        width: {width}px;
        max-width: 100%;
        display: flex;
        gap: 16px;
        padding: 20px;
        background: #E9EEF5;
        border: 3px solid #475569;
        border-radius: 16px;
        box-sizing: border-box;
    ">
        <style>
            .kpi-card {{
                flex: 1;
                min-width: 0;
                padding: 24px 12px;
                text-align: center;
                background: white;
                border: 2px solid #64748B;
                border-radius: 12px;
                box-shadow: 4px 4px 8px rgba(0,0,0,0.25);
            }}
            .kpi-value {{
                font-size: 30px;
                font-weight: 700;
                color: #1E293B;
            }}
            .kpi-label {{
                font-size: 14px;
                color: #475569;
                margin-top: 10px;
            }}
        </style>
        {cards}
    </div>
    """

    
    
    if leading_text:
        display(Markdown(leading_text))
        
    display(HTML(output))
    
    if trailing_text:
        display(Markdown(trailing_text))

def generate_metric_val_perc_pivot_heatmap(
    df, 
    index, 
    columns, 
    numerator, 
    denominator,
    leading_text=None,
    trailing_text=None
):

    """
Definition:
    Function to create 2 Heatmaps, which Display the Total Value of 1 Column, and the Relative Percentage. Used to articulate how values are distributed absolutely and relatively within segmentation in a bi-variable relationship.

Parameters:
    df(dataframe):
    index(str): Column Name to be used as heatmap Index.
    columns(str): Column Name to be used as heatmap Column
    numerator(str): Column Name of value to be Used in Primary Heat Map
    denomincator(str): Column Name of value to divide numerator by in order to Calculate Percentage
    leading_text(str): HTML Formatted text to proceed Visualization
    trailing_text(str): HTML Formatted text to follow Visualization

Returns:
    Rendered Object
Date Created:
    21-Oct-26
Date Last Modified:
    21-Oct-26
Process:
    TBD
Categorization:
    TBD
Usage:
    generate_metric_val_perc_pivot_heatmap(
        df2,
        "FINAL_TRANSACTION",
        "TRANSACTION_VAL_SEGMENT",
        numerator="FINAL_DECISION_COUNT",
        denominator='Total',
        leading_text='# Leading Test',
        trailing_text='# Trailing Test'
        )
Notes:
    None
Required Functions:
    None
    
    """

    
    if leading_text:
        display(Markdown(leading_text))  
    
    # Aggregate numerator and denominator
    aggregated = df.pivot_table(
        index=index,
        columns=columns,
        values=[numerator, denominator],
        aggfunc="sum",
        fill_value=0
    )

    numerator_table = aggregated[numerator]
    denominator_table = aggregated[denominator]

    # Calculate percentages from aggregated values
    percentage_table = (
        numerator_table
        .div(denominator_table.where(denominator_table.ne(0)))
        .mul(100)
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 8), sharey=True, dpi=200
    )

    # Numerator counts
    sns.heatmap(
        numerator_table,
        annot=True,
        fmt=",.0f",
        cmap="RdBu_r",
        ax=ax1,
        norm=LogNorm(),
        mask=numerator_table.eq(0)
    )

    ax1.set_title("By Count")

    # Percentage of denominator
    sns.heatmap(
        percentage_table,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        ax=ax2,
        mask=percentage_table.isna()
    )

    ax2.set_title("By Percentage")

    for axis in (ax1, ax2):
        axis.set_xlabel(columns.replace("_", " ").title())
        axis.set_ylabel(index.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.show()

    if trailing_text:
        display(Markdown(trailing_text))  


def visualize_dataframe_in_notebook(
    df,
    show_as_int=[],
    show_as_number=[],
    show_as_currency=[],
    show_as_currency_without_dec=[],
    show_as_percentage=[],
    normalize_percentage=False,
    return_value='render',
    hide_index=True
):
    
    '''
    
    Definition:
        Function which applies some modest formating to a Dataframe to increase visual astetic.

    Parameters:
        df (dataframe): Any DataFrame
        show_as_int(list): List of Columns which are to be displayed as format :,.0f
        show_as_number(list): List of Columns which are to be displayed as format :,.2f
        show_as_currency(list): List of Columns which are to be displayed as format $:,.2f
        show_as_currency_without_dec(list): List of Columns which are to be displayed as format $:,.0f
        show_as_percentage(list): List of Columns which are to be displayed as format :.2f%        
        normalize_percentage (binary): Optional Argument to multiple Percentage Column by 100, to take ensure percentages display consistent visually.

    Returns:
        Object Type

    date_created: 27-Aug-26
    date_last_modified: 27-Aug-26
    classification:TBD
    sub_classification:TBD
    usage:
        visualize_dataframe_in_notebook(df)

    '''
    df = df.copy()

    formats = {}
    
    for val in show_as_number:
        formats[val] = '{:,.2f}'

    for val in show_as_int:
        formats[val] = '{:,.0f}'

    for val in show_as_currency_without_dec:
        formats[val] = '${:,.0f}'

    for val in show_as_currency:
        formats[val] = '${:,.2f}'

    for val in show_as_percentage:
        if normalize_percentage:
            df[val] = df[val]*100
        formats[val] = '{:.2f}%'

    
    #df = df.replace(r'\$', r'\\$', regex=True)
 
    styled_df = (
        df.style
        .format(formats,escape='html')
        .set_table_styles(COMMON_TABLE_STYLE)
    )

    if hide_index:
        styled_df = styled_df.hide(axis='index')
            
    if return_value !='render':
        return styled_df

    display(styled_df)    
    return pd.DataFrame()


def visualize_overtime_change(
    pivot_df,
    decimals=1,
    cmap='RdYlGn',
    calculation='percentage',
    return_value='render',
    hide_index=False
):
    '''
    
    
    
    
    '''

    pct_change_df = (
        pivot_df
        .pct_change(axis=1)
        .replace([np.inf, -np.inf], np.nan).iloc[:, 1:]
    )

    def color_change(v):

        if pd.isna(v):
            return ''

        # Material decline
        if v <= -0.20:
            return 'background-color:#8B0000;color:white'

        # Moderate decline
        elif v <= -0.10:
            return 'background-color:#F4CCCC'

        # Minor decline
        elif v < -0.05:
            return 'background-color:#FCE5CD'

        # Stable
        elif v <= 0.05:
            return 'background-color:white'

        # Minor increase
        elif v < 0.10:
            return 'background-color:#D9EAD3'

        # Moderate increase
        elif v < 0.20:
            return 'background-color:#93C47D'

        # Material increase
        else:
            return 'background-color:#38761D;color:white'

    if calculation == 'percentage':

        formated_df = (
        pct_change_df
            .style
            #.applymap(color_change)
            .map(color_change)
            .format(f'{{:.{decimals}%}}')
            .format_index(lambda x: x.strftime('%Y-%m-%d'), axis='columns')
            .set_table_styles(COMMON_TABLE_STYLE)
            
        )

    
    else:

        formated_df = (
            pivot_df
            .style
            #.applymap(color_change, subset=pct_change_df.columns)
            .map(color_change, subset=pct_change_df.columns)  
            .format(f'{{:,.{decimals}f}}')
            .format_index(lambda x: x.strftime('%Y-%m-%d'), axis='columns')
            .set_table_styles(COMMON_TABLE_STYLE)
        )
    
    if hide_index:
        formated_df = formated_df.hide(axis='index')

    if return_value !='render':
        return formated_df
    
    display(formated_df)
    return pd.DataFrame()



def visualize_dataframe_by_row(
    df,
    show_as_int=[],
    show_as_number=[],
    show_as_currency=[],
    show_as_currency_without_dec=[],
    show_as_percentage=[],
    return_value='render'
):

    formats = {}

    for row in show_as_int:
        df.loc[row] = pd.to_numeric(df.loc[row], errors='coerce')
        formats[row] = '{:,.0f}'

    for row in show_as_number:
        df.loc[row] = pd.to_numeric(df.loc[row], errors='coerce')
        formats[row] = '{:,.2f}'

    for row in show_as_currency:
        df.loc[row] = pd.to_numeric(df.loc[row], errors='coerce')
        formats[row] = '${:,.2f}'

    for row in show_as_currency_without_dec:
        df.loc[row] = pd.to_numeric(df.loc[row], errors='coerce')
        formats[row] = '${:,.0f}'

    for row in show_as_percentage:
        df.loc[row] = pd.to_numeric(df.loc[row], errors='coerce')
        formats[row] = '{:.2f}%'

    styled_df = df.style

    for row, fmt in formats.items():
        styled_df = styled_df.format(
            fmt,
            subset=pd.IndexSlice[[row], :]
        )
        
        styled_df = (
        styled_df.set_table_styles(COMMON_TABLE_STYLE)
        )

    if return_value!='render':
        return styled_df
    
    display(styled_df)
    
    return pd.DataFrame()