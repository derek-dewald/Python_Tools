from IPython.display import display, HTML
import pandas as pd
import numpy as np


def notes_df_to_outline_html(
    df: pd.DataFrame,
    column_order=None):
    
    """

    Function to Take a Dataframe and convert it into A Structured Indented Point form Format. 
    Used for Clear Visualization of Notes.
    
    Parameters:
        df(df): Any DataFrame
        column_order(list): List of Columns to Include, in Order. If not defined, all will be included.

    Returns:
        str

    date_created:12-Dec-25
    date_last_modified: 12-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        from connections import d_google_sheet_to_csv
        df = import_d_google_sheet('Notes')
        notes_df_to_outline_html(df)

    """

    if column_order is None:
        column_order = df.columns.tolist()

    missing = [c for c in column_order if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df1 = df[column_order].copy()

    def clean(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    last = [""] * len(column_order)

    # ---- HTML + CSS wrapper ----
    html = """
            <style>
            .notes-container {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
            }
            
            .notes-item {
                line-height: 1.45;
                margin: 2px 0;
            }
            
            .notes-l0 { font-size: 18px; font-weight: 600; margin-left: 0px; }
            .notes-l1 { font-size: 16px; font-weight: 500; margin-left: 18px; }
            .notes-l2 { font-size: 14px; font-weight: 400; margin-left: 36px; }
            .notes-l3 { font-size: 13px; font-weight: 400; margin-left: 54px; opacity: 0.85; }
            .notes-l4 { font-size: 12px; font-weight: 400; margin-left: 72px; opacity: 0.8; }
            </style>
            
            <div class="notes-container">
            """

    for _, row in df1.iterrows():
        vals = [clean(row[c]) for c in column_order]
        if all(v == "" for v in vals):
            continue

        # Find first level where value changes
        change_level = None
        for i, v in enumerate(vals):
            if v and v != last[i]:
                change_level = i
                break

        # If nothing changes, show deepest non-blank value
        if change_level is None:
            for i in range(len(vals) - 1, -1, -1):
                if vals[i]:
                    change_level = i
                    break

        # Reset deeper levels when higher level changes
        if change_level is not None:
            for j in range(change_level, len(last)):
                last[j] = ""

        # Render new values
        for i in range(change_level, len(vals)):
            v = vals[i]
            if not v:
                continue
            if v != last[i]:
                level = min(i, 4)  # cap style depth
                bullet = "• " if i > 0 else ""
                html += (
                    f'<div class="notes-item notes-l{level}">'
                    f'{bullet}{v}'
                    f'</div>\n'
                )
                last[i] = v

    html += "</div>"

    display(HTML(html))

    return html
