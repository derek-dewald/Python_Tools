import pandas as pd
import numpy as np
import html
import textwrap

#from IPython.display import display, HTML

def notes_df_to_outline_html(
    df: pd.DataFrame,
    column_order=None):
    
    """

    Function to Take a Dataframe and convert it into A Structured Indented Point form Format. 
    Used for Clear Visualization of Notes.
    
    Parameters:
        df(df): Any DataFrame
        column_order(list): List of Columns to Include, in Order. If not defined, all will be included.
        print_(bool): Option as to whether you wish to directly Render a print out in the Python Session. Added because of Streamlit Error, need to suppress.

    Returns:
        str

    date_created:12-Dec-25
    date_last_modified: 18-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        from connections import d_google_sheet_to_csv
        df = import_d_google_sheet('Notes')
        notes_df_to_outline_html(df)

    Update: 
        Added display parameter to support Streamlit Adoption.

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

    html_ = """
    <style>
    .notes-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; }
    .notes-item { line-height: 1.45; margin: 2px 0; }

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

        # Still nothing? (paranoia guard)
        if change_level is None:
            continue

        # Reset deeper levels when higher level changes (deeper only)
        for j in range(change_level + 1, len(last)):
            last[j] = ""

        # Render new values from change_level downward
        for i in range(change_level, len(vals)):
            v = vals[i]
            if not v:
                continue
            if v != last[i]:
                level = min(i, 4)  # cap style depth
                safe_v = html.escape(v) # Escape Function, not String.
                html_ += f'<div class="notes-item notes-l{level}">{safe_v}</div>\n'
                last[i] = v

    html_ += "</div>"
    html_ = textwrap.dedent(html_).lstrip()
    return html_

def final_dataset_for_markdown(notes=None,
                               definitions=None):
    '''
    Function which helps to combined Notes and Definitions into a Single Combined Representation which can ultimately be used as a Learning Reference Tools.


    
    '''

    from data_d_dicts import links

    try:
        len(notes)
    except:
        notes = pd.read_csv(links['google_notes_csv'])

    try:
        len(definitions)
    except:
        definitions = pd.read_csv(links['google_definition_csv'])

    merge_def = definitions[['Word','Definition','Category']]

    # For All Unique Categories, Add Definitions. Need to fill in other Columns of Dataframe for Future Merging
    temp_df = notes.drop_duplicates("Category")[['Category']]
    temp_df['Categorization'] = 'Definition'
    temp_df['Word'] = temp_df['Category']
    # Rank Utilized to provide Definitions with Highest Value when Categorizing, otherwise Sort a Issue when Merging
    temp_df['RANK'] = 0 
    
    # Merge In Combination of Category and Categorization
    temp_df1 = notes.drop_duplicates(['Category','Categorization']).drop(['Word','Notes'],axis=1).copy()
    temp_df1 = temp_df1.merge(definitions[['Word','Definition','Category']].rename(columns={'Word':'Categorization','Definition':"Notes"}),on=['Category','Categorization'],how='left')
    temp_df1 = temp_df1[temp_df1['Notes'].notnull()].copy()
    temp_df1['Word'] = temp_df1['Categorization'].copy()
    temp_df1['RANK'] = 0 
 
    # Merge In Definition to Category
    temp_df = temp_df.merge(merge_def.rename(columns={'Definition':"Notes"}),on=['Word','Category'],how='left')   
    temp_df2 = notes.merge(merge_def,on=['Word','Category'],how='left')
    temp_df2['RANK'] = 1

    # Merge In Listed Parameters 
    temp_df3 =  notes[['Category','Categorization','Word']].drop_duplicates(['Categorization','Word'])

    # Generate Information From Definition. Must remove Note column from current dataframe to accept for purposes of merging.
    list_ = definitions.drop('Notes',axis=1).rename(columns={'Word':'Notes','Categorization':'Word','Category':'Categorization'})[['Categorization','Word','Notes','Definition']]
    list_["Notes"] = list_["Notes"].astype(str) + ": " + list_["Definition"].astype(str)
    list_.drop('Definition',axis=1,inplace=True)
    
    # Merge Into Notes.
    temp_df3 =  temp_df3.merge(list_,on=['Categorization','Word'],how='inner')
    
    # Create Final DataFrame
    final_df = pd.concat([temp_df,temp_df1,temp_df2,temp_df3])

    # Base Column is Notes, Definitions Merged in to Help Articulate and combine, thus if Notes Available, it will take precendent
    final_df['Notes'] = np.where(final_df['Notes']=="",final_df['Definition'],final_df['Notes'])
    final_df.drop('Definition',axis=1,inplace=True)
    
    # Calculuate Order DF. Using Order as defined in Notes Template.
    order_df =  notes[['Category','Categorization']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'ORDER'})
    final_df = final_df.merge(order_df,on=['Category','Categorization'],how='left')
    final_df['ORDER'] = final_df['ORDER'].fillna(0) 
    
    # Sort    
    final_df =  final_df.sort_values(['ORDER','RANK']).drop(['ORDER','RANK'],axis=1)

    # Remove Blank Place Holders that were satisfied by Merging Parameter Listing (Optimization, Optimizers as Example)
    final_df['count'] = final_df.groupby(['Category','Categorization','Word']).transform('size')
    final_df = final_df[~((final_df['count']>1)&(final_df['Notes'].isnull()))].copy()

    return final_df.drop('count',axis=1)

