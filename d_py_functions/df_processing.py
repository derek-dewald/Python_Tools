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
                                  definitions=None,
                                  export_location='/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/DataDictionary/'):
    
    '''
    
    Function which helps to combined Notes and Definitions into a Single Combined Representation which can ultimately be used as a Learning Reference Tools.

    How this function Works:
    It takes the two sheets and attemps to Consolidate them together to make Final Dataset, generated as d_learning_notes.csv.

    Approach is to Take Notes As the Framework and Distribute All Information into the Notes Sequentially in Specific Order so i can 
    Understand the structure and how to continue and utilize the Sheet.

    Principles: 
        - Define all types of Records in Notes and then Move them Out Definiton By Definition
        - Worry about Order at End.

    Step 1: Insert Records where Categorization = Definition directly into Sheet.
            With Modification:
                Where Word is also A Categorization, Update Categorization from Definition to Word, Change Word to Definition, 
                and Updated Definition to include WORD:

    Step 2: Take Everything else. Should be nothing remaining from Notes Category, so need to update Category, by moving all information 
            to the Right 1 column and consolidating Definition.
            
    Parameters:
        notes(df): DataFrame of D Notes as stored in: https://docs.google.com/spreadsheets/d/e/2PACX-1vSQF2lNc4WPeTRQ_VzWPkqSZp4RODFkbap8AqmolWp5bKoMaslP2oRVVG21x2POu_JcbF1tGRcBgodu/pub?output=csv
        definitions(df): DataFrame of D Definitions as stored in: https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv
        export_location(str): Location of where to Save CSV File. If blank, no CSV is made.

    date_created:20-Dec-25
    date_last_modified: 21-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        final_dataset_for_markdown()

    ##############

    Has been tested for a Single Value - Machine Learning. Need to Validate once extending.

    ##############

    
    '''

    from data_d_dicts import links

    try:
        len(notes)
    except:
        notes = pd.read_csv(links['google_notes_csv'])

    final_df = notes.copy()
    
    try:
        len(definitions)
    except:
        definitions = pd.read_csv(links['google_definition_csv'])

    temp_def = definitions[['Category','Categorization','Word','Definition']].copy()

    # Step 1
    # Create Definition only df and a Residual Definition DF res_def_df
    df_cat_is_definition = temp_def[temp_def['Categorization']=='Definition'].copy()

    
    
    cat_list = notes['Categorization'].unique().tolist()
    # When Word is also a Categorization, update so it is grouped correctly when sorting
    df_cat_is_definition['Categorization'] = np.where(df_cat_is_definition['Word'].isin(cat_list),df_cat_is_definition['Word'],df_cat_is_definition['Categorization'])
    
    mask = df_cat_is_definition["Categorization"] != "Definition"

    df_cat_is_definition.loc[mask, "Definition"] = (
        df_cat_is_definition.loc[mask, "Word"].astype(str)
        + ": "
        + df_cat_is_definition.loc[mask, "Definition"].astype(str)
    )

    df_cat_is_definition['Word'] = np.where(df_cat_is_definition['Categorization']!='Definition','Definition',df_cat_is_definition['Word'])
    res_def_df = temp_def[temp_def['Categorization']!='Definition'].copy()
    

    #Step 2
    
    def prepare_df_for_insert(df,notes=notes):
        df = df.copy()
        df['Definition'] = df['Word'] + ": " + df['Definition']
        df['Word'] = df["Categorization"].copy()
        df['Categorization'] = df["Category"].copy()
        df.drop('Category',axis=1,inplace=True)
        
        # Need to Identify Category. Should Primarily be from Categorization, might be from Word in Rare cases ("Regularization")
        df1 =  df[['Categorization']].drop_duplicates().merge(notes[['Category','Categorization']].drop_duplicates(),on='Categorization',how='left')    
        return df.merge(df1,on=['Categorization'],how='left')

    res_def_df = prepare_df_for_insert(res_def_df)
    
    # Insert 
    final_df = pd.concat([final_df,df_cat_is_definition,res_def_df])

    ################
    
    # Ranking for Sort ORder
    rank_df1 = notes.drop_duplicates('Category')[['Category']].reset_index(drop=True).reset_index().rename(columns={'index':"CY_RANK"})
    rank_df2 = notes.drop_duplicates(['Category','Categorization'])[['Category','Categorization']].reset_index(drop=True).reset_index().rename(columns={'index':"CZ_RANK"})
    rank_df2['CZ_RANK'] = rank_df2['CZ_RANK'] + 1
    rank_df3 = notes.drop_duplicates(['Category','Categorization','Word']).reset_index(drop=True).reset_index().drop('Definition',axis=1).rename(columns={'index':'WORD_RANK'})
    rank_df3['WORD_RANK'] = rank_df3['WORD_RANK'] + 1

    final_df = final_df.merge(rank_df1,on=['Category'],how='left').merge(rank_df2,on=['Category','Categorization'],how='left').merge(rank_df3,on=['Category','Categorization','Word'],how='left')
    final_df['CZ_RANK'] = np.where(final_df['Categorization']=='Definition',0,final_df['CZ_RANK'])
    final_df['WORD_RANK'] = np.where(final_df['Word']=='Definition',0,final_df['WORD_RANK'])
    final_df =  final_df.sort_values(['CY_RANK','CZ_RANK','WORD_RANK'])

    if export_location:
        final_df.to_csv(f"{export_location}d_learning_notes.csv",index=False)
    
    return final_df