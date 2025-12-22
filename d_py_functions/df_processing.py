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

    Step 1: Insert Definition for Values in Column CATEGORY has a definition record in column WORD
            notes.Category = definitions.Word
    Step 2: Insert Definition for Values in Column Category where the Corresponding Note
            notes.Category = definition.Category and definition.Categorization = Definition
    
    Step 3: Insert Definitions for Value in Categorization which have definitions in Word Column
        notes.categorization = definitions.word (Add Definition Categorization and Existing Category)
   
    Step 4: Insert Parameter Mapping, where List Type Options Described in Definitions.
           notes.Categorization = definitions.Category and notes.Word = definitions.Categorization

    Order Follows what is in Notes. Uses filtering to rank, and places values with Categorization of Definition for each topic at the top.

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

    try:
        len(definitions)
    except:
        definitions = pd.read_csv(links['google_definition_csv'])
    
    temp_def = definitions[['Category','Categorization','Word','Definition']].copy()

    # Step 1
    # Create Unique Category DF
    unique_cat_df = notes.drop_duplicates(['Category'])[['Category']]

    # Process temp_def such that 
    step1_df = unique_cat_df.merge(temp_def.drop('Category',axis=1),left_on='Category',right_on='Word',how='left')
    step1_df['Categorization'] = 'Definition'
    step1_df['Word'] = step1_df['Category'].copy()
    step1_df['Definition'] = step1_df['Definition'].fillna('Not Defined')

    # Step 2
    unique_cat_df['Categorization'] = 'Definition'
    step2_df = unique_cat_df.merge(temp_def,on=['Category','Categorization'],how='inner')

    # Step 3
    step_3df = notes.drop_duplicates('Categorization')[['Category','Categorization']].merge(definitions[['Word','Definition']].rename(columns={'Word':'Categorization'}),on='Categorization',how='left')
    step_3df['Word'] = 'Definition'
    step_3df['Definition'] = step_3df['Definition'].fillna('Not Defined')
    
    # Step 4
    mod_def = definitions[['Category','Categorization','Word','Definition']].copy()
    mod_def['Definition'] = mod_def['Word'] + ": " + mod_def['Definition']
    mod_def = mod_def.drop('Word',axis=1).reset_index(drop=True)
    mod_def = mod_def.rename(columns={'Categorization':'Word','Category':'Categorization'})
    
    # Merge Notes.Categorization and Notes.Word to Definitions.Category and Definitions.Categorization
    step_4df = notes.drop_duplicates(['Categorization','Word'])[['Category','Categorization','Word']].merge(mod_def,on=['Categorization','Word'],how='inner')
    
    # Can DO ORDER AT THE END. USING THE ORIGINAL DATASET and Category MERGE IN

    final_df = pd.concat([notes,step1_df,step2_df,step_3df,step_4df]).drop_duplicates()

    rank_df1 = notes.drop_duplicates('Category')[['Category']].reset_index(drop=True).reset_index().rename(columns={'index':"Rank1"})
    rank_df2 = notes.drop_duplicates(['Category','Categorization'])[['Category','Categorization']].reset_index(drop=True).reset_index().rename(columns={'index':"Rank2"})
    rank_df2['Rank2'] = rank_df2['Rank2'] + 1
    rank_df3 = notes.drop_duplicates(['Category','Categorization','Word']).reset_index(drop=True).reset_index().drop('Definition',axis=1).rename(columns={'index':'Rank3'})
    rank_df3['Rank3'] = rank_df3['Rank3'] + 1
    final_df = final_df.merge(rank_df1,on=['Category'],how='left').merge(rank_df2,on=['Category','Categorization'],how='left').merge(rank_df3,on=['Category','Categorization','Word'],how='left').fillna(0).sort_values(['Rank1','Rank2','Rank3']).reset_index(drop=True).drop(['Rank1','Rank2','Rank3'],axis=1)

    final_df['COUNT'] = final_df.groupby(['Category','Categorization','Word']).transform('size')
    final_df = final_df[~((final_df['Definition']=='')&(final_df['COUNT']!=1))].drop('COUNT',axis=1).reset_index(drop=True)
    # Need to Delete

    if export_location:
        final_df.to_csv(f"{export_location}d_learning_notes.csv",index=False)
    
    return final_df