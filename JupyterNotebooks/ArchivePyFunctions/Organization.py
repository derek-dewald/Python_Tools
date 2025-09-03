# File Description: 
from IPython.display import display, Markdown, Math
from SharedFolder import ReadDirectory
from Connections import ParamterMapping
import pandas as pd
import re

def D_Notes_Reader(topic=None):
    '''
    Function to read Notes Files Saved in Google Docs

    Parameters: 
        topic (str): Argument to enable Filtering of Returned Dataframe to a Specific Topic. If a Filter is not applied it can be difficult
        to read the output as it's one continuous text without the Header Comments.

    Returns:
        Printed version of Notes (Powerpoint Like Format)

    Date Created: August 17, 20225
    Date Last Modified: 
    
    '''
    
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSQF2lNc4WPeTRQ_VzWPkqSZp4RODFkbap8AqmolWp5bKoMaslP2oRVVG21x2POu_JcbF1tGRcBgodu/pub?output=csv')
    if topic:
        temp = df[df['Process']==topic]
        if len(temp)>0:
            return JupyterNotebookMarkdown(temp)
    else:
        return JupyterNotebookMarkdown(df)
            

def CreateMarkdown(df, return_value=""):
    '''
    Create a Markdown string from a DataFrame with columns: Title, Header, Description
    '''
    try:
        df1 = df[['Title', 'Header', 'Description']]
    except:
        print('DataFrame must contain: Title, Header, Description')
        return ''

    title = ""
    step_number = 1
    text = ""

    for _, row in df1.iterrows():
        # New section
        if title != row['Title']:
            if title != "":
                text += "\n"  # separate sections
            title = row['Title']
            text += f"### {step_number}. {title}\n"
            step_number += 1

        # Level 2 bullet
        if isinstance(row['Header'], str) and row['Header'].strip():
            text += f"- {row['Header'].strip()}\n"

            # Level 3 bullet (indented)
            if isinstance(row['Description'], str) and row['Description'].strip():
                text += f"    - {row['Description'].strip()}\n"

    return text


def CreateMarkdownfromProcess(process_name=None,
                              return_value=""):
    '''
    Function to call Process Map from Google sheet with a single reference to the name of the process.
    
    Args:
        process_name (str): Process Name as defined in Google Sheet
        return_value (str): Value Desired to be returned, as required input from CreateMarkdown
    
    Returns:
        If return_value == "", then Displayed Markdown
        If return_value == "text", HTML formatted markdown text
        otherwise, returns ""
    '''
    # Call Parameter Mapping function to return DF of Process Sheet

    df = pd.read_csv(ParamterMapping('ProcessSheet')['CSV'].item())
    
    if process_name !=None:
        df1 = df[df['Process']==process_name].drop('Process',axis=1)
    else:
        return df
        
    try:
        if return_value =="":
            return CreateMarkdown(df1)
        elif return_value == "text":
            return CreateMarkdown(df1,'text')
    
    except:
        print("Could Not Format Data")
        return df

def DailyTest(questions=20,updates=5):
    
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv')
    
    updates = df[df['Definition'].isnull()].sample(updates)
    print(updates['Word'].tolist())
    
    questions = df[df['Definition'].notnull()].sample(questions)
    
    for row in range(len(df)):
        display_term_latex_dynamic(questions.iloc[row])

def display_term_latex_dynamic(row):
    '''
    Function Created to Sample Data Dictionary Rows and present information in incremental Format
    
    
    Parameters:
        Series
        
    Returns:
        Nil
    

    '''
    print(f"\n=== {row['Word']} ===")
    print(f"Category: {row['Category']}")
    input()
    print(f"Sub Category: {row['Sub Categorization']}")
    input()
    print(f"Definition: {row['Definition']}\n")
    input()
    if pd.notna(row['Markdown Equation']):
        eq_text = row['Markdown Equation']
        
        # Extract equation
        main_eq = re.search(r"\$\$(.*?)\$\$", eq_text, re.DOTALL)
        if main_eq:
            display(Markdown("**Equation:**"))
            display(Math(main_eq.group(1).strip()))
        
        # Extract "where" section
        where_part = re.split(r"\bwhere:\b", eq_text, flags=re.IGNORECASE)
        if len(where_part) > 1:
            display(Markdown("**Where:**"))
            where_lines = where_part[1].strip().splitlines()
            for line in where_lines:
                cleaned = line.strip("-• ").strip()
                if cleaned:
                    display(Math(cleaned))
    if pd.notna(row['Link']):
        display(Markdown(f"[More Info]({row['Link']})"))



    
def JupyterNotebookMarkdown(df, return_value=""):
    '''
    Function to Create a Markdown file from Process DF, which is a data frame of the structure, 
    Title, Header, Description

    Args:
        df (DataFrame): Must include columns Title, Header, Description
        return_value (str): 
            If "", renders HTML in notebook.
            If text, returns HTML Markdown string.
    
    Returns:
        str or display: Based on return_value


    Date Created:
    Date Last Maintained: August 17, 2025

    '''
    try:
        df1 = df[['Title', 'Header', 'Description']]
    except:
        print('DataFrame must include columns: Title, Header, Description')
        return ''

    text = ""
    step_number = 1
    last_title = None
    last_header = None
    open_l2 = False  # Track if L2 <ul> is open
    open_l3 = False  # Track if L3 <ul> is open

    for _, row in df1.iterrows():
        curr_title = row['Title']
        curr_header = row['Header']
        curr_description = row['Description']

        # If new Title
        if curr_title != last_title:
            if open_l3:
                text += "</ul>\n"
                open_l3 = False
            if open_l2:
                text += "</ul>\n"
                open_l2 = False
            if last_title is not None:
                text += "</ul>\n"  # Close previous title's outer <ul>

            text += f"<h4>{step_number}. {curr_title}</h4>\n<ul>\n"
            step_number += 1
            last_title = curr_title
            last_header = None  # Reset header context

        # If new Header
        if curr_header != last_header and isinstance(curr_header, str) and curr_header.strip():
            if open_l3:
                text += "</ul>\n"
                open_l3 = False
            if open_l2:
                text += "</ul>\n"
                open_l2 = False

            text += f"  <ul><li>{curr_header}</li>\n"
            open_l2 = True
            last_header = curr_header

        # If Description exists
        if isinstance(curr_description, str) and curr_description.strip():
            if not open_l3:
                text += "    <ul>\n"
                open_l3 = True
            text += f"      <li>{curr_description}</li>\n"

    # Close any open lists at the end
    if open_l3:
        text += "    </ul>\n"
    if open_l2:
        text += "  </ul>\n"
    text += "</ul>\n"

    if return_value == "":
        display(HTML(text))
    else:
        return text