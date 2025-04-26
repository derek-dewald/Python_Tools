from IPython.core.display import display, HTML
import pandas as pd
import numpy as np
import datetime

def JupyterNotebookMarkdown(df,return_value=""):
    
    '''
    Function to Create a Markdown file from Process DF, which is a data frame of the structure, 
    Title,Header,Description
    
    Args:
        Dataframe( Must be of format, Title, Header, Description)
        return_value (str: "" or text):
            If Blank, will render text in HTML Format. 
            If text, then will return text for rendering in HTML Markdown
    
    Returns:
        Conditional on Return Value. Please read Args.
    
    
    '''
    
    try:
        df1 = df[['Title','Header','Description']]
    
    except:
        
        print('DataFrame does not meet structure requirement, which must include 3 Column: Title, Header, Description')
        return ''
    
    title= ""
    step_number = 1
    text = ""

    l2_bullet = '-'  # Level 2 Bullet
    l3_bullet = '*'  # Level 3 Bullet

    for index, row in df1.iterrows():
        # Ensure previous list is closed before starting a new title
        if title and title != row.iloc[0]:  
            text += "</ul>\n"  # Close the last unordered list before switching to a new title

        # If it's a new title, start a new section
        if title == "" or title != row.iloc[0]:
            text += f"<h4>{step_number}. {row.iloc[0]}</h4>\n<ul>\n"  # Reset indentation
            step_number += 1
            title = row.iloc[0]  # Store the new title

        # Add Level 2 content (Column 2)
        if isinstance(row.iloc[1], str) and row.iloc[1].strip():
            text += f"  <li>{row.iloc[1]}</li>\n"  # L2 starts here

            # Add Level 3 content (Column 3) only if it exists
            if isinstance(row.iloc[2], str) and row.iloc[2].strip():
                text += f"    <ul><li>{row.iloc[2]}</li></ul>\n"  # L3 indented under L2

    text += "</ul>\n"  # Close any remaining lists

    if return_value =="":# Display the formatted HTML output in Jupyter Notebook
        display(HTML(text))
        
    else:
        return text