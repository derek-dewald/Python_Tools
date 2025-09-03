from IPython.display import display, Markdown, Math
import pandas as pd
import re

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