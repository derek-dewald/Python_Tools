import pandas as pd
import numpy as np
import datetime

current_date = datetime.datetime.now().strftime('%d-%b-%y')

def DailyTask(dictionary,date=current_date,**kwargs):

    df = pd.DataFrame()
    
    for key,value in dictionary.items():
        if key in kwargs:
            value = kwargs[key]
        else:
            value = input(value)

        temp_df = pd.DataFrame([value],index=[date],columns=[key])
        try:
            df = pd.concat([df,temp_df],axis=1)
        except:
            pass

    print(df['Text'].item())
            
    return df.reset_index().rename(columns={'index':'Date'})

def generate_markdown_presentation(df, process_name, output_filename="presentation.md"):
    """
    Generates a Markdown presentation file from a filtered Pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'Process', 'Header', 'SubHeader', and 'Bullet'.
        process_name (str): The process name to filter the DataFrame.
        output_filename (str): The name of the Markdown file to generate.
    """
    # Filter the DataFrame based on process name
    df_filtered = df[df["Process"] == process_name]

    # Open file to write Markdown content
    with open(output_filename, "w", encoding="utf-8") as f:
        # Keep track of unique Headers & SubHeaders to ensure correct nesting
        last_header = None
        last_subheader = None

        for _, row in df_filtered.iterrows():
            header = row["Header"]
            subheader = row["SubHeader"]
            bullet = row["Bullet"]

            # Write Header if it's new
            if header != last_header:
                f.write(f"# {header}\n\n")  # Largest font size
                last_header = header
                last_subheader = None  # Reset last subheader since we switched header
            
            # Write Subheader if it's new
            if subheader != last_subheader:
                f.write(f"## {subheader}\n\n")  # 2 points smaller than header
                last_subheader = subheader

            # Write Bullet Point with square bullet (▪) and indent
            f.write(f"- {bullet}\n\n")  # 2 points smaller than subheader

    print(f"Markdown presentation saved as {output_filename}")

