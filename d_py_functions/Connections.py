
import pandas as pd

def DownloadDataURL(link,
                    source="",
                    file_type='csv'):
    
    if source.lower()=='github':
        link = f"https://raw.githubusercontent.com/{link}"
        
    if file_type=='csv':        
        return pd.read_csv(link)
    else:
        return pd.read_excel(link)