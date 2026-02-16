import pandas as pd
import numpy as np
import datetime

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from daily_processes import generate_knowledge_base,create_py_table_dict,parse_dot_py_folder, generate_streamlit_definition_summary
from connections import download_file_from_git,backup_google_worksheets

# Updates saved //Github_Repo//Data//daily_test_results.csv which is Streamlit Input, Via Git.
# Note this is saved in a different place than other Daily Functions. BAD APPROACH
print('Generated D Knowledge Base')
generate_knowledge_base()

# Updates Github_Repo/Streamlit/DataDictionary/folder_listing.csv' which is Streamlit Input via Git. 
# Note this is saved in a different place then other Daily Functions. BAD APPROACH
print('Generated Python Table of Contents')
create_py_table_dict()

# Updates Github_Repo/Streamlit/DataDictionary/folder_listing.csv' which is Streamlit Input via Git. 
# Note this is saved in a different place than other Daily Functions. BAD APPROACH
print('Generated Other Pythong File?')
parse_dot_py_folder()


# Updates Dashboard File Definition Summary.
generate_streamlit_definition_summary('google_definition_csv')
generate_streamlit_definition_summary('d_learning_notes_url')
generate_streamlit_definition_summary('google_notes_csv')

# Back Up Google Sheets and Git Files Once a Week (On Wednesday)
# Github_Repo/CSV Backup Files/
if datetime.datetime.now().weekday()==2:
    print('Its Wednesday tine to back up Google Sheets and Git Hub Functions!')
    download_file_from_git()
    backup_google_worksheets()
