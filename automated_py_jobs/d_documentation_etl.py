import pandas as pd
import numpy as np
import datetime

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from taxonomy_admin_functions import create_knowledge_base,create_python_documentation,generate_objects_automated_py
from objects_automated import object_dict

notes = pd.read_csv(object_dict['csv_links']['python_object']['google_notes_csv'])
definitions = pd.read_csv(object_dict['csv_links']['python_object']['google_definition_csv'])

knowledge_base_df= create_knowledge_base(
    definition_df=definitions,
    notes_df=notes
)

# Update automated_objects.py with Refreshed Information
generate_objects_automated_py()

# Update Python Documentation.
function_list,parameter_list,function_file_definition = create_python_documentation()
