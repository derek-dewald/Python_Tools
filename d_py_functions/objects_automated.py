

'''
module_name: objects_automated
module_purpose: Created to serve as a repository for automatically created lists, dictionaries and strings from Google Notes, Dictionaries and other sources as appropriate.  File is created by _____. Whenever run it is automatically overwriden
    
'''
object_dict = {}

object_dict['cat_reference_list'] = {
    'Process':"Categorization Values Currently in Use",
    'Categorization':'Reference List',
    'Word':"Parameter",
    'Definition':"Comprehensive List of all Values utilized in Organizational Taxonomy in the Column Categorization",
    'publish':1,
    'python_object':['Information', 'Guidance', 'Requirement', 'Process Step', 'Concept', 'Definition', 'Regularization', 'Transformation', 'Feature Selection', 'Required', 'Column List', 'Reference List', 'Filter Order']
    }

object_dict['process_reference_list'] = {
    'Process':"Process Values Currently in Use",
    'Categorization':'Reference List',
    'Word':"Parameter",
    'Definition':"Comprehensive List of all Values utilized in Organizational Taxonomy in the Column Process",
    'publish':1,
    'python_object':['Behavioural Economics', 'Feature Engineering', 'Goal Setting', 'Problem Definition', 'Data Collection', 'General Definition', 'Data Preparation', 'Model Evaluation', 'Statistics', 'Best Linear Unbiased Estimator', 'Data Dictionary', 'Documentation', 'Machine Learning Lifecycle', 'Notes', 'Problem Solving Framework']
    }

object_dict['csv_links'] = {
    'Process':"Organization",
    'Categorization':'Reference Dictionary',
    'Word':"CSV Links",
    'Definition':"Dictionary of Links to Google Sheet, Git Hub and other pertinent datasource",
    'publish':0,
    'python_object':{'google_definition_csv': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv', 'google_notes_csv': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSQF2lNc4WPeTRQ_VzWPkqSZp4RODFkbap8AqmolWp5bKoMaslP2oRVVG21x2POu_JcbF1tGRcBgodu/pub?output=csv', 'google_word_quote': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=1117793378&single=true&output=csv', 'google_daily_activities': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=472900611&single=true&output=csv', 'technical_notes': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSnwd-zccEOQbpNWdItUG0qXND5rPVFbowZINjugi15TdWgqiy3A8eMRhbmSMBiRhHt1Qsry3E8tKY8/pub?output=csv', 'd_links': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=469651051&single=true&output=csv', 'function_list_url': 'https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_list.csv', 'parameter_list_url': 'https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_parameters.csv', 'folder_toc_url': 'https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/DataDictionary/folder_listing.csv', 'd_knowledge_base_url': 'https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/DataDictionary/d_knowledge_base.csv', 'process_cl': 'https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Data/d_knowledge_process_checklist.csv', 'SKlearn Model Parameters': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSnzcSYTXm2jl9GXvrNaH3b3TPbNufGJwRTMeJ8Ckhse_r9CWGjlXWlUfyGTwcnoXqT7ZLLyBAk2rKO/pub?gid=175044227&single=true&output=csv', 'Sklearn Models': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vToNab_ADzmxRZkg4bJ5wOwLZcrdwNYrxwWBETdGlfGSoUpnyy799EpbYqDqnwyKs2bEyJHUu58SX8Q/pub?gid=1125524518&single=true&output=csv'}
        }
        
object_dict['url_links'] = {
    'Process':"Organization",
    'Categorization':'Reference Dictionary',
    'Word':"URL Links",
    'Definition':"Dictionary of Links to Google Sheet, Git Hub and other pertinent datasource",
    'publish':0,
    'python_object':{'google_definition_csv': 'https://docs.google.com/spreadsheets/d/1tZ-_5Vv99_bm9CCEdDDN0KkmsFNcjWeKM86237yrCTQ/edit?gid=0#gid=0', 'google_notes_csv': 'https://docs.google.com/spreadsheets/d/1jddkkF5IWRr_eV1hTjB-T_IlmGJ-GpizYgeMgZQQxb4/edit?gid=0#gid=0', 'technical_notes': 'https://docs.google.com/spreadsheets/d/1FpYYq4LN6AZBaNRhnj1f76YNvnG-hTco40wJ1PUugto/edit?gid=0#gid=0', 'd_links': 'https://docs.google.com/spreadsheets/d/14CtNmNIajcY1mlEkRWw4ka93MQYGs2wN-kvL5qhvQX0/edit?gid=469651051#gid=469651051', 'Job Search': 'https://docs.google.com/spreadsheets/d/1sMdgmp80DXDojDl6uYS6INdoJxVDbhIwpwCtpxBjHVg/edit?gid=668067090#gid=668067090', 'OurWorldData': 'www.ourworlddata.org', 'KaggelDatasets': 'https://www.kaggle.com/datasets', 'StLouisFed': 'https://fred.stlouisfed.org', 'GoggleResearch': 'https://datasetsearch.research.google.com', 'UCI': 'https://archive.ics.uci.edu', 'AWS': 'https://registry.opendata.aws/', 'AWSDataExchange': 'https://aws.amazon.com/data-exchange/', 'AWSPublicData': 'https://aws.amazon.com/public-datasets/', 'SKlearn Model Parameters': 'https://docs.google.com/spreadsheets/d/1GhIiuEMY-A-SNtQOj5Z24wARLVGvEd10qVT3MpGtj2Y/edit?gid=175044227#gid=175044227', 'TensorFlow Playground': 'https://playground.tensorflow.org/'}
        }
    