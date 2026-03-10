

# Manually Maintained List of Python Function Files
function_table_dictionary = {
    'connections':'Functions Connecting to External Data Sources',
    'data_d_dicts':'Repository of Dictionaries which have been saved for Easy Use and Reference',
    'data_d_lists':'Repository of Lists which have been saved for Easy Use and Reference',
    'data_d_strings':'Repository of Strings which have been saved for Easy Use and Reference',
    'df_processing': 'Functions related to dataframe Transformations, including; Filtering, Slicing, Parsing, Transposing. Individual Column Level Transformations Primarily in FeatureEngineering',
    'dict_processing':'Functions related to Manipulating, Transforming and Altering Dictionaries',
    'input_functions_ignore':'Functions which are Created for Single Use, or I can not reconcile their purpose, but they are used as Input for previosuly created functions',
    'list_processing':'Functions related to Manipulating, Transforming and Altering Lists',
    'string_processing':'Functions related to Manipulating, Transforming and Altering Strings',
    'shared_folder':'Functions related to Management, Maintenance and Upkeep of Windows and Mac OS File Folders',
    'sql_':'Functions related to Processing of SQL',
    'utility_functions':"Functions which are in development, do not nicely fit into the existing schema, or otherwise need attention"
   
    }

# List of Historical Python Function Files, to be widdled down to Deletion as finalizing Work.
tbd = {
    'data_creation': "Functions related to the creation of Data, for testing, reference or validation",
    'df_eda':"Functions related to the Structured Exploration of Datasets, to understand and explain",
    'date_functions':'Functions related Manipulation, Change and process of datetime',
    'df_stats':"Functions utilizing Statistical Concepts, and Calculations",
    'feature_engineering':'Functions related to the creation of New Columns, excludes TEXT cleaning.',
    'ml_pipeline':"Functions related to Custom Build ML Pipelines, Approaches and Techniques",
    'validations':"Functions related to Validation of Dataframes, Data Sets and Comparisons. This differs from EDA, where it is only about understanding, not Comparison",
    
}

links = {
    'google_mapping_sheet_csv':'https://docs.google.com/spreadsheets/d/e/2PACX-1vSwDznLz-GKgWFT1uN0XZYm3bsos899I9MS-pSvEoDC-Cjqo9CWeEuSdjitxjqzF3O39LmjJB0_Fg-B/pub?output=csv',
    'google_notes_csv':'https://docs.google.com/spreadsheets/d/e/2PACX-1vSQF2lNc4WPeTRQ_VzWPkqSZp4RODFkbap8AqmolWp5bKoMaslP2oRVVG21x2POu_JcbF1tGRcBgodu/pub?output=csv',
    'google_definition_csv':'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv',
    'google_word_quote':'https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=1117793378&single=true&output=csv',
    'google_daily_activities':'https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=472900611&single=true&output=csv',
    'function_list_url':"https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_list.csv",
    'parameter_list_url':"https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_parameters.csv",
    'folder_toc_url':"https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/DataDictionary/folder_listing.csv",
    'd_knowledge_base_url':"https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/DataDictionary/d_knowledge_base.csv",
    #'d_learning_notes_url':"https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/DataDictionary/d_learning_notes.csv",
    'technical_notes':'https://docs.google.com/spreadsheets/d/e/2PACX-1vSnwd-zccEOQbpNWdItUG0qXND5rPVFbowZINjugi15TdWgqiy3A8eMRhbmSMBiRhHt1Qsry3E8tKY8/pub?output=csv',
    'd_daily_test_score':'https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Data/daily_test_results.csv',
    'process_cl':'https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Data/d_knowledge_process_checklist.csv'
}

emoji_dict = {
    "STATUS": {
        "label": "Status / Progress / Outcome",
        "description": "Execution state, progress tracking, completion, and flow of work.",
        "emojis": ['✅', '✔️', '⏸️', '🟢', '🟡', '🔴', '⏳', '🕐', '📤', '📥', '🗂️']
    },
    "DATA": {
        "label": "Data / Analytics / Artifacts",
        "description": "Raw data, analytical outputs, reports, documentation, and structured artifacts.",
        "emojis": ['📊', '📈', '📉', '📋', '📝', '📚', '🔢', '📦', '🧾']
    },
    "QUALITY": {
        "label": "Data Quality / Validity / Trust",
        "description": "Data cleanliness, assumptions, validation, lineage, and reliability checks.",
        "emojis": ['🧼', '🔍', '🧪', '🧾', '🧯', '🪪']
    },
    "ANALYSIS": {
        "label": "Analysis / Statistics / Reasoning",
        "description": "Measurement, statistical thinking, mathematical rigor, and analytical reasoning.",
        "emojis": ['🧠', '🧮', '📐', '📏', '🎲', '🔔', '🌡️', '⚖️']
    },
    "MODEL": {
        "label": "Modeling / Machine Learning",
        "description": "Feature engineering, model construction, training, tuning, and pipelines.",
        "emojis": ['🤖', '🧩', '🧱', '🏗️', '🎛️', '🪜', '🪝', '🔄']
    },
    "EVAL": {
        "label": "Evaluation / Performance / Testing",
        "description": "Model assessment, metrics, validation, experimentation, and testing outcomes.",
        "emojis": ['🧪', '🎯', '📉', '📈', '🔁']
    },
    "INSIGHT": {
        "label": "Insight / Discovery / Understanding",
        "description": "Pattern discovery, interpretation, understanding, and sense-making.",
        "emojis": ['💡', '💭', '🤔', '🧐', '🧭']
    },
    "DECISION": {
        "label": "Decision / Optimization / Strategy",
        "description": "Judgment, trade-offs, prioritization, optimization, and final recommendations.",
        "emojis": ['⚖️', '🎯', '🚦', '🏁', '🧠']
    },
    "ACTION": {
        "label": "Tools / Engineering / Execution",
        "description": "Implementation, engineering work, tooling, and operational execution.",
        "emojis": ['🛠️', '🔧', '🧰', '⚙️', '🗜️']
    },
    "TIME": {
        "label": "Time / Change / Drift",
        "description": "Trends over time, decay, refresh cycles, and model or data drift.",
        "emojis": ['🕰️', '🌊', '📉', '🔄', '🧊']
    },
    "CAUSAL": {
        "label": "Causality / Impact / Inference",
        "description": "Cause-effect reasoning, inference, treatments, and impact assessment.",
        "emojis": ['🔗', '🧪', '🎯', '🧠', '🧯']
    },
    "RISK": {
        "label": "Risk / Alerts / Flags",
        "description": "Warnings, failures, blockers, risks, and critical issues.",
        "emojis": ['⚠️', '❌', '🚨', '🚩', '🛑', '❗', '❓', '⛔']
    },
    "COLLAB": {
        "label": "Collaboration / Communication",
        "description": "Sharing, alignment, discussion, and coordination with others.",
        "emojis": ['📣', '🗣️', '👥', '🤝', '📨']
    },
    "ENERGY": {
        "label": "Motivation / Momentum / Wins",
        "description": "Progress celebration, motivation, momentum, and morale.",
        "emojis": ['🚀', '🔥', '💥', '💫', '🌟', '🏆', '💎', '🌱']
    }
}
