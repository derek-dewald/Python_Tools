'''
module_name: objects_manual
module_purpose: Created to serve as a repository for manually created lists, dictionaries, etc.
default_structure: object_dict['NAME'] = {'Process':"",'Categorization':'','Word':"Name",'Definition':"",'publish':0,'python_object':''}
module_guidance: This is the Default Location for Lists, Templates and process order. Publish relates to automatic inclusion into Notes

'''


import datetime
object_dict = {}

object_dict['Example'] = {
    'Process':"",
    'Categorization':'',
    'Word':"Example",
    'Definition':"",
    'publish':0,
    'python_object':[]
    }

#############################
# Default Categorization Options for this Sheet. 

object_dict['objects_manual_cat_list'] = {
    'Process':"Documentation",
    'Categorization':'Reference List',
    'Word':"Objects Categorization",
    'Definition':"Default Categorization List Options for Objects, which assigns taxonmy to my Organizational Structure",
    'publish':1,
    'python_object':[
        'Process Step',
        'Relative Order',
        'Column List',
        'Material Change Log',
        'Reference List',
        'Reference String',    
        'Reference Dictionary'
        ]
    }



#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################
#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################

#############################
object_dict['blue_requirmements'] = {
    'Process':"Best Linear Unbiased Estimator",
    'Categorization':'Required',
    'Word':"Best Linear Unbiased Estimator",
    'Definition':"The Best Linear Unbiased Estimator (BLUE) is an estimation method in statistics that produces parameter estimates that are linear in the observed data, unbiased, and have the smallest possible variance among all such estimators. It originates from the Gauss–Markov theorem, developed in the early 19th century in the context of least squares and error theory. Its importance lies in providing a theoretical benchmark: under specific assumptions about error structure, no other linear unbiased estimator can be more precise. In practice, BLUE underpins ordinary and generalized least squares, and is widely applied in econometrics, engineering, and data science when modeling relationships with correlated or heteroskedastic errors",
    'publish':1,
    'python_object':[
        'Homoscedasticity',
        'Independence',
        'Linearity',
        'Normality'
]}

#############################

object_dict['problem_framework_relative_order'] = {
    'Process':"Notes",
    'Categorization':'Filter Order',
    'Word':"Categorization Filter Order",
    'Definition':"Default Column Order to be applied while filtering Notes to D Knowledge Base or/and D Processes",
    'publish':1,
    'python_object':[
        'Process Definition',
        'Definition',
        'Process Step',
        'Guiding Principle',
        'Guidance',
        'Expected Outcomes',
        'Parameter',
        'Algorithm',
        'Information']}

object_dict['problem_solving_framework'] = {
    'Process':"Problem Solving Framework",
    'Categorization':'Process',
    'Word':"Problem Solving Framework",
    'Definition':"Structured approach to solving complex problems.",
    'publish':1,
    'python_object':[
        'Goal Setting',
        'Problem Definition',
        'Current State Assessment',
        'Data Collection',
        'Root Cause Analysis',
        'Research and Exploration',
        'Option Generation',
        'Solution Design',
        'Feasibility Assessment',
        'Prioritization and Decision Making',
        'Implementation Planning',
        'Execution',
        'Testing and Validation',
        'Evaluation',
        'Optimization',
        'Documentation and Knowledge Sharing',
        'Monitoring',
        'Continuous Improvement',
        'Governance, Risk, and Ethics'
        ]
    }




object_dict['machine_learning_process'] = {
    'Process':"Machine Learning Lifecycle",
    'Categorization':'Process',
    'Word':"Machine Learning Framework",
    'Definition':"Heirarchal Structure to guide for a robust and successful Machine Learning Project, including all of the high level steps/consideration which must be given.",
    'publish':1,
    'python_object':[
            'Goal Setting',
            'Problem Definition',
            'Data Collection',
            'Data Preparation',
            'Exploratory Data Analysis',
            'Feature Engineering',
            'Feature Selection',
            'Model Selection',
            'Training',
            'Hyperparameter Tuning',
            'Model Evaluation',
            'Validation',
            'Model Interpretability',
            'Deployment',
            'Monitoring',
            'Bias, Fairness, and Ethics']}

object_dict['data_dictionary_columns'] = {
    'Process':"Data Dictionary",
    'Categorization':'Column List',
    'Word':"Data Dictionary Columns",
    'Definition':"List of Default Columns Used in Data Dictionary",
    'publish':1,
    'python_object':[
        'FINALNAME',
        'COLUMN',
        'METRIC_TYPE',
        'METRIC_CLASSIFICATION',
        'PRODUCT',
        'DESCRIPTION',
        'METRIC',
        'NOTES',
        'RECORD_TYPE',
        'UPDATE_DATE'
        ]}

object_dict['activity_notes'] = {
    'Process':"D Organization",
    'Categorization':'Material Change Log',
    'Word':"activity_notes",
    'Definition':"List of Log Decisions, Material Updated and Changes to D Reporting Structure",
    'publish':0,
    'python_object':[

    {
        'DATE':'01-JAN-01',
        'ACTIVITY':'Notes',
        'NOTES':'',
        'STATUS':'',
        'PENDING_ACTION':'',  
        'CLASSIFICATION':'',
        'SUB_CLASSIFICATION':''
    },
    {
        'DATE':'26-JUN-26',
        'ACTIVITY':'Moved All .py files from Folder d_py_functions to   d_py_functions>>archive>>historical and made a Folder Delete. Purpose to streamline and implement consistent approach. Logic. Naming etc.',
        'NOTES':'',
        'STATUS':'',
        'PENDING_ACTION':'',  
        'CLASSIFICATION':'',
        'SUB_CLASSIFICATION':''
    },
    {
        'DATE':'26-JUN-26',
        'ACTIVITY':'Deleted Process/Map Page/Daily Activities on Google Sheets. Process Page to be automated action from notes, Map moved to D Sheet.',
        'NOTES':'If I have not done something for 20 years, is it resonable or likely that I will start. Change Action, not documentation',
        'STATUS':'',
        'PENDING_ACTION':'',  
        'CLASSIFICATION':'',
        'SUB_CLASSIFICATION':''
        
    },
    {
        'DATE':'26-JUN-26',
        'ACTIVITY':'Created New .py files reference_data and manual_data',
        'NOTES':'',
        'STATUS':'',
        'PENDING_ACTION':'',  
        'CLASSIFICATION':'',
        'SUB_CLASSIFICATION':''
    },
    {
        'DATE':'26-JUN-26',
        'ACTIVITY':'',
        'NOTES':'',
        'STATUS':'',
        'PENDING_ACTION':'',  
        'CLASSIFICATION':'',
        'SUB_CLASSIFICATION':''
    },]}


object_dict['hex_color_list'] = {
    'Process':"Visualziation",
    'Categorization':'Reference List',
    'Word':"hex_color_list",
    'Definition':"List of over 500 distinct Hex Color Comnbinations. ",
    'publish':0,
    'python_object':[
'#808080','#efc050','#0000cd','#060','#ff4040','#FFC0CB','#EEDFCC','#0ff','#8a3324','#6495ed','#ff7f00','#8a2be2','#050505','#841b2d','#a4c639','#cd9575','#ffbf00','#c46210','#E3CF57','#f4c2c2',
'#fae7b5','#ffe4c4', '#fe6f5e','#bf4f51','#a57164','#ace5ee','#5d8aa8','#00308f','#72a0c1','#a32638','#f0f8ff','#e32636','#efdecd','#e52b50','#ff7e00','#ff033e','#96c', '#f2f3f4','#915c83',
'#faebd7','#008000','#8db600','#fbceb1','#7fffd4','#4b5320','#3b444b','#e9d66b','#b2beb5','#87a96b','#f96','#a52a2a','#fdee00','#6e7f80','#568203','#007fff','#f0ffff','#89cff0','#a1caf1',
'#21abcd','#ffe135','#7c0a02','#848482','#98777b','#bcd4e6','#9f8170','#f5f5dc','#9c2542','#3d2b1f','#000','#3d0c02','#253529','#3b3c36','#ffebcd','#318ce7','#faf0be','#00f','#a2a2d0','#1f75fe',
'#0d98ba','#0093af','#0087bd','#339','#0247fe','#126180','#de5d83','#79443b','#0095b6','#e3dac9','#c00','#006a4e','#873260','#0070ff','#b5a642','#cb4154','#1dacd6','#6f0','#bf94e4','#c32148',
'#ff007f','#08e8de','#d19fe8','#f4bbff','#ff55a3','#fb607f','#004225','#cd7f32','#964b00','#a52a2a','#ffc1cc','#e7feff','#f0dc82','#480607','#800020','#deb887','#c50','#e97451','#808080',
'#bd33a4','#702963','#536872','#5f9ea0','#91a3b0','#006b3c','#ed872d','#e30022','#fff600','#a67b5b','#4b3621','#1e4d2b','#a3c1ad','#c19a6b','#efbbcc','#78866b','#ffef00','#ff0800','#e4717a',
'#00bfff','#592720','#c41e3a','#0c9','#960018','#d70040','#eb4c42','#ff0038','#ffa6c9','#b31b1b','#99badd','#ed9121','#062a78','#92a1cf','#ace1af','#007ba7','#2f847c','#b2ffff','#4997d0',
'#de3163','#ec3b83','#007ba7','#2a52be','#6d9bc3','#007aa5', '#e03c31','#a0785a','#fad6a5','#36454f','#e68fac','#dfff00','#7fff00','#de3163','#ffb7c5','#cd5c5c','#de6fa1','#a8516e','#aa381e',
'#7b3f00','#d2691e','#ffa700','#98817b','#e34234','#d2691e','#e4d00a','#fbcce7','#0047ab','#d2691e','#6f4e37','#9bddff','#f88379','#002e63','#8c92ac','#b87333','#da8a67','#ad6f69','#cb6d51',
'#ff3800','#ff7f50','#f88379','#808080','#893f45','#fbec5d','#b31b1b','#fff8dc','#fff8e7','#ffbcd9','#fffdd0', '#dc143c','#be0032', '#0ff','#00b7eb', '#ffff31', '#f0e130', '#00008b',
'#654321','#5d3954', '#a40000', '#08457e','#986960', '#cd5b45', '#008b8b', '#536878', '#b8860b',  '#a9a9a9', '#013220', '#00416a','#1a2421', '#bdb76b', '#483c32', '#734f96', '#8b008b','#556b2f',
'#ff8c00','#9932cc', '#779ecb', '#03c03c', '#966fd6', '#c23b22', '#e75480','#039', '#872657', '#8b0000','#e9967a', '#560319', '#8fbc8f', '#3c1414', '#483d8b', '#2f4f4f','#177245', '#918151',
'#ffa812','#483c32', '#cc4e5c', '#00ced1', '#9400d3','#9b870c', '#00703c', '#555', '#d70a53', '#a9203e', '#ef3038','#e9692c', '#da3287', '#fad6a5','#b94e48', '#704241', '#c154c1', '#004b49',
'#ffcba4','#ff1493','#843f5b', '#f93', '#00bfff', '#66424d', '#1560bd', '#c19a6b', '#edc9af', '#696969','#1e90ff', '#d71868','#85bb65', '#967117', '#00009c', '#e1a95f', '#555d50', '#c2b280', 
'#614051','#f0ead6', '#1cac78', '#1034a6', '#7df9ff', '#ff003f', '#0ff', '#0f0', '#6f00ff', '#f4bbff', '#cf0', '#bf00ff', '#3f00ff','#8f00ff', '#ff0', '#50c878', '#b48395', '#96c8a2', '#c19a6b', 
'#801818','#b53389','#2c1608','#00FF00','#f400a1', '#e5aa70', '#4d5d53', '#4f7942', '#ff2800', '#6c541e', '#ce2029', '#b22222','#ff0', '#9acd32','#e25822', '#fc8eac', '#f7e98e', '#eedc82', 
'#fffaf0','#ffbf00', '#ff1493', '#cf0','#efcc00', '#ffd300','#ff004f', '#014421', '#228b22', '#a67b5b', '#0072bb', '#86608e', '#cf0', '#c72c48','#009f6b', '#00a550', '#f64a8a', '#f0f', '#c154c1',
'#c74375','#e48400','#fefe33','#0014a8','#66b032', '#adff2f','#dcdcdc', '#e49b0f', '#f8f8ff', '#b06500', '#6082b6', '#e6e8fa', '#d4af37', '#ffd700','#bebebe', '#0f0', '#a99a86',
'#00ff7f','#663854', '#446ccf', '#5218fa', '#e9d66b', '#3fff00', '#c90016','#008000', '#00a877', '#da9100', '#808000', '#df73ff', '#f400a1', '#f0fff0', '#007fbf', '#49796b', '#ff1dce',
'#ff69b4','#355e3b','#71a6d2', '#fcf75e', '#002395', '#b2ec5d', '#138808','#cd5c5c', '#e3a857','#6f00ff', '#00416a', '#4b0082', '#002fa7', '#ff4f00', '#ba160c', '#c0362c', '#5a4fcf','#a8e4a0',
'#465945','#f4f0ec', '#009000', '#fffff0', '#00a86b', '#f8de7e', '#d73b3e', '#a50b5e', '#343434','#ffdf00', '#daa520', '#fada5e', '#bdda57', '#29ab87', '#4cbb17', '#7c1c05', '#c3b091',
'#f0e68c','#e8000d','#996515', '#fcc200', '#b57edc', '#c4c3d0', '#9457eb', '#ee82ee', '#e6e6fa', '#fbaed2', '#967bb6', '#fba0e3','#ccf', '#fff0f5','#e6e6fa', '#7cfc00', '#fff700',
'#fffacd','#e3ff00', '#1a1110', '#fdd5b1', '#add8e6','#a9ba9d', '#cf1020', '#b5651d', '#e66771', '#f08080', '#93ccea', '#f56991', '#e0ffff', '#f984ef', '#fafad2','#26619c', '#fefe22',
'#d3d3d3','#90ee90', '#f0e68c', '#b19cd9', '#ffb6c1', '#e97451','#ffa07a', '#f99','#087830', '#d6cadd', '#20b2aa', '#87cefa', '#789', '#b38b6d', '#e68fac', '#ffffe0', '#c8a2c8', '#bfff00',
'#BFD0CA','#32cd32', '#0f0', '#9dc209', '#195905', '#faf0e6', '#c19a6b', '#6ca0dc', '#534b4f', '#e62020', '#f0f', '#ca1f7b', '#ff0090', '#aaf0d1', '#f8f4ff', '#c04000', '#fbec5d','#6050dc',
'#0bda51','#979aaa', '#ff8243', '#74c365', '#880085', '#c32148', '#800000','#b03060', '#e0b0ff', '#915f6d', '#ef98aa', '#73c2fb', '#e5b73b', '#6da', '#e2062c','#af4035', '#f3e5ab','#035096',
'#1c352d','#dda0dd', '#ba55d3', '#0067a5', '#9370db','#bb3385', '#aa4069', '#3cb371', '#7b68ee', '#c9dc87', '#00fa9a', '#674c47', '#48d1cc','#79443b', '#d9603b', '#c71585','#f8b878',
'#f8de7e','#fdbcb4', '#191970', '#004953','#ffc40c', '#3eb489', '#f5fffa', '#98ff98', '#ffe4e1', '#faebd7', '#967117', '#73a9c2', '#ae0c00', '#addfad','#30ba8f', '#997a8d', '#18453b']
}


today = datetime.datetime.now().strftime('%d-%b-%y')


object_dict['template_doc_string'] = {
    'Process':"D Organziation",
    'Categorization':'Reference String',
    'Word':"template_doc_string",
    'Definition':"Default Structure for Python Doc Strings",
    'publish':0,
    'python_object':f'''

    Definition of Function

    Parameters:
        List of Parameters

    Returns:
        Object Type

    date_created:{today}
    date_last_modified: {today}
    classification:TBD
    sub_classification:TBD
    usage:
        Example Function Call
'''
}

object_dict['emoji_dict'] = {
    'Process':"Visualization",
    'Categorization':'Reference List',
    'Word':"emoji_dict",
    'Definition':"A Dictionary of Emojis which can be utilized for the purpose Visualization",
    'publish':0,
    'python_object':{
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
}}