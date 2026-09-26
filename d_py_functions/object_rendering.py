'''
module_word: object_rendering
module_definition: Functions that transform Python objects or structured data into human-readable, formatted representations or documents without changing the underlying meaning of the data. Rendering is the process of converting data or an object from its internal structure into a visual or formatted output suitable for viewing, interpretation, or sharing.

'''
import pandas as pd

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def txt_to_python(file_name,encoding="utf-8"):

    
    '''
    Definition: 
        Function Used to Import .txt or .py File into Python.
    Parameters: 
        file_name(str): Name of File, including path location for import
        encoding(str): Encoding to be applied by With Open call. Default is utf-8.

    Returns:
        Dataframe
    Date Created:
        3-Dec-25
    Date Last Modified:
        3-Dec-25
    Process:
        OS Folder Management
    Categorization:
        File Management
    usage:
        location = '/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions/DFProcessing.py'
        file = TextFileImport(location)
    
    '''

    with open(file_name, "r", encoding=encoding) as file:
        data = file.read()
    
    return data

def export_formatted_df_to_single_xlsx(
    df,
    file_name,
    sheet_name="Sheet1",
    index=False,
    freeze_header=True,
    column_formats=None,  # dict: {col_name: format_type}
):
    '''
    Definition:
        Function Used to Simplify the required formating to a excel output from Python to Excel
    Parameters:
        df(dataframe): Any Dataframe
        file_name(str): Name of Excel Output File
        sheet_name(str): Name of Sheet to be generated in Excel
        index(bool): Include DF index in output as default action
        freeze_header(bool): Freeze Excel File Column Header as default action
        column_foratas(dict): Dictionary to format specific columns based on desired type, options include 
    Returns:
        TBD
    Date Created:
        28-Aug-26
    Date Last Modified:
        28-Aug-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        TBD
    Notes:
        None
    Required Functions:
        None
    
    
    '''
    
    if not file_name.endswith(".xlsx"):
            file_name += ".xlsx"
            
    with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=index, sheet_name=sheet_name)

        workbook = writer.book
        ws = writer.sheets[sheet_name]

        if freeze_header:
            ws.freeze_panes(1, 0)
        
        # ---- Base format (default for all cells) ----
        base_format = workbook.add_format({
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter'
        })

        # ---- Predefined special formats ----
        format_map = {
            "acct": workbook.add_format({'num_format': '0'}),
            "date": workbook.add_format({'num_format': 'yyyy-mm-dd'}),
            "money": workbook.add_format({'num_format': '$#,##0.00'}),
        }

        # ---- Auto-size columns ----
        for col_idx, col in enumerate(df.columns):
            max_len = min(
                80,
                max(df[col].astype(str).map(len).max(), len(col)) + 2
            )

            # Check if column has a special format
            if column_formats and col in column_formats:
                fmt_key = column_formats[col]
                fmt = format_map.get(fmt_key, base_format)
            else:
                fmt = base_format

            ws.set_column(col_idx, col_idx, max_len, fmt)





def visualize_dataframe_in_notebook(
    df,
    show_as_number=[],
    show_as_currency=[],
    show_as_currency_without_dec=[],
    show_as_percentage=[],
):
    
    '''
    
    Definition:
        Function which applies some modest formating to a Dataframe to increase visual astetic.

    Parameters:
        df (dataframe): Any DataFrame
        show_as_number(list): List of Columns which are to be displayed as format :,.2f
        show_as_currency(list): List of Columns which are to be displayed as format $:,.2f
        show_as_currency_without_dec(list): List of Columns which are to be displayed as format $:,.0f
        show_as_percentage(list): List of Columns which are to be displayed as format :.2f%        
        

    Returns:
        Object Type

    date_created: 27-Aug-26
    date_last_modified: 27-Aug-26
    classification:TBD
    sub_classification:TBD
    usage:
        visualize_dataframe_in_notebook(df)

    '''

    formats = {}
    
    for val in show_as_number:
        formats[val] = '{:,.2f}'
    
    for val in show_as_currency_without_dec:
        formats[val] = '${:,.0f}'

    for val in show_as_currency:
        formats[val] = '${:,.2f}'

    for val in show_as_percentage:
        formats[val] = '{:.2f}%'

    
    #df = df.replace(r'\$', r'\\$', regex=True)
 
    styled_df = (
        df.style
        .hide(axis='index')
        .format(formats,escape='html')
        .set_table_styles([
              {'selector': 'table',
               'props': [('border-collapse', 'collapse')]},
              {'selector': 'th',
               'props': [('border', '1px solid black'),
                         ('padding', '5px'),
                         ('text-align', 'center'),
                         ('vertical-align', 'middle'),
                         ('white-space', 'normal')]},
              {'selector': 'td',
               'props': [('border', '1px solid black'),
                         ('padding', '5px'),
                        ('text-align', 'center'),
                         ('vertical-align', 'middle'),
                         ('white-space', 'normal')]}
          ])
    )

    display(styled_df)

def print_dict(d, indent=0):
    '''
    Definition:
        Print Text from Dictionary into a structured output in Console.
    
    Parameters:
        d(dict): Any Dictionary.
        indent(float): Indent Required
    
    Returns:
        None. Prints Structured Text in Console.
    
    Date Created:
        28-Aug-26
        
    Date Last Modified:
        28-Aug-26
    
    Process:
        TBD
    
    Categorization:
        TBD
    
    Usage:
        TBD
    
    Notes:
        Developed for usage with Machine Learning Project. 
    
    Required Functions:
        None
    
    '''
    for key, value in d.items():
        prefix = "    " * indent

        if isinstance(value, dict):
            print(f"\n{prefix}{key}")
            print_dict(value, indent + 1)

        elif isinstance(value, list):
            print(f"\n{prefix}{key}")
            for item in value:
                print(f"{prefix}    - {item.strip()}")

        else:
            print(f"\n{prefix}{key}")
            print(f"{prefix}    {str(value).strip()}")

def create_short_form_ml_project_dict(df=None):
    '''
    Definition:
        Creates a formatted text representation of a wireframe dictionary for the Short Form ML Project Process.

    Parameters:
        df(dataframe): Optional. DataFrame containing the Short Form ML Project definitions. If not provided, the knowledge base is loaded directly.

    Returns:
        str: Formatted text containing a valid Python dictionary assignment that can be printed, displayed, or saved.

    Date Created:
        28-Aug-26

    Date Last Modified:
        09-Sep-26

    Process:
        TBD

    Categorization:
        Object Rendering

    Usage:
        short_form_text = create_short_form_ml_project_dict()
        
    Notes:
        This function is slightly unique as it takes the DataFrame directly and converts it into a formatted dictionary representation. Solely used for the Short Form ML Project Process.

    Required Functions:
        None
    '''

    if df is None:
        knowledge_base_df = pd.read_excel(
            '/Users/derekdewald/Documents/Python/Github_Repo/'
            'Streamlit/Data/knowledge_base.xlsx'
        )

        df = knowledge_base_df[
            knowledge_base_df['Process'] == 'Short Form ML Project'
        ].copy()

    lines = [
        "short_form_dict = {"
    ]

    for word in df['Word'].unique():

        temp_df = df[
            df['Word'] == word
        ]

        ps = temp_df[
            temp_df['Categorization'] == 'Process Step'
        ]['Definition'].item()

        ex = temp_df[
            temp_df['Categorization'] == 'Example'
        ]['Definition'].item()

        lines.append(
            f"    {repr(word)}: {{"
        )

        lines.append(
            f"        'Guidance': {repr(ps)},"
        )

        lines.append(
            f"        'Example': {repr(ex)},"
        )

        lines.append(
            "        'Project Requirement': ''"
        )

        lines.append(
            "    },"
        )

    lines.append(
        "}"
    )

    text = "\n".join(lines)
    print(text)

    return text


def create_machine_learning_project_baseline():
    
    '''
    Definition:
        Create Baseline Formatted Machine Learning Dictionary for purposes of copy and pasting into new Project.
        
    Parameters:
        None
    Returns:
        dict
    Date Created:
        09-Sep-26
    Date Last Modified:
        09-Sep-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        TBD
    Notes:
        Created for simplificity.
    Required Functions:
        None
    
    '''

    base_dict,base_df = create_base_ml_dictionary_from_df()
    return create_baseline_dictionary_for_project(base_dict)

def create_base_ml_dictionary_from_df():
    '''
    '''
    def ml_dict_to_df(base_dict):
        records = []
        for process, details in base_dict.items():
            process_steps = details.get('Required Process Steps', {})
            # If process has required steps
            if process_steps:
                for step, step_details in process_steps.items():
                    records.append({
                        'PROCESS': process,
                        'STEP OBJECTIVE': details.get('Step Objective'),
                        'PROJECT REQUIREMENT': details.get('Project Requirement'),
                        'REQUIRED PROCESS STEP': step,
                        'PROCESS STEP OBJECTIVE': step_details.get('Step Objective'),
                        'PROCESS STEP PROJECT REQUIREMENT': step_details.get('Project Requirements'),
                        'PROCESS STEP STATUS': step_details.get('Status'),
                        'PROJECT SPECIFIC DELIVERABLE(S)': details.get('Project Specific Deliverable(s)'),
                        'OBJECT(S)': details.get('object(s)'),
                        'STATUS': details.get('Status')})
            else:
                records.append({
                    'PROCESS': process,
                    'STEP OBJECTIVE': details.get('Step Objective'),
                    'PROJECT REQUIREMENT': details.get('Project Requirement'),
                    'REQUIRED PROCESS STEP': None,
                    'PROCESS STEP OBJECTIVE': None,
                    'PROCESS STEP PROJECT REQUIREMENT': None,
                    'PROCESS STEP STATUS': None,
                    'PROJECT SPECIFIC DELIVERABLE(S)': details.get('Project Specific Deliverable(s)'),
                    'OBJECT(S)': details.get('object(s)'),
                    'STATUS': details.get('Status')})
        return pd.DataFrame(records)
        
    df = pd.read_excel('/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx')

    temp_df = df[
        (df['Process'].str.contains('Machine Learning Lifecycle'))&
        (df['Source']!='LVL2')&
        (df['Word']!='Definition')
    ]
    base_dict = {}

    for index,row in temp_df.iterrows():
        if row['Source']=='Knowledge Base':
            word = row['Word']
            definition = row['Definition'] 
            try:
                process = {}
                for index,row in temp_df[temp_df['Categorization']==word][['Word','Definition']].iterrows():
                    process[row['Word']] = {'Step Objective':row['Definition'],'Project Requirements':'Not Defined','Status':'Not Started'}
            except:
                process = {}
        
            base_dict[word] = {
                'Step Objective':definition,
                'Project Requirement':"Not Defined",
                "Required Process Steps":process,
                "Project Specific Deliverable(s)":[],
                'object(s)':[],
                'Status':"Not Started"
            }
        else:
            pass

    base_df = ml_dict_to_df(base_dict)
    
    return base_dict,base_df

def create_baseline_dictionary_for_project(
    d,
    dict_name="project_dict",
    blank_keys=None
):
    '''
    Definition:
        Creates a formatted text representation of a dictionary that
        can be used in a Notebook to track project completion.
        Note that the ML Project requires an intermediary step because
        the DataFrame is first processed in a unique way.

    Parameters:
        d(dict):
            Source dictionary used to create the baseline dictionary.

        dict_name(str):
            Name assigned to the formatted dictionary.
            Defaults to 'project_dict'.

        blank_keys(list):
            Optional list of keys whose existing values should be
            replaced with blank strings for manual completion.

    Returns:
        str:
            Formatted text containing a valid Python dictionary
            assignment.

    Date Created:
        09-Sep-26

    Date Last Modified:
        09-Sep-26

    Process:
        TBD

    Categorization:
        Object Rendering

    Usage:
        baseline_text = create_baseline_dictionary_for_project(
            my_dict,
            dict_name='ml_project_dict',
            blank_keys=[
                'Project Requirement',
                'Project Requirements'
            ]
        )

        print(baseline_text)

    Notes:
        Recursively processes nested dictionaries and lists.
        Keys identified in blank_keys are returned with blank values.

    Required Functions:
        None
    '''

    if blank_keys is None:
        blank_keys = []

    def format_value(value, indent=0):

        # ---------------------------------------------
        # Dictionary
        # ---------------------------------------------
        if isinstance(value, dict):

            lines = ["{"]

            items = list(value.items())

            for key, item in items:

                prefix = "    " * (indent + 1)

                # Blank fields intended for completion
                if key in blank_keys:

                    formatted_value = "''"

                else:

                    formatted_value = format_value(
                        item,
                        indent + 1
                    )

                lines.append(
                    f"{prefix}{repr(key)}: "
                    f"{formatted_value},"
                )

            lines.append(
                "    " * indent + "}"
            )

            return "\n".join(lines)

        # ---------------------------------------------
        # List
        # ---------------------------------------------
        elif isinstance(value, list):

            if len(value) == 0:
                return "[]"

            lines = ["["]

            for item in value:

                prefix = "    " * (indent + 1)

                formatted_value = format_value(
                    item,
                    indent + 1
                )

                lines.append(
                    f"{prefix}{formatted_value},"
                )

            lines.append(
                "    " * indent + "]"
            )

            return "\n".join(lines)

        # ---------------------------------------------
        # NaN / None
        # ---------------------------------------------
        elif value is None:

            return "None"

        else:

            try:
                if pd.isna(value):
                    return "None"
            except:
                pass

            # -----------------------------------------
            # Scalar
            # -----------------------------------------
            return repr(value)

    # -------------------------------------------------
    # Create final formatted text object
    # -------------------------------------------------
    formatted_dict = format_value(
        d,
        indent=0
    )

    text = f"{dict_name} = {formatted_dict}"
    print(text)

    return text

def visualize_dict_as_docx(
    data,
    output_file,
    split_pages=False
):
    '''
    Definition:
        Convert a Python Dictionary Object into a visually appealing .docx document. 
        Primarily used to convert Short Form ML Projects or Machine Learning Projects to Stakeholders.
    
    Parameters:
        data(dict): Dictionary
        output_file(str): Name of docx file.
        split_pages(bool): True or False Flag to determine whether each 1st level key will be generated immediately after or split to a new page.
    
    Returns:
        docx
    
    Date Created:
        28-Aug-26
    
    Date Last Modified:
        28-Aug-26
    
    Process:
        TBD
    
    Categorization:
        TBD
    
    Usage:
        visualize_dict_as_docx(short_form_dict,'text.docx')
        visualize_dict_as_docx(base_dict,'text1.docx',True)
    
    Notes:
        100% Created by GPT.
    
    Required Functions:
        None
    
    '''
    
    document = Document()

    # =========================================================
    # PAGE SETUP
    # =========================================================
    section = document.sections[0]

    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)

    normal_style = document.styles["Normal"]
    normal_style.font.name = "Aptos"
    normal_style.font.size = Pt(10)

    # =========================================================
    # HELPER FUNCTIONS
    # =========================================================

    def clean_value(value, default="To Be Defined"):

        if value is None:
            return default

        try:
            if pd.isna(value):
                return default
        except:
            pass

        value = str(value).strip()

        if not value:
            return default

        return value


    def add_text(
        paragraph,
        text,
        size=10,
        bold=False,
        color="404040",
        italic=False
    ):

        run = paragraph.add_run(str(text))

        run.font.name = "Aptos"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = RGBColor.from_string(color)

        return run


    def set_cell_shading(cell, fill):

        tcPr = cell._tc.get_or_add_tcPr()

        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), fill)

        tcPr.append(shd)


    def set_cell_margins(
        cell,
        top=120,
        start=150,
        bottom=120,
        end=150
    ):

        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()

        tcMar = tcPr.first_child_found_in("w:tcMar")

        if tcMar is None:

            tcMar = OxmlElement("w:tcMar")
            tcPr.append(tcMar)

        for margin_name, value in {
            "top": top,
            "start": start,
            "bottom": bottom,
            "end": end
        }.items():

            node = tcMar.find(
                qn(f"w:{margin_name}")
            )

            if node is None:

                node = OxmlElement(
                    f"w:{margin_name}"
                )

                tcMar.append(node)

            node.set(
                qn("w:w"),
                str(value)
            )

            node.set(
                qn("w:type"),
                "dxa"
            )


    def set_cell_border(
        cell,
        color="BFBFBF",
        size="6"
    ):

        tcPr = cell._tc.get_or_add_tcPr()

        borders = tcPr.first_child_found_in(
            "w:tcBorders"
        )

        if borders is None:

            borders = OxmlElement("w:tcBorders")
            tcPr.append(borders)

        for edge in [
            "top",
            "left",
            "bottom",
            "right"
        ]:

            element = borders.find(
                qn(f"w:{edge}")
            )

            if element is None:

                element = OxmlElement(
                    f"w:{edge}"
                )

                borders.append(element)

            element.set(
                qn("w:val"),
                "single"
            )

            element.set(
                qn("w:sz"),
                size
            )

            element.set(
                qn("w:color"),
                color
            )


    def add_horizontal_line(paragraph, color="D9D9D9"):

        p = paragraph._p
        pPr = p.get_or_add_pPr()

        pBdr = OxmlElement("w:pBdr")

        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "4")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), color)

        pBdr.append(bottom)
        pPr.append(pBdr)

    # =========================================================
    # RECURSIVE CONTENT RENDERER
    #
    # Everything is rendered into the SAME component cell.
    # No additional outer boxes are created.
    # =========================================================

    def render_content(
        content,
        cell,
        level=1
    ):

        items = list(content.items())

        for item_number, (key, value) in enumerate(
            items,
            start=1
        ):

            # =================================================
            # NESTED DICTIONARY
            # =================================================
            if isinstance(value, dict):

                # ---------------------------------------------
                # LEVEL 1 NESTED DICTIONARY
                #
                # e.g. "Required Process Steps"
                # ---------------------------------------------
                if level == 1:

                    p = cell.add_paragraph()

                    p.paragraph_format.space_before = Pt(8)
                    p.paragraph_format.space_after = Pt(5)

                    set_cell_shading_paragraph(
                        p,
                        "EAF2F8"
                    )

                    add_text(
                        p,
                        key.upper(),
                        size=10,
                        bold=True,
                        color="1F4E79"
                    )

                # ---------------------------------------------
                # DEEPER NESTED DICTIONARY
                #
                # e.g. "Business Requirements"
                # ---------------------------------------------
                else:

                    p = cell.add_paragraph()

                    p.paragraph_format.left_indent = Inches(
                        0.15 * (level - 1)
                    )

                    p.paragraph_format.space_before = Pt(6)
                    p.paragraph_format.space_after = Pt(2)

                    add_text(
                        p,
                        key,
                        size=10,
                        bold=True,
                        color="303030"
                    )

                # Recurse into same cell
                render_content(
                    value,
                    cell,
                    level + 1
                )

            # =================================================
            # LIST
            # =================================================
            elif isinstance(value, list):

                p = cell.add_paragraph()

                p.paragraph_format.space_before = Pt(5)
                p.paragraph_format.space_after = Pt(2)

                p.paragraph_format.left_indent = Inches(
                    0.15 * max(level - 1, 0)
                )

                add_text(
                    p,
                    key,
                    size=10,
                    bold=True,
                    color="1F4E79"
                )

                if value:

                    for list_item in value:

                        p = cell.add_paragraph(
                            style="List Bullet"
                        )

                        p.paragraph_format.left_indent = Inches(
                            0.2 * level
                        )

                        add_text(
                            p,
                            clean_value(list_item),
                            size=10
                        )

                else:

                    p = cell.add_paragraph()

                    p.paragraph_format.left_indent = Inches(
                        0.15 * level
                    )

                    add_text(
                        p,
                        "None Defined",
                        size=10,
                        color="808080"
                    )

            # =================================================
            # SCALAR
            # =================================================
            else:

                p = cell.add_paragraph()

                p.paragraph_format.left_indent = Inches(
                    0.15 * max(level - 1, 0)
                )

                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(1)

                add_text(
                    p,
                    key,
                    size=10,
                    bold=True,
                    color="1F4E79"
                )

                p = cell.add_paragraph()

                p.paragraph_format.left_indent = Inches(
                    0.15 * max(level - 1, 0)
                )

                p.paragraph_format.space_after = Pt(4)

                add_text(
                    p,
                    clean_value(value),
                    size=10,
                    color="404040"
                )

            # -------------------------------------------------
            # Light divider between entries
            # -------------------------------------------------
            if item_number < len(items):

                # Only use visible divider for deeper items
                if level >= 2:

                    p = cell.add_paragraph()

                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(1)

                    add_horizontal_line(
                        p,
                        color="E7E6E6"
                    )


    # =========================================================
    # PARAGRAPH SHADING
    # Used for second-level subsection headers
    # =========================================================

    def set_cell_shading_paragraph(
        paragraph,
        fill
    ):

        pPr = paragraph._p.get_or_add_pPr()

        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), fill)

        pPr.append(shd)

    # =========================================================
    # FIRST LEVEL COMPONENTS
    # =========================================================

    items = list(data.items())

    for number, (
        component_name,
        component_value
    ) in enumerate(
        items,
        start=1
    ):

        # -----------------------------------------------------
        # ONE TABLE = ONE COMPONENT BOX
        # -----------------------------------------------------
        component_table = document.add_table(
            rows=2,
            cols=1
        )

        component_table.alignment = (
            WD_TABLE_ALIGNMENT.CENTER
        )

        component_table.autofit = True

        header_cell = component_table.cell(
            0,
            0
        )

        content_cell = component_table.cell(
            1,
            0
        )

        # -----------------------------------------------------
        # Outer borders
        # -----------------------------------------------------
        set_cell_border(
            header_cell,
            color="1F4E79",
            size="8"
        )

        set_cell_border(
            content_cell,
            color="BFBFBF",
            size="6"
        )

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------
        set_cell_shading(
            header_cell,
            "1F4E79"
        )

        set_cell_margins(
            header_cell,
            top=140,
            bottom=140,
            start=160,
            end=160
        )

        p = header_cell.paragraphs[0]

        add_text(
            p,
            f"{number}. {component_name}",
            size=14,
            bold=True,
            color="FFFFFF"
        )

        # -----------------------------------------------------
        # Component content cell
        # -----------------------------------------------------
        set_cell_margins(
            content_cell,
            top=100,
            bottom=120,
            start=160,
            end=160
        )

        # Remove the automatically created empty paragraph
        # visually by minimizing spacing
        first_p = content_cell.paragraphs[0]

        first_p.paragraph_format.space_before = Pt(0)
        first_p.paragraph_format.space_after = Pt(0)

        # -----------------------------------------------------
        # Render everything inside same component box
        # -----------------------------------------------------
        if isinstance(
            component_value,
            dict
        ):

            render_content(
                component_value,
                content_cell,
                level=1
            )

        elif isinstance(
            component_value,
            list
        ):

            for item in component_value:

                p = content_cell.add_paragraph(
                    style="List Bullet"
                )

                add_text(
                    p,
                    clean_value(item),
                    size=10
                )

        else:

            p = content_cell.add_paragraph()

            add_text(
                p,
                clean_value(component_value),
                size=10
            )

        # -----------------------------------------------------
        # Component separation
        # -----------------------------------------------------
        if number < len(items):

            if split_pages:

                document.add_page_break()

            else:

                p = document.add_paragraph()
                p.paragraph_format.space_after = Pt(8)

    # =========================================================
    # SAVE
    # =========================================================

    document.save(output_file)