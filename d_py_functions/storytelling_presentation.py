from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont,ImageOps
from pathlib import Path


from io import BytesIO
import pandas as pd
import numpy as np
import math
import os
import re
from IPython.display import display


color_dict = {
    'white': '#FFFFFF',
    'navy': '#10263E',
    'teal': '#236C91',
    'muted': '#536477',
    'pale': '#E8EDF3',
    'line': '#C6D1DE',
    'light_teal': '#DDE8EC',
    'border':"#F0F2F3",
    'neutral':"#F0F2F3"
    
}

def frame_the_question(
    question,
    context,
    why,
    scope,
    folder_location=None,
    title='Frame the Question',
    header='',
    left_footer='',
    right_footer='',
    return_value='',
    context_image=None,
    why_image=None,
    scope_image=None,
    image_size=140,
    color_dict=color_dict
):
    '''
    Definition:
        Creates a 1600x900 presentation slide with a central
        analytical question and three supporting sections.

    Parameters:
        question, context, why, scope:
            Text displayed on the slide.

        context_image, why_image, scope_image:
            Paths to standardized icon images.

        image_size:
            Maximum width and height of each image.

        return_value:
            'file' saves the PNG; otherwise displays the slide.

    Required Functions:
        load_presentation_font
    '''

    folder = (
        Path(folder_location).expanduser()
        if folder_location is not None
        else Path.cwd()
    )

    if not folder.is_dir():
        raise FileNotFoundError(
            f'Folder does not exist: {folder}'
        )

    output = folder / '1_frame_the_question'

    if output.suffix.lower() != '.png':
        output = output.with_suffix('.png')

    W, H = 1600, 900


    navy  = color_dict['navy']
    teal  = color_dict['teal']
    muted = color_dict['muted']
    pale  = color_dict['pale']
    line  = color_dict['line']
    white = color_dict['white']

    img = Image.new('RGB', (W, H), white)
    d = ImageDraw.Draw(img)

    
    def load_font(size, bold=False):
        return load_presentation_font(size, bold=bold)

    def wrap(text, font, max_width):
        words = str(text).split()
        lines, current = [], ''

        for word in words:
            trial = (current + ' ' + word).strip()

            if d.textbbox((0, 0), trial, font=font)[2] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines

    def paragraph(
        text, x, y, width,
        size=26,
        fill=navy,
        bold=False,
        spacing=10,
        max_lines=None
    ):
        font = load_font(size, bold)
        lines = wrap(text, font, width)

        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]

            while (
                lines[-1]
                and d.textbbox(
                    (0, 0),
                    lines[-1] + '…',
                    font=font
                )[2] > width
            ):
                lines[-1] = lines[-1][:-1]

            lines[-1] += '…'

        step = size + spacing

        for i, line_text in enumerate(lines):
            d.text(
                (x, y + i * step),
                line_text,
                font=font,
                fill=fill
            )

        return y + len(lines) * step

    # ---------------------------------------------
    # Image helper
    # ---------------------------------------------
    def place_image(image_path, cx, cy, size):
        if image_path is None:
            return

        path = Path(image_path).expanduser()

        # Allow paths without extensions
        if not path.is_file() and not path.suffix:
            for extension in ('.png', '.jpg', '.jpeg', '.webp'):
                candidate = path.with_suffix(extension)
                if candidate.is_file():
                    path = candidate
                    break

        if not path.is_file():
            raise FileNotFoundError(
                f'Presentation image not found: {image_path}'
            )

        icon = Image.open(path).convert('RGBA')

        # Scale proportionally, including enlargement
        scale = min(
            size / icon.width,
            size / icon.height
        )

        new_width = max(1, round(icon.width * scale))
        new_height = max(1, round(icon.height * scale))

        icon = icon.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        x = round(cx - icon.width / 2)
        y = round(cy - icon.height / 2)

        img.paste(icon, (x, y), icon)

    # ---------------------------------------------
    # Header
    # ---------------------------------------------
    d.text(
        (95, 62),
        header,
        font=load_font(22, True),
        fill=teal
    )

    paragraph(
        title, 95, 111, 1300,
        54, navy, True
    )

    d.line(
        (95, 215, 1505, 215),
        fill=line,
        width=3
    )

    # ---------------------------------------------
    # Central analytical question
    # ---------------------------------------------
    d.rounded_rectangle(
        (95, 264, 870, 746),
        radius=28,
        fill=pale
    )

    d.rounded_rectangle(
        (95, 264, 105, 746),
        radius=4,
        fill=teal
    )

    d.text(
        (147, 320),
        'Central Analytical Question',
        font=load_font(21, True),
        fill=teal
    )

    paragraph(
        question,
        147, 390, 650,
        32, navy, True,
        spacing=14,
        max_lines=6
    )

    # ---------------------------------------------
    # Supporting sections
    # ---------------------------------------------
    sections = [
        ('CONTEXT', context, context_image, 323),
        ('WHY IT MATTERS', why, why_image, 482),
        ('SCOPE', scope, scope_image, 641),
    ]

    for label, body, image_path, cy in sections:

        place_image(
            image_path,
            cx=950,
            cy=cy+15,
            size=image_size
        )

        d.text(
            (1040, cy - 40),
            label,
            font=load_font(20, True),
            fill=teal
        )

        paragraph(
            body,
            1040,
            cy - 4,
            435,
            23,
            navy,
            spacing=8,
            max_lines=3
        )

    # ---------------------------------------------
    # Footer
    # ---------------------------------------------
    d.line(
        (95, 798, 1505, 798),
        fill=line,
        width=2
    )

    d.text(
        (95, 823),
        left_footer,
        font=load_font(18),
        fill=muted
    )

    d.text(
        (1390, 823),
        right_footer,
        font=load_font(18),
        fill=muted
    )

    if return_value == 'file':
        img.save(output)

    return display(img)



def member_population_overview(
    demographics,
    activity,
    observations,
    folder_location=None,
    images=None,
    title='Understanding the Affected Population',
    header='STRUCTURE THE EVALUATION',
    left_footer='',
    right_footer='',
    return_value='',
    color_dict=color_dict
):
    """
    Create a 1600 x 900 infographic-style presentation slide.

    Parameters
    ----------
    demographics : dict
        Label/value pairs describing member demographics.

    activity : dict
        Label/value pairs describing financial and
        transactional engagement.

    observations : list[str]
        Two to four preliminary observations.

    Returns
    -------
    PIL.Image.Image
    """

    images = images or {}
    
    folder = (
        Path(folder_location).expanduser()
        if folder_location is not None
        else Path.cwd()
    )

    if not folder.is_dir():
        raise FileNotFoundError(
            f'Folder does not exist: {folder}'
        )

    output = (folder / 'member_population_overview').with_suffix('.png')

    W, H = 1600, 900

    navy  = color_dict['navy']
    teal  = color_dict['teal']
    muted = color_dict['muted']
    pale  = color_dict['pale']
    line  = color_dict['line']
    white = color_dict['white']
    
    img = Image.new('RGB', (W, H), white)
    draw = ImageDraw.Draw(img)

    def font(size, bold=False):
        return load_presentation_font(size, bold=bold)

    def wrap(text, fnt, max_width):
        words = str(text).split()
        lines = []
        current = ''

        for word in words:
            trial = (current + ' ' + word).strip()

            if draw.textbbox(
                (0, 0), trial, font=fnt
            )[2] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines

    def paragraph(
        text, x, y, width,
        size=23, color=navy,
        bold=False, spacing=8,
        max_lines=None
    ):
        fnt = font(size, bold)
        lines = wrap(text, fnt, width)

        if max_lines is not None:
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                last = lines[-1]

                while last and draw.textbbox(
                    (0, 0), last + '…', font=fnt
                )[2] > width:
                    last = last[:-1]

                lines[-1] = last + '…'

        for i, text_line in enumerate(lines):
            draw.text(
                (x, y + i * (size + spacing)),
                text_line,
                font=fnt,
                fill=color
            )

        return y + len(lines) * (size + spacing)

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    draw.text(
        (95, 62),
        header,
        font=font(22, True),
        fill=teal
    )

    draw.text(
        (95, 111),
        title,
        font=font(54, True),
        fill=navy
    )

    draw.line(
        (95, 215, 1505, 215),
        fill=line,
        width=3
    )

    def place_image(image_path, cx, cy, size=(85, 85)):
        """
        Insert an external image, preserving its aspect ratio.
    
        cx, cy : centre position on the slide
        size   : maximum image width and height
        """
    
        if image_path is None:
            return
    
        image_path = Path(image_path).expanduser()
    
        if not image_path.is_file():
            raise FileNotFoundError(f'Image not found: {image_path}')
    
        icon = Image.open(image_path).convert('RGBA')
        icon.thumbnail(size, Image.Resampling.LANCZOS)
    
        x = int(cx - icon.width / 2)
        y = int(cy - icon.height / 2)
    
        img.paste(icon, (x, y), icon)
        # --------------------------------------------------
        # Information cards
        # --------------------------------------------------

    def info_card(bounds, heading, values, image_path):
        x1, y1, x2, y2 = bounds
    
        draw.rounded_rectangle(
            bounds,
            radius=22,
            fill=pale
        )
    
        # Insert the supplied image
        place_image(
            image_path,
            cx=x1+65,
            cy=y1+62,
            size=(105, 105)
        )
    
        draw.text(
            (x1+125, y1+46),
            heading,
            font=font(26, True),
            fill=navy
        )

        items = list(values.items())

        if not items:
            return

        # Two-column layout for key indicators
        col_width = (x2-x1-80)//2

        for i, (label, value) in enumerate(items[:8]):
            col = i % 2
            row = i // 2

            x = x1 + 35 + col*(col_width+10)
            y = y1 + 128 + row*79

            paragraph(
                str(value),
                x, y, col_width-8,
                size=29,
                bold=True,
                max_lines=1
            )

            paragraph(
                str(label),
                x, y+39, col_width-8,
                size=17,
                color=muted,
                max_lines=2
            )

    # --------------------------------------------------
    # Demographics and activity
    # --------------------------------------------------

    info_card(
        (95, 250, 785, 605),
        '  Member Demographics',
        demographics,
        images.get('demographics')
    )
    
    info_card(
        (815, 250, 1505, 605),
        '  Member Activity',
        activity,
        images.get('activity')
    )

    # --------------------------------------------------
    # Preliminary observations
    # --------------------------------------------------

    x1, y1, x2, y2 = 95, 625, 1505, 775

    draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=22,
        fill=pale
    )

    place_image(
        images.get('observations'),
        cx=160,
        cy=715,
        size=(120, 120)
    )

    draw.text(
        (215, 650),
        'Preliminary Observations',
        font=font(26, True),
        fill=navy
    )

    observation_list = list(observations)[:4]

    if observation_list:
        col_width = 580

        for i, observation in enumerate(observation_list):
            col = i % 2
            row = i // 2

            x = 220 + col*630
            y = 697 + row*35

            draw.ellipse(
                (x, y+7, x+9, y+16),
                fill=teal
            )

            paragraph(
                observation,
                x+20, y,
                col_width,
                size=17,
                max_lines=2,
                spacing=4
            )

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------

    draw.line(
        (95, 798, 1505, 798),
        fill=line,
        width=2
    )

    draw.text(
        (95, 823),
        left_footer,
        font=font(18),
        fill=muted
    )

    draw.text(
        (1390, 823),
        right_footer,
        font=font(18),
        fill=muted
    )
    
    if return_value == 'file':
        img.save(output)

    return display(img)


def peer_group_introduction(
    groups,
    target_population,
    folder_location=None,
    images=None,
    title="Establishing the Comparison Framework",
    header="",
    subtitle="",
    methodology_note="",
    left_footer="",
    right_footer="",
    return_value="",
    color_dict=color_dict,
    group_image_size=120,
    concept_image_size=115,
):
    """
    Create a 1600 x 900 comparison framework slide.

    groups:
        List of four dictionaries containing:
        - title / name
        - description / definition

    target_population:
        String or dictionary describing the target population.

    images:
        {
            "target": "...",
            "group_1": "...",
            "group_2": "...",
            "group_3": "...",
            "group_4": "...",
            "context": "...",
            "comparison": "...",
            "testing": "...",
            "decision": "..."
        }

    Images can be paths, PIL Images, or None.
    """

    images = images or {}

    W, H = 1600, 900

    WHITE = color_dict["white"]
    NAVY = color_dict["navy"]
    TEAL = color_dict["teal"]
    MUTED = color_dict["muted"]
    PALE = color_dict["pale"]
    LINE = color_dict["line"]

    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    # --------------------------------------------------
    # Fonts
    # --------------------------------------------------

    font_title = load_presentation_font(43, bold=True)
    font_target = load_presentation_font(25, bold=True)
    font_group = load_presentation_font(24, bold=True)
    font_body = load_presentation_font(20)
    font_statement = load_presentation_font(29, bold=True)
    font_concept = load_presentation_font(21, bold=True)
    font_footer = load_presentation_font(15)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def text_width(text, font):
        return draw.textbbox((0, 0), str(text), font=font)[2]

    def center_text(text, cx, y, font, fill):
        draw.text(
            (cx - text_width(text, font) / 2, y),
            str(text),
            font=font,
            fill=fill,
        )

    def wrap_text(text, font, max_width):
        words = str(text).split()
        lines = []
        current = ""

        for word in words:
            candidate = f"{current} {word}".strip()

            if text_width(candidate, font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines

    def place_image(source, cx, cy, size=(120, 120)):
        if source is None:
            return

        if isinstance(source, Image.Image):
            icon = source.copy()
        else:
            path = Path(source)

            if not path.exists():
                if not path.suffix:
                    for ext in (".png", ".webp", ".jpg", ".jpeg"):
                        candidate = Path(str(path) + ext)
                        if candidate.exists():
                            path = candidate
                            break

            if not path.exists():
                return

            icon = Image.open(path)

        icon = icon.convert("RGBA")
        icon = ImageOps.contain(
            icon,
            size,
            method=Image.Resampling.LANCZOS,
        )

        x = int(cx - icon.width / 2)
        y = int(cy - icon.height / 2)

        img.paste(icon, (x, y), icon)

    def group_field(group, keys, default=""):
        if isinstance(group, dict):
            for key in keys:
                if group.get(key):
                    return str(group[key])
            return default

        return str(group)

    # --------------------------------------------------
    # Slide title
    # --------------------------------------------------

    draw.text(
        (70, 49),
        title,
        font=font_title,
        fill=NAVY,
    )

    draw.line(
        (70, 119, 1530, 119),
        fill=LINE,
        width=2,
    )

    # --------------------------------------------------
    # Target population
    # --------------------------------------------------

    target_y1, target_y2 = 142, 225

    draw.rounded_rectangle(
        (70, target_y1, 1530, target_y2),
        radius=15,
        fill=PALE,
    )

    target_text = group_field(
        target_population,
        ["description", "definition", "text", "name", "title"],
    )

    draw.text(
        (100, 155),
        "Target Population",
        font=font_target,
        fill=NAVY,
    )

    draw.text(
        (100, 190),
        target_text,
        font=font_body,
        fill=MUTED,
    )

    # --------------------------------------------------
    # Four comparison groups
    # --------------------------------------------------

    if isinstance(groups, dict):
        group_items = [groups[key] for key in sorted(groups)]
    else:
        group_items = list(groups)

    if len(group_items) != 4:
        raise ValueError("Exactly four comparison groups are required.")

    card_positions = [
        (70, 243, 790, 405),
        (810, 243, 1530, 405),
        (70, 420, 790, 582),
        (810, 420, 1530, 582),
    ]

    for i, (group, rect) in enumerate(
        zip(group_items, card_positions)
    ):
        x1, y1, x2, y2 = rect

        draw.rounded_rectangle(
            rect,
            radius=15,
            fill=WHITE,
            outline=LINE,
            width=2,
        )

        # Large user-provided group image
        image_source = images.get(f"group_{i + 1}")

        if image_source is None and isinstance(group, dict):
            image_source = group.get("image")

        place_image(
            image_source,
            cx=x1 + 95,
            cy=(y1 + y2) / 2,
            size=(group_image_size, group_image_size),
        )

        group_title = group_field(
            group,
            ["title", "name", "label", "group"],
            f"Comparison Group {i + 1}",
        )

        group_description = group_field(
            group,
            ["description", "definition", "purpose", "text"],
        )

        tx = x1 + 180

        draw.text(
            (tx, y1 + 24),
            group_title,
            font=font_group,
            fill=NAVY,
        )

        description_lines = wrap_text(
            group_description,
            font_body,
            max_width=x2 - tx - 25,
        )

        for j, line in enumerate(description_lines[:4]):
            draw.text(
                (tx, y1 + 65 + j * 26),
                line,
                font=font_body,
                fill=MUTED,
            )

    # --------------------------------------------------
    # Change is Relative
    # --------------------------------------------------

    draw.line(
        (70, 603, 1530, 603),
        fill=LINE,
        width=2,
    )

    statement = (
        "Change is relative. Without meaningful reference points, "
        "even measurable change can be misleading."
    )

    center_text(
        statement,
        cx=800,
        y=621,
        font=font_statement,
        fill=NAVY,
    )

    # Four concepts, each with a supplied image
    concepts = [
        (
            "context",
            "Context:",
            "Understand the Significance of Change",
        ),
        (
            "comparison",
            "Comparison:",
            "Identify Differences and Patterns",
        ),
        (
            "testing",
            "Testing:",
            "Evaluate Assumptions and Explanations",
        ),
        (
            "decision",
            "Decision-making:",
            "Support Evidence-Informed Decisions",
        ),
    ]

    concept_centers = [245, 615, 985, 1355]

    for cx, (image_key, label, description) in zip(
        concept_centers, concepts
    ):
        place_image(
            images.get(image_key),
            cx=cx,
            cy=735,
            size=(concept_image_size, concept_image_size),
        )

        # Exactly two lines:
        # 1. Concept name
        # 2. Explanation
        center_text(
            label,
            cx=cx,
            y=803,
            font=font_concept,
            fill=NAVY,
        )

        # Shrink the second line if needed
        available_width = 340
        description_font = load_presentation_font(19)

        while (
            text_width(description, description_font)
            > available_width
        ):
            current_size = description_font.size - 1

            if current_size < 13:
                break

            description_font = load_presentation_font(
                current_size
            )

        center_text(
            description,
            cx=cx,
            y=833,
            font=description_font,
            fill=MUTED,
        )

    # --------------------------------------------------
    # Optional footer
    # --------------------------------------------------

    if left_footer:
        draw.text(
            (70, 878),
            left_footer,
            font=font_footer,
            fill=MUTED,
        )

    if right_footer:
        draw.text(
            (1530 - text_width(right_footer, font_footer), 878),
            right_footer,
            font=font_footer,
            fill=MUTED,
        )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    if return_value == "file":
        output_folder = Path(folder_location or ".")
        output_folder.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_folder / "peer_group_introduction.png"
        )

        img.save(output_path)
        return str(output_path)

    if return_value == "image":
        return img

    display(img)



def engagement_measurement_grid(
    questions,
    folder_location=None,
    title='A Framework for Defining, Understanding and Evaluating Member Engagement',
    subtitle=(
        'Defining core measurement categories and KPIs enables '
        'the establishment of expectations, performance tracking, '
        'and evaluation of changes in engagement. '
        'This results in a structured approach providing a '
        'consistent foundation for assessment.'
    ),
    left_footer='',
    right_footer='',
    color_dict=color_dict,
    return_value=''
):
    """
    Create a dynamic 1600 x 900 measurement framework slide.

    Supports:
        - 1 to 5 question rows
        - 1 to 5 metrics per question
        - Optional images for each metric
        - Automatic image scaling and centering
        - Dynamic title and subtitle positioning
        - Column headings
        - No question numbering

    Metric format:
        {
            'label': 'Deposit Balances',
            'image': '/path/to/deposits.png'
        }

    Returns
    -------
    PIL.Image.Image
    """

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    if not 1 <= len(questions) <= 5:
        raise ValueError(
            'Provide between 1 and 5 questions.'
        )

    if any(
        not 1 <= len(q.get('metrics', [])) <= 5
        for q in questions
    ):
        raise ValueError(
            'Each question must contain 1 to 5 metrics.'
        )

    max_metrics = max(
        len(q['metrics']) for q in questions
    )

    folder = (
        Path(folder_location).expanduser()
        if folder_location is not None
        else Path.cwd()
    )

    if not folder.is_dir():
        raise FileNotFoundError(
            f'Folder does not exist: {folder}'
        )

    output = folder / 'engagement_measurement_grid.png'

    # --------------------------------------------------
    # Canvas and colours
    # --------------------------------------------------

    W, H = 1600, 900

    navy = color_dict['navy']
    teal = color_dict['teal']
    muted = color_dict['muted']
    pale = color_dict['pale']
    line = color_dict['line']
    white = color_dict['white']

    img = Image.new('RGB', (W, H), white)
    draw = ImageDraw.Draw(img)

    # --------------------------------------------------
    # Text helpers
    # --------------------------------------------------

    def font(size, bold=False):
        return load_presentation_font(
            size,
            bold=bold
        )

    def text_width(text, fnt):
        bbox = draw.textbbox(
            (0, 0),
            str(text),
            font=fnt
        )
        return bbox[2] - bbox[0]

    def wrap(text, fnt, max_width):
        words = str(text).split()

        lines = []
        current = ''

        for word in words:
            trial = (current + ' ' + word).strip()

            if text_width(trial, fnt) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines

    def centered_text(
        text,
        cx,
        cy,
        width,
        size=20,
        color=navy,
        bold=False,
        max_lines=3
    ):
        fnt = font(size, bold)
        lines = wrap(text, fnt, width)

        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]

            while (
                last
                and text_width(last + '…', fnt) > width
            ):
                last = last[:-1]

            lines[-1] = last + '…'

        line_height = size + 6
        total_height = len(lines) * line_height

        y = cy - total_height / 2

        for line_text in lines:
            x = cx - text_width(line_text, fnt) / 2

            draw.text(
                (x, y),
                line_text,
                font=fnt,
                fill=color
            )

            y += line_height

    # --------------------------------------------------
    # Image helper
    # --------------------------------------------------

    def render_metric_image(
        image_path,
        cx,
        cy,
        max_width,
        max_height
    ):
        if not image_path:
            return False

        path = Path(image_path).expanduser()

        if not path.is_absolute():
            path = folder / path

        if not path.is_file():
            raise FileNotFoundError(
                f'Metric image not found: {path}'
            )

        with Image.open(path) as source:
            metric_img = source.convert('RGBA')

        # Crop transparent margins where available.
        alpha = metric_img.getchannel('A')
        bbox = alpha.getbbox()

        if bbox:
            metric_img = metric_img.crop(bbox)

        metric_img = ImageOps.contain(
            metric_img,
            (
                max(1, int(max_width)),
                max(1, int(max_height))
            ),
            method=Image.Resampling.LANCZOS
        )

        x = round(cx - metric_img.width / 2)
        y = round(cy - metric_img.height / 2)

        img.paste(
            metric_img,
            (x, y),
            metric_img
        )

        return True

    # --------------------------------------------------
    # Title and subtitle
    # --------------------------------------------------

    title_y = 45
    title_font = font(45, True)
    title_line_height = 53

    title_lines = wrap(
        title,
        title_font,
        1410
    )

    for i, title_line in enumerate(title_lines):
        draw.text(
            (
                95,
                title_y + i * title_line_height
            ),
            title_line,
            font=title_font,
            fill=navy
        )

    # Position subtitle dynamically beneath title.
    subtitle_y = (
        title_y
        + len(title_lines) * title_line_height
        + 16
    )

    subtitle_font = font(20)
    subtitle_line_height = 28

    subtitle_lines = wrap(
        subtitle,
        subtitle_font,
        1390
    )

    for i, subtitle_line in enumerate(
        subtitle_lines
    ):
        draw.text(
            (
                95,
                subtitle_y + i * subtitle_line_height
            ),
            subtitle_line,
            font=subtitle_font,
            fill=muted
        )

    # Position separator dynamically beneath subtitle.
    separator_y = (
        subtitle_y
        + len(subtitle_lines) * subtitle_line_height
        + 20
    )

    draw.line(
        (95, separator_y, 1505, separator_y),
        fill=line,
        width=3
    )

    # --------------------------------------------------
    # Dynamic grid dimensions
    # --------------------------------------------------

    grid_left = 95
    grid_right = 1505

    column_header_y = separator_y + 20

    grid_top = column_header_y + 38
    grid_bottom = 825

    row_count = len(questions)

    row_gap = 10
    column_gap = 10

    grid_width = grid_right - grid_left
    grid_height = grid_bottom - grid_top

    row_height = (
        grid_height - row_gap * (row_count - 1)
    ) / row_count

    if row_height <= 0:
        raise ValueError(
            'Insufficient vertical space for the grid. '
            'Reduce title/subtitle length or question count.'
        )

    # Question column is wider than metric columns.
    question_weight = 1.60

    metric_width = (
        grid_width - column_gap * max_metrics
    ) / (max_metrics + question_weight)

    question_width = metric_width * question_weight

    question_font_size = (
        25 if row_count <= 3
        else 21 if row_count == 4
        else 18
    )

    metric_font_size = (
        17 if max_metrics <= 3
        else 15
    )

    # --------------------------------------------------
    # Column headings
    # --------------------------------------------------

    question_heading = 'GUIDING QUESTIONS'
    metric_heading = 'KEY PERFORMANCE INDICATORS (KPIs)'

    heading_font = font(17, True)

    # Left column heading
    draw.text(
        (
            grid_left
            + (
                question_width
                - text_width(
                    question_heading,
                    heading_font
                )
            ) / 2,
            column_header_y
        ),
        question_heading,
        font=heading_font,
        fill=teal
    )

    # Right columns heading
    metrics_left = (
        grid_left + question_width + column_gap
    )

    metrics_width = grid_right - metrics_left

    draw.text(
        (
            metrics_left
            + (
                metrics_width
                - text_width(
                    metric_heading,
                    heading_font
                )
            ) / 2,
            column_header_y
        ),
        metric_heading,
        font=heading_font,
        fill=teal
    )

    # --------------------------------------------------
    # Render rows
    # --------------------------------------------------

    for row_index, question_data in enumerate(
        questions
    ):

        y1 = (
            grid_top
            + row_index * (row_height + row_gap)
        )
        y2 = y1 + row_height

        # ----------------------------------------------
        # Question card
        # ----------------------------------------------

        qx1 = grid_left
        qx2 = qx1 + question_width

        draw.rounded_rectangle(
            (qx1, y1, qx2, y2),
            radius=16,
            fill=pale
        )

        centered_text(
            question_data['question'],
            cx=(qx1 + qx2) / 2,
            cy=(y1 + y2) / 2,
            width=question_width - 36,
            size=question_font_size,
            bold=True,
            max_lines=4
        )

        # ----------------------------------------------
        # Metric cards
        # ----------------------------------------------

        for metric_index, metric in enumerate(
            question_data['metrics']
        ):

            if isinstance(metric, str):
                label = metric
                image_path = None
            else:
                label = metric.get('label', '')
                image_path = metric.get('image')

            mx1 = (
                qx2 + column_gap
                + metric_index
                * (metric_width + column_gap)
            )

            mx2 = mx1 + metric_width

            draw.rounded_rectangle(
                (mx1, y1, mx2, y2),
                radius=16,
                fill=white,
                outline=line,
                width=2
            )

            cx = (mx1 + mx2) / 2

            # ------------------------------------------
            # Image region
            # ------------------------------------------

            label_height = min(
                24,
                row_height * 0.25
            )

            image_top = y1 + 5
            image_bottom = (
                y2 - label_height - 4
            )

            image_height = max(
                1,
                image_bottom - image_top
            )

            image_cy = (
                image_top + image_bottom
            ) / 2

            if image_path:
                render_metric_image(
                    image_path=image_path,
                    cx=cx,
                    cy=image_cy,
                    max_width=metric_width - 8,
                    max_height=image_height
                )

            # ------------------------------------------
            # Metric label
            # ------------------------------------------

            centered_text(
                label,
                cx=cx,
                cy=y2 - label_height / 2 - 3,
                width=metric_width - 16,
                size=metric_font_size,
                bold=True,
                max_lines=2
            )

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------

    footer_line_y = 840
    footer_text_y = 850

    draw.line(
        (95, footer_line_y, 1505, footer_line_y),
        fill=line,
        width=2
    )

    draw.text(
        (95, footer_text_y),
        left_footer,
        font=font(17),
        fill=muted
    )

    footer_font = font(10)

    draw.text(
        (
            1505 - text_width(
                right_footer,
                footer_font
            ),
            footer_text_y
        ),
        right_footer,
        font=footer_font,
        fill=muted
    )

    if return_value == 'file':
        img.save(output)

    return display(img)


def engagement_comparison_matrix(
    df,
    folder_location=None,
    title="Comparing Observed Performance Across Groups",
    subtitle="",
    left_footer="",
    right_footer="",
    font_path=None,
    row_formats=None,
    return_value="",
    output=None,
    color_dict=color_dict,
):
    """
    Create a 1600 x 900 presentation comparison matrix.

    DataFrame structure:
        Index   = KPI labels
        Columns = Comparison groups
        Values  = Numeric observations

    Formatting:
        '%'   Percentage change (5.2 means 5.2%)
        '$'   Dollar values
        'int' Counts / whole numbers

    Visual rules:
        - Highest observed value: blue border
        - Lowest observed value: navy border
        - Other values: neutral border
        - Ties are highlighted equally
        - Identical values receive no highlighting
        - Missing values are excluded from comparisons
        - Percentage bars show direction and magnitude

    Requires:
        load_presentation_font()
    """

    # --------------------------------------------------
    # Validate data
    # --------------------------------------------------
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(
            "df must be a non-empty DataFrame."
        )

    if df.index.has_duplicates or df.columns.has_duplicates:
        raise ValueError(
            "Row and column labels must be unique."
        )

    df = df.apply(pd.to_numeric, errors="raise")

    row_formats = row_formats or {}

    n_rows, n_cols = df.shape

    if not 1 <= n_rows <= 6:
        raise ValueError(
            "Use between 1 and 6 KPI rows."
        )

    if not 1 <= n_cols <= 7:
        raise ValueError(
            "Use between 1 and 7 comparison columns."
        )

    # --------------------------------------------------
    # Presentation palette
    # --------------------------------------------------
    WHITE = color_dict["white"]
    NAVY = color_dict["navy"]
    TEAL = color_dict["teal"]
    MUTED = color_dict["muted"]
    PALE = color_dict["pale"]
    LINE = color_dict["line"]

    # --------------------------------------------------
    # High-resolution rendering
    # --------------------------------------------------
    SCALE = 2
    W, H = 1600, 900

    img = Image.new(
        "RGB",
        (W * SCALE, H * SCALE),
        WHITE
    )

    draw = ImageDraw.Draw(img)

    def px(value):
        return round(value * SCALE)

    def font(size, bold=False):
        return load_presentation_font(
            px(size),
            bold=bold,
            font_path=font_path
        )

    f_title = font(45, True)
    f_subtitle = font(20)
    f_header = font(17, True)
    f_footer = font(16)

    # --------------------------------------------------
    # Drawing helpers
    # --------------------------------------------------
    def text_width(text, f):
        box = draw.textbbox(
            (0, 0),
            str(text),
            font=f
        )

        return (box[2] - box[0]) / SCALE

    def write(
        text,
        x,
        y,
        f,
        color,
        anchor="lt"
    ):
        draw.text(
            (px(x), px(y)),
            str(text),
            font=f,
            fill=color,
            anchor=anchor
        )

    def rectangle(
        box,
        fill,
        outline=None,
        radius=0,
        stroke=1
    ):
        draw.rounded_rectangle(
            tuple(px(v) for v in box),
            radius=px(radius),
            fill=fill,
            outline=outline,
            width=px(stroke)
        )

    def rule(
        x1,
        y1,
        x2,
        y2,
        color=LINE,
        stroke=1
    ):
        draw.line(
            (
                px(x1),
                px(y1),
                px(x2),
                px(y2)
            ),
            fill=color,
            width=px(stroke)
        )

    def wrap(text, f, max_width):
        lines = []
        current = ""

        for word in str(text).split():

            candidate = (
                current + " " + word
            ).strip()

            if text_width(candidate, f) <= max_width:
                current = candidate

            else:
                if current:
                    lines.append(current)

                current = word

        if current:
            lines.append(current)

        return lines

    def fit_font(
        text,
        max_width,
        initial=31,
        minimum=15,
        bold=True
    ):
        for size in range(
            initial,
            minimum - 1,
            -1
        ):
            f = font(size, bold)

            if text_width(text, f) <= max_width:
                return f

        return font(minimum, bold)

    def centered_wrapped(
        text,
        cx,
        cy,
        max_width,
        initial_size=23,
        max_lines=3,
        color=None,
        bold=True
    ):
        color = color or NAVY

        for size in range(
            initial_size,
            13,
            -1
        ):
            f = font(size, bold)

            lines = wrap(
                text,
                f,
                max_width
            )

            if len(lines) <= max_lines:
                break

        line_height = size + 7

        start_y = (
            cy
            - (len(lines) - 1)
            * line_height / 2
        )

        for k, line_text in enumerate(lines):

            write(
                line_text,
                cx,
                start_y + k * line_height,
                f,
                color,
                anchor="mm"
            )

    # --------------------------------------------------
    # Automatic KPI formatting
    # --------------------------------------------------
    def infer_scale(label):

        label = str(label).lower()

        if re.search(
            r"growth|percent|perc|%|rate|change",
            label
        ):
            return "%"

        if re.search(
            r"count|number|volume",
            label
        ):
            return "int"

        if re.search(
            r"balance|footing|deposit|loan|value|revenue",
            label
        ):
            return "$"

        return "%"

    def format_value(value, scale):

        if pd.isna(value):
            return "N/A"

        value = float(value)

        if scale == "%":
            return (
                f"{value:+,.1f}%"
                if value != 0
                else "0.0%"
            )

        if scale == "$":
            return f"${value:,.0f}"

        if scale == "int":
            return f"{value:,.0f}"

        raise ValueError(
            f"Unsupported format: {scale}"
        )

    # --------------------------------------------------
    # Calculate shared percentage scale
    # --------------------------------------------------
    percentage_values = []

    for metric, values in df.iterrows():

        metric_scale = row_formats.get(
            metric,
            infer_scale(metric)
        )

        if metric_scale not in ("%", "$", "int"):
            raise ValueError(
                f"Unsupported format '{metric_scale}' "
                f"for '{metric}'"
            )

        if metric_scale == "%":

            percentage_values.extend(
                abs(float(v))
                for v in values
                if pd.notna(v)
                and math.isfinite(float(v))
            )

    max_percentage = max(
        percentage_values,
        default=0
    )

    bar_scale = max(
        5,
        math.ceil(max_percentage / 5) * 5
    )

    # --------------------------------------------------
    # Diverging bar helper
    # --------------------------------------------------
    def draw_diverging_bar(
        x0,
        x1,
        y,
        value,
        height=10
    ):

        if pd.isna(value):
            return

        value = float(value)

        if not math.isfinite(value):
            return

        center_x = (x0 + x1) / 2

        available_half_width = (
            x1 - x0 - 32
        ) / 2

        track_left = (
            center_x - available_half_width
        )

        track_right = (
            center_x + available_half_width
        )

        rectangle(
            (
                track_left,
                y,
                track_right,
                y + height
            ),
            fill=PALE,
            radius=3
        )

        rule(
            center_x,
            y - 3,
            center_x,
            y + height + 3,
            MUTED,
            stroke=1
        )

        if value == 0:
            return

        proportion = min(
            abs(value) / bar_scale,
            1
        )

        bar_width = (
            proportion
            * available_half_width
        )

        if value > 0:

            bar_left = center_x
            bar_right = center_x + bar_width

            bar_color = TEAL

        else:

            bar_left = center_x - bar_width
            bar_right = center_x

            bar_color = NAVY

        rectangle(
            (
                bar_left,
                y,
                bar_right,
                y + height
            ),
            fill=bar_color,
            radius=2
        )

        rule(
            center_x,
            y - 3,
            center_x,
            y + height + 3,
            MUTED,
            stroke=1
        )

    # --------------------------------------------------
    # Slide title
    # --------------------------------------------------
    LEFT = 95
    RIGHT = 1505

    title_lines = wrap(
        title,
        f_title,
        RIGHT - LEFT
    )

    title_y = 44

    for k, line_text in enumerate(title_lines):

        write(
            line_text,
            LEFT,
            title_y + k * 53,
            f_title,
            NAVY
        )

    content_y = (
        title_y
        + len(title_lines) * 53
        + 7
    )

    if subtitle:

        subtitle_lines = wrap(
            subtitle,
            f_subtitle,
            RIGHT - LEFT
        )

        for k, line_text in enumerate(
            subtitle_lines
        ):

            write(
                line_text,
                LEFT,
                content_y + k * 27,
                f_subtitle,
                MUTED
            )

        content_y += (
            len(subtitle_lines) * 27
            + 15
        )

    else:
        content_y += 6

    rule(
        LEFT,
        content_y,
        RIGHT,
        content_y,
        LINE,
        stroke=2
    )

    # --------------------------------------------------
    # Matrix geometry
    # --------------------------------------------------
    header_top = content_y + 17
    header_height = 62

    row_top = (
        header_top
        + header_height
        + 6
    )

    row_bottom = 758
    row_gap = 11

    row_height = (
        row_bottom
        - row_top
        - row_gap * (n_rows - 1)
    ) / n_rows

    if row_height < 62:
        raise ValueError(
            "Insufficient space for the supplied KPI rows."
        )

    label_width = 405
    col_gap = 10

    value_width = (
        RIGHT
        - LEFT
        - label_width
        - n_cols * col_gap
    ) / n_cols

    # --------------------------------------------------
    # Column headings
    # --------------------------------------------------
    write(
        "KEY PERFORMANCE INDICATORS",
        LEFT + 17,
        header_top + header_height / 2,
        f_header,
        TEAL,
        anchor="lm"
    )

    for j, column in enumerate(df.columns):

        x0 = (
            LEFT
            + label_width
            + col_gap
            + j * (value_width + col_gap)
        )

        centered_wrapped(
            str(column),
            x0 + value_width / 2,
            header_top + header_height / 2,
            value_width - 12,
            initial_size=19,
            max_lines=3,
            color=TEAL
        )

    # --------------------------------------------------
    # KPI rows
    # --------------------------------------------------
    for i, (metric, values) in enumerate(
        df.iterrows()
    ):

        y0 = (
            row_top
            + i * (row_height + row_gap)
        )

        y1 = y0 + row_height

        cy = (y0 + y1) / 2

        scale = row_formats.get(
            metric,
            infer_scale(metric)
        )

        # ----------------------------------------------
        # Identify highest and lowest observed values
        # Applies to ALL KPI formats
        # ----------------------------------------------
        valid_values = [
            float(v)
            for v in values
            if pd.notna(v)
            and math.isfinite(float(v))
        ]

        highest_value = None
        lowest_value = None

        if len(valid_values) >= 2:

            row_max = max(valid_values)
            row_min = min(valid_values)

            # No highlighting if all values are equal
            if not math.isclose(
                row_max,
                row_min,
                rel_tol=1e-12,
                abs_tol=1e-12
            ):
                highest_value = row_max
                lowest_value = row_min

        # ----------------------------------------------
        # KPI label panel
        # ----------------------------------------------
        rectangle(
            (
                LEFT,
                y0,
                LEFT + label_width,
                y1
            ),
            fill=PALE,
            radius=15
        )

        rectangle(
            (
                LEFT,
                y0 + 15,
                LEFT + 7,
                y1 - 15
            ),
            fill=TEAL,
            radius=3
        )

        centered_wrapped(
            metric,
            LEFT + label_width / 2 + 5,
            cy,
            label_width - 55,
            initial_size=23,
            max_lines=3,
            color=NAVY
        )

        # ----------------------------------------------
        # Comparison value cards
        # ----------------------------------------------
        for j, value in enumerate(values):

            x0 = (
                LEFT
                + label_width
                + col_gap
                + j * (value_width + col_gap)
            )

            x1 = x0 + value_width

            # ------------------------------------------
            # Highest / lowest border highlighting
            # ------------------------------------------
            is_valid = (
                pd.notna(value)
                and math.isfinite(float(value))
            )

            is_highest = (
                is_valid
                and highest_value is not None
                and math.isclose(
                    float(value),
                    highest_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12
                )
            )

            is_lowest = (
                is_valid
                and lowest_value is not None
                and math.isclose(
                    float(value),
                    lowest_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12
                )
            )

            if is_highest:

                border_color = TEAL
                border_width = 4

            elif is_lowest:

                border_color = NAVY
                border_width = 4

            else:

                border_color = LINE
                border_width = 1

            rectangle(
                (x0, y0, x1, y1),
                fill=WHITE,
                outline=border_color,
                radius=15,
                stroke=border_width
            )

            # ------------------------------------------
            # Numeric value
            # ------------------------------------------
            value_text = format_value(
                value,
                scale
            )

            value_font = fit_font(
                value_text,
                value_width - 24,
                initial=31,
                minimum=15
            )

            if scale == "%":

                write(
                    value_text,
                    (x0 + x1) / 2,
                    cy - 13,
                    value_font,
                    NAVY,
                    anchor="mm"
                )

                draw_diverging_bar(
                    x0=x0,
                    x1=x1,
                    y=cy + 24,
                    value=value,
                    height=10
                )

            else:

                write(
                    value_text,
                    (x0 + x1) / 2,
                    cy,
                    value_font,
                    NAVY,
                    anchor="mm"
                )

    # --------------------------------------------------
    # Percentage change legend
    # --------------------------------------------------
    if percentage_values:

        legend_y = 784

        write(
            "PERCENTAGE CHANGE",
            LEFT,
            legend_y,
            f_footer,
            MUTED
        )

        rectangle(
            (
                310,
                legend_y + 3,
                333,
                legend_y + 11
            ),
            fill=NAVY,
            radius=2
        )

        write(
            "Decrease",
            344,
            legend_y,
            f_footer,
            MUTED
        )

        rectangle(
            (
                480,
                legend_y + 3,
                503,
                legend_y + 11
            ),
            fill=TEAL,
            radius=2
        )

        write(
            "Increase",
            514,
            legend_y,
            f_footer,
            MUTED
        )

        write(
            f"Shared Scale: ±{bar_scale:g}%",
            RIGHT,
            legend_y,
            f_footer,
            MUTED,
            anchor="rt"
        )
    # --------------------------------------------------
    # Border highlighting legend
    # --------------------------------------------------
    border_legend_y = 807
    
    write(
        "BORDER HIGHLIGHTS",
        LEFT,
        border_legend_y,
        f_footer,
        MUTED
    )
    
    write(
        "Blue = Highest Observed  |  Navy = Lowest Observed",
        310,
        border_legend_y,
        f_footer,
        MUTED
    )
    # --------------------------------------------------
    # Footer
    # --------------------------------------------------
    footer_y = 835

    rule(
        LEFT,
        footer_y,
        RIGHT,
        footer_y,
        LINE,
        stroke=2
    )

    if left_footer:

        write(
            left_footer,
            LEFT,
            footer_y + 16,
            f_footer,
            MUTED
        )

    if right_footer:

        write(
            right_footer,
            RIGHT,
            footer_y + 16,
            f_footer,
            MUTED,
            anchor="rt"
        )

    # --------------------------------------------------
    # Final anti-aliased image
    # --------------------------------------------------
    img = img.resize(
        (W, H),
        Image.Resampling.LANCZOS
    )

    if return_value == "image":
        return img

    if return_value == "file":

        if output is None:

            output = (
                Path(folder_location or ".")
                / "engagement_comparison_matrix.png"
            )

        output = Path(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        img.save(output)

        return str(output)

    display(img)

    return None


def statistical_assessment_matrix(
    questions,
    groups,
    df,
    target_final_assessment,
    folder_location=None,
    title='Evaluating Differences Across Our Guiding Questions',
    subtitle='Statistical comparisons with peer groups inform the overall assessment of the target population.',
    left_footer='',
    right_footer='',
    color_dict=color_dict,
    return_value=''
):
    """
    Render Slide 6: Statistical Assessment Matrix.

    Parameters
    ----------
    questions : list
        Guiding questions from:
        presentation_dict['engagement_measurement_grid']['questions']

    groups : list or dict
        Four peer groups from:
        presentation_dict['peer_group_introduction']['groups']

    df : pandas.DataFrame
        Required columns:
            question, peer_group, decision

        True  = statistically significant difference detected
        False = no statistically significant difference detected
        None  = not tested

    target_final_assessment : dict
        Overall decisions keyed by guiding question:
            {question: True / False / None}

    return_value : str
        'image' -> return PIL image
        'file'  -> save PNG and return path
        other   -> display image in notebook

    Notes
    -----
    Statistical testing is performed outside this function.
    This function only renders supplied decisions.
    """

    from pathlib import Path
    import pandas as pd
    import numpy as np
    from PIL import Image, ImageDraw
    from IPython.display import display

    # =========================================================
    # 1. Validate and prepare inputs
    # =========================================================

    group_items = (
        list(groups.values())
        if isinstance(groups, dict)
        else list(groups)
    )

    def group_name(group):
        if isinstance(group, dict):
            return str(
                group.get('title')
                or group.get('name')
                or group.get('label')
                or ''
            )
        return str(group)

    names = [group_name(g) for g in group_items]

    question_names = [
        q.get('question', '') if isinstance(q, dict) else str(q)
        for q in questions
    ]

    if len(names) != 4:
        raise ValueError('Exactly four peer groups are required.')

    if len(set(names)) != 4 or any(not n for n in names):
        raise ValueError(
            'Peer group names must be unique and non-empty.'
        )

    if not 1 <= len(question_names) <= 5:
        raise ValueError(
            'Provide between one and five guiding questions.'
        )

    if len(set(question_names)) != len(question_names):
        raise ValueError('Guiding questions must be unique.')

    if any(not q for q in question_names):
        raise ValueError('Guiding questions cannot be blank.')

    required_columns = {
        'question',
        'peer_group',
        'decision'
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f'DataFrame must contain: {required_columns}'
        )

    if df.duplicated(['question', 'peer_group']).any():
        raise ValueError(
            'Duplicate question/peer-group combinations found.'
        )

    expected = pd.MultiIndex.from_product(
        [question_names, names],
        names=['question', 'peer_group']
    )

    actual = pd.MultiIndex.from_frame(
        df[['question', 'peer_group']]
    )

    missing = expected.difference(actual)
    unexpected = actual.difference(expected)

    if len(missing) or len(unexpected):
        raise ValueError(
            'Assessment combinations do not match the presentation. '
            f'Missing: {list(missing)}; '
            f'Unexpected: {list(unexpected)}'
        )

    if not isinstance(target_final_assessment, dict):
        raise TypeError(
            'target_final_assessment must be a dictionary.'
        )

    if set(target_final_assessment) != set(question_names):
        raise ValueError(
            'target_final_assessment keys must exactly match '
            'the guiding questions.'
        )

    for question, result in target_final_assessment.items():
        if not isinstance(result, dict):
            raise TypeError(f'Result for {question!r} must be a dictionary.')
    
        if not {'decision', 'rationale'}.issubset(result):
            raise ValueError(
                f'Result for {question!r} must contain '
                "'decision' and 'rationale'."
            )
    
        decision = result['decision']
    
        if decision is not None and not isinstance(decision, (bool, np.bool_)):
            raise ValueError(
                f'Invalid decision for {question!r}: {decision!r}'
            )

    # =========================================================
    # 2. Initialize canvas and colours
    # =========================================================

    C = color_dict

    WHITE = C['white']
    NAVY = C['navy']
    TEAL = C['teal']
    MUTED = C['muted']
    PALE = C['pale']
    LINE = C['line']
    LIGHT = C['light_teal']

    W, H = 1600, 900

    img = Image.new(
        'RGB',
        (W, H),
        WHITE
    )

    draw = ImageDraw.Draw(img)

    # =========================================================
    # 3. Typography helpers
    # =========================================================

    def font(size, bold=False):
        return load_presentation_font(
            size,
            bold=bold
        )

    def text_width(value, f):
        bounds = draw.textbbox(
            (0, 0),
            str(value),
            font=f
        )

        return bounds[2] - bounds[0]

    def wrap(value, f, max_width):
        lines = []
        current = ''

        for word in str(value).split():
            candidate = (
                current + ' ' + word
            ).strip()

            if text_width(candidate, f) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)

                current = word

        if current:
            lines.append(current)

        return lines

    def fit_lines(
        value,
        max_width,
        max_lines,
        size=22,
        min_size=13,
        bold=False
    ):
        for n in range(size, min_size - 1, -1):
            f = font(n, bold)

            lines = wrap(
                value,
                f,
                max_width
            )

            if len(lines) <= max_lines:
                return lines, f, n

        f = font(min_size, bold)

        lines = wrap(
            value,
            f,
            max_width
        )[:max_lines]

        if lines:
            last = lines[-1]

            while (
                last
                and text_width(last + '…', f) > max_width
            ):
                last = last[:-1]

            lines[-1] = last + '…'

        return lines, f, min_size

    def text_box(
        value,
        bounds,
        size=21,
        bold=False,
        color=None,
        align='left',
        max_lines=4
    ):
        if color is None:
            color = NAVY

        x1, y1, x2, y2 = bounds
        margin = 14

        lines, f, actual = fit_lines(
            value,
            x2 - x1 - 2 * margin,
            max_lines,
            size=size,
            bold=bold
        )

        step = actual + 7

        start_y = (
            y1 + y2 - len(lines) * step
        ) / 2

        for i, line in enumerate(lines):

            if align == 'center':
                tx = (
                    x1 + x2 - text_width(line, f)
                ) / 2
            else:
                tx = x1 + margin

            draw.text(
                (tx, start_y + i * step),
                line,
                font=f,
                fill=color
            )

    # =========================================================
    # 4. Antialiased decision indicators
    # =========================================================

    def decision_indicator(
        value,
        bounds,
        radius=17,
        scale=4
    ):
        """
        Render at 4x resolution and downsample.

        True  : filled blue circle with white checkmark
        False : pale circle with navy dash
        None  : muted dot
        """

        x1, y1, x2, y2 = bounds

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        diameter = int(radius * 2 + 8)
        size = diameter * scale

        icon = Image.new(
            'RGBA',
            (size, size),
            (0, 0, 0, 0)
        )

        d = ImageDraw.Draw(icon)

        center = size / 2
        r = radius * scale

        circle = (
            center - r,
            center - r,
            center + r,
            center + r
        )

        if pd.isna(value):

            dot_r = 4 * scale

            d.ellipse(
                (
                    center - dot_r,
                    center - dot_r,
                    center + dot_r,
                    center + dot_r
                ),
                fill=LINE
            )

        elif isinstance(value, (bool, np.bool_)):

            if bool(value):

                # Filled blue circle
                d.ellipse(
                    circle,
                    fill=TEAL
                )

                # White checkmark
                points = [
                    (
                        center - 0.46 * r,
                        center - 0.02 * r
                    ),
                    (
                        center - 0.12 * r,
                        center + 0.32 * r
                    ),
                    (
                        center + 0.51 * r,
                        center - 0.39 * r
                    )
                ]

                stroke = max(
                    2,
                    int(0.19 * r)
                )

                d.line(
                    points,
                    fill=WHITE,
                    width=stroke,
                    joint='curve'
                )

                # Rounded checkmark endpoints
                endpoint_radius = stroke / 2

                for px, py in [
                    points[0],
                    points[-1]
                ]:
                    d.ellipse(
                        (
                            px - endpoint_radius,
                            py - endpoint_radius,
                            px + endpoint_radius,
                            py + endpoint_radius
                        ),
                        fill=WHITE
                    )

            else:

                # Pale outlined circle
                d.ellipse(
                    circle,
                    fill=PALE,
                    outline=LINE,
                    width=max(
                        2,
                        int(0.09 * r)
                    )
                )

                # Navy horizontal dash
                stroke = max(
                    2,
                    int(0.16 * r)
                )

                left = center - 0.40 * r
                right = center + 0.40 * r

                d.line(
                    (
                        left,
                        center,
                        right,
                        center
                    ),
                    fill=NAVY,
                    width=stroke
                )

                # Rounded dash endpoints
                endpoint_radius = stroke / 2

                for px in [left, right]:
                    d.ellipse(
                        (
                            px - endpoint_radius,
                            center - endpoint_radius,
                            px + endpoint_radius,
                            center + endpoint_radius
                        ),
                        fill=NAVY
                    )

        else:
            raise ValueError(
                f'Invalid decision {value!r}. '
                'Expected True, False or None.'
            )

        # Supersampling / antialiasing
        icon = icon.resize(
            (diameter, diameter),
            Image.Resampling.LANCZOS
        )

        paste_x = int(
            round(cx - diameter / 2)
        )

        paste_y = int(
            round(cy - diameter / 2)
        )

        img.paste(
            icon,
            (paste_x, paste_y),
            icon
        )

    # =========================================================
    # 5. Slide title and subtitle
    # =========================================================

    # Section header intentionally removed.

    title_lines, title_font, _ = fit_lines(
        title,
        1410,
        2,
        45,
        bold=True
    )

    for i, line in enumerate(title_lines):
        draw.text(
            (95, 54 + i * 53),
            line,
            font=title_font,
            fill=NAVY
        )

    sub_y = (
        54
        + len(title_lines) * 53
        + 12
    )

    subtitle_lines, sub_font, _ = fit_lines(
        subtitle,
        1410,
        2,
        20
    )

    for i, line in enumerate(subtitle_lines):
        draw.text(
            (95, sub_y + i * 27),
            line,
            font=sub_font,
            fill=MUTED
        )

    separator_y = (
        sub_y
        + len(subtitle_lines) * 27
        + 24
    )

    draw.line(
        (
            95,
            separator_y,
            1505,
            separator_y
        ),
        fill=LINE,
        width=3
    )

    # =========================================================
    # 6. Table layout
    # =========================================================

    # Column widths:
    # Evaluation Dimension : 430
    # Peer groups          : 105 each
    # Target Assessment    : 130
    # Rationale            : 430

    x = [
        95,
        525,
        630,
        735,
        840,
        945,
        1075,
        1505
    ]

    header_y = separator_y + 18
    table_y = header_y + 70
    bottom_y = 775

    gap = 10

    row_h = (
        bottom_y
        - table_y
        - gap * (len(question_names) - 1)
    ) / len(question_names)

    if row_h < 74:
        raise ValueError(
            'Insufficient vertical space for table rows.'
        )

    headers = [
        'EVALUATION DIMENSION',
        *names,
        'TARGET ASSESSMENT',
        'RATIONALE'
    ]

    assessment_lookup = df.set_index(['question', 'peer_group'])['decision'].to_dict()

    for j, heading in enumerate(headers):

        text_box(
            heading,
            (
                x[j] + 2,
                header_y,
                x[j + 1] - 2,
                header_y + 62
            ),
            size=18 if j in (0, 6) else 15,
            bold=True,
            color=NAVY,
            align=(
                'left'
                if j in (0, 6)
                else 'center'
            ),
            max_lines=4
        )

    # =========================================================
    # 7. Populate assessment rows
    # =========================================================

    for i, question in enumerate(question_names):

        y1 = table_y + i * (row_h + gap)
        y2 = y1 + row_h

        # Draw cell backgrounds
        for j in range(7):

            bounds = (
                x[j] + 3,
                y1,
                x[j + 1] - 3,
                y2
            )

            if j == 0:
                fill = PALE
            elif j == 5:
                fill = LIGHT
            else:
                fill = WHITE

            draw.rounded_rectangle(
                bounds,
                radius=12,
                fill=fill,
                outline=(
                    None
                    if j == 0
                    else LINE
                ),
                width=2
            )

        # Evaluation Dimension - 33% smaller font
        text_box(
            question,
            (
                x[0] + 3,
                y1,
                x[1] - 3,
                y2
            ),
            size=18,
            bold=True,
            max_lines=4
        )

        # Peer-group statistical decisions
        for j, name in enumerate(names):

            value = assessment_lookup[
                (question, name)
            ]

            decision_indicator(
                value,
                (
                    x[j + 1],
                    y1,
                    x[j + 2],
                    y2
                ),
                radius=17,
                scale=4
            )

        # Overall target assessment
        decision_indicator(
            target_final_assessment[question]['decision'],
            (
                x[5],
                y1,
                x[6],
                y2
            ),
            radius=21,
            scale=4
        )

        # Rationale placeholder
        text_box(
            target_final_assessment[question]['rationale'],
            (
                x[6] + 3,
                y1,
                x[7] - 3,
                y2
            ),
            size=19,
            color=MUTED,
            max_lines=4
        )

    # =========================================================
    # 8. Legend and footer
    # =========================================================

    legend_y = 803

    decision_indicator(
        True,
        (
            95,
            legend_y - 15,
            129,
            legend_y + 19
        ),
        radius=12,
        scale=4
    )

    draw.text(
        (139, legend_y - 10),
        'Statistically different',
        font=font(16),
        fill=MUTED
    )

    decision_indicator(
        False,
        (
            365,
            legend_y - 15,
            399,
            legend_y + 19
        ),
        radius=12,
        scale=4
    )

    draw.text(
        (409, legend_y - 10),
        'No statistically significant difference detected',
        font=font(16),
        fill=MUTED
    )

    draw.line(
        (
            95,
            835,
            1505,
            835
        ),
        fill=LINE,
        width=2
    )

    draw.text(
        (95, 851),
        left_footer,
        font=font(16),
        fill=MUTED
    )

    if right_footer:

        f = font(16)

        draw.text(
            (
                1505 - text_width(
                    right_footer, f
                ),
                851
            ),
            right_footer,
            font=f,
            fill=MUTED
        )

    # =========================================================
    # 9. Output
    # =========================================================

    if return_value == 'image':
        return img

    if return_value == 'file':

        folder = Path(
            folder_location or '.'
        ).expanduser()

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        output = (
            folder
            / 'statistical_assessment_matrix.png'
        )

        img.save(output)

        return str(output)

    display(img)

    return None


def key_findings(
    findings,
    folder_location=None,
    title="Key Findings",
    subtitle="What the analysis suggests about changes in member engagement.",
    left_footer="",
    right_footer="",
    font_path=None,
    color_dict=color_dict,
    return_value=""
):
    """
    Create a 1600 x 900 Key Findings slide.

    findings: list of 2 or 3 dictionaries:
        {
            "headline": "Main finding",
            "support": "Supporting explanation",
            "image": "/path/to/image.png"  # Optional
        }

    return_value:
        ''      Return Pillow image
        'image' Return Pillow image
        'file'  Save PNG and return file path

    Requires:
        load_presentation_font()
    """

    if not 2 <= len(findings) <= 3:
        raise ValueError("Provide 2 or 3 findings.")

    W, H = 1600, 900

    # --------------------------------------------------
    # Colours
    # --------------------------------------------------

    WHITE = color_dict["white"]
    NAVY = color_dict["navy"]
    TEAL = color_dict["teal"]
    MUTED = color_dict["muted"]
    PALE = color_dict["pale"]
    BORDER = color_dict["border"]

    # --------------------------------------------------
    # Fonts
    # --------------------------------------------------
    def font(size, bold=False):
        return load_presentation_font(
            size,
            bold=bold,
            font_path=font_path
        )

    f_title = font(48, True)
    f_subtitle = font(23)
    f_headline = font(30, True)
    f_support = font(22)
    f_footer = font(17)

    # --------------------------------------------------
    # Canvas
    # --------------------------------------------------
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    def text_width(text, f):
        box = draw.textbbox(
            (0, 0),
            str(text),
            font=f
        )
        return box[2] - box[0]

    def wrap_text(text, f, max_width):
        lines = []
        current = ""

        for word in str(text).split():
            trial = f"{current} {word}".strip()

            if text_width(trial, f) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines

    # --------------------------------------------------
    # Header
    # --------------------------------------------------
    LEFT = 70
    RIGHT = 1530

    draw.text(
        (LEFT, 43),
        title,
        font=f_title,
        fill=NAVY
    )

    subtitle_lines = wrap_text(
        subtitle,
        f_subtitle,
        RIGHT - LEFT
    )

    subtitle_y = 116

    for line in subtitle_lines:
        draw.text(
            (LEFT, subtitle_y),
            line,
            font=f_subtitle,
            fill=MUTED
        )
        subtitle_y += 31

    # No line beneath the subtitle.

    # --------------------------------------------------
    # Finding cards
    # --------------------------------------------------
    top = max(195, subtitle_y + 32)
    bottom = 820
    gap = 18

    n = len(findings)

    card_height = (
        bottom - top - (n - 1) * gap
    ) / n

    for i, finding in enumerate(findings):

        y1 = top + i * (card_height + gap)
        y2 = y1 + card_height

        x1, x2 = LEFT, RIGHT

        # ----------------------------------------------
        # Main card
        # ----------------------------------------------
        draw.rounded_rectangle(
            (
                round(x1),
                round(y1),
                round(x2),
                round(y2)
            ),
            radius=18,
            fill=PALE,
            outline=BORDER,
            width=2
        )

        # Left teal accent
        draw.rounded_rectangle(
            (
                round(x1),
                round(y1 + 12),
                round(x1 + 7),
                round(y2 - 12)
            ),
            radius=3,
            fill=TEAL
        )

        # ----------------------------------------------
        # Optional image
        # ----------------------------------------------
        image_path = finding.get("image")

        if image_path:

            path = Path(image_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            icon = Image.open(path).convert("RGBA")

            # Reserved image region
            image_region_left = x1 + 35
            image_region_right = x1 + 235

            image_region_width = (
                image_region_right
                - image_region_left
            )

            image_region_height = (
                card_height - 35
            )

            # Preserve aspect ratio
            scale = min(
                image_region_width / icon.width,
                image_region_height / icon.height
            ) * 0.85

            new_width = max(
                1,
                round(icon.width * scale)
            )

            new_height = max(
                1,
                round(icon.height * scale)
            )

            icon = icon.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )

            # Centre image within reserved region
            image_x = round(
                image_region_left
                + (image_region_width - new_width) / 2
            )

            image_y = round(
                y1
                + (card_height - new_height) / 2
            )

            # Preserve transparency
            img.paste(
                icon,
                (image_x, image_y),
                icon
            )

            # Vertical divider
            divider_x = x1 + 260

            draw.line(
                (
                    round(divider_x),
                    round(y1 + 28),
                    round(divider_x),
                    round(y2 - 28)
                ),
                fill=BORDER,
                width=2
            )

            text_x = x1 + 295

        else:

            # Use full width when no image exists
            text_x = x1 + 43

        available_width = (
            x2 - text_x - 45
        )

        # ----------------------------------------------
        # Wrap text
        # ----------------------------------------------
        headline_lines = wrap_text(
            finding["headline"],
            f_headline,
            available_width
        )

        support_lines = wrap_text(
            finding["support"],
            f_support,
            available_width
        )

        headline_line_height = 39
        support_line_height = 30
        text_gap = 13

        headline_height = (
            len(headline_lines)
            * headline_line_height
        )

        support_height = (
            len(support_lines)
            * support_line_height
        )

        content_height = (
            headline_height
            + text_gap
            + support_height
        )

        if content_height > card_height - 28:
            raise ValueError(
                f"Finding {i + 1} contains too much "
                "text for the available card height."
            )

        # ----------------------------------------------
        # Vertically centre text
        # ----------------------------------------------
        content_y = (
            y1 + (card_height - content_height) / 2
        )

        # Headline
        headline_y = content_y

        for line in headline_lines:
            draw.text(
                (text_x, headline_y),
                line,
                font=f_headline,
                fill=NAVY
            )

            headline_y += headline_line_height

        # Supporting text
        support_y = headline_y + text_gap

        for line in support_lines:
            draw.text(
                (text_x, support_y),
                line,
                font=f_support,
                fill=MUTED
            )

            support_y += support_line_height

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------
    footer_y = 855

    draw.line(
        (LEFT, footer_y, RIGHT, footer_y),
        fill=BORDER,
        width=1
    )

    if left_footer:

        draw.text(
            (LEFT, footer_y + 12),
            left_footer,
            font=f_footer,
            fill=MUTED
        )

    if right_footer:

        draw.text(
            (
                RIGHT - text_width(
                    right_footer,
                    f_footer
                ),
                footer_y + 12
            ),
            right_footer,
            font=f_footer,
            fill=MUTED
        )

    # --------------------------------------------------
    # Save / Return
    # --------------------------------------------------
    if return_value == "file":

        output = (
            Path(folder_location or ".")
            / filename
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        img.save(output)

        return str(output)

    display(img)
    return img


def opportunities_next_steps(
    opportunities,
    folder_location=None,
    title="Opportunities and Next Steps",
    subtitle="Translating our findings into practical opportunities for deeper insight and better decisions.",
    left_footer="Member Engagement Analysis",
    right_footer="08",
    color_dict=color_dict,
    return_value=''
):
    """
    Create a 1600 x 900 Pillow presentation slide.

    Parameters
    ----------
    opportunities : list
        List of 2 or 3 dictionaries:
            {
                "title": "Opportunity title",
                "description": "Short supporting paragraph.",
                "image": "/absolute/path/to/icon.png"
            }

    folder_location : str or Path, optional
        Destination folder when return_value='file'.

    color_dict : dict
        Shared presentation colour palette.

    return_value : str
        'image' -> return PIL Image
        'file'  -> save PNG and return file path
        other   -> display image in notebook
    """

    from pathlib import Path
    from PIL import Image, ImageDraw
    from IPython.display import display

    # =========================================================
    # 1. Validate inputs
    # =========================================================

    if not 2 <= len(opportunities) <= 3:
        raise ValueError(
            "Provide 2 or 3 opportunities."
        )

    # =========================================================
    # 2. Canvas and shared colour palette
    # =========================================================

    W, H = 1600, 900

    C = color_dict

    WHITE = C['white']
    NAVY = C['navy']
    TEAL = C['teal']
    MUTED = C['muted']
    BORDER = C['line']

    img = Image.new(
        'RGB',
        (W, H),
        WHITE
    )

    draw = ImageDraw.Draw(img)

    # =========================================================
    # 3. Typography
    # =========================================================

    def font(size, bold=False):
        return load_presentation_font(
            size,
            bold=bold
        )

    f_title = font(45, bold=True)
    f_subtitle = font(20)
    f_card_title = font(29, bold=True)
    f_description = font(21)
    f_footer = font(16)

    def text_width(text, f):
        bounds = draw.textbbox(
            (0, 0),
            str(text),
            font=f
        )
        return bounds[2] - bounds[0]

    def wrap_text(text, f, max_width):
        lines = []
        current = ''

        for word in str(text).split():
            trial = f'{current} {word}'.strip()

            if text_width(trial, f) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines

    def draw_wrapped(
        text,
        x,
        y,
        f,
        fill,
        width,
        line_height
    ):
        lines = wrap_text(
            str(text),
            f,
            width
        )

        for line in lines:
            draw.text(
                (x, y),
                line,
                font=f,
                fill=fill
            )
            y += line_height

        return y

    # =========================================================
    # 4. Slide title and subtitle
    # =========================================================

    # No section header

    draw.text(
        (95, 54),
        title,
        font=f_title,
        fill=NAVY
    )

    draw.text(
        (95, 119),
        subtitle,
        font=f_subtitle,
        fill=MUTED
    )

    draw.line(
        (95, 178, 1505, 178),
        fill=BORDER,
        width=3
    )

    # =========================================================
    # 5. Card layout
    # =========================================================

    card_top = 220
    card_bottom = 790
    card_gap = 26

    left = 95
    right = 1505

    n = len(opportunities)

    card_width = (
        (right - left) - (n - 1) * card_gap
    ) / n

    for i, opportunity in enumerate(opportunities):

        x1 = left + i * (card_width + card_gap)
        x2 = x1 + card_width

        # -----------------------------------------------------
        # Main card
        # -----------------------------------------------------

        draw.rounded_rectangle(
            (x1, card_top, x2, card_bottom),
            radius=18,
            fill=WHITE,
            outline=BORDER,
            width=2
        )

        # Thin blue accent at top of card
        draw.rounded_rectangle(
            (
                x1 + 30,
                card_top + 24,
                x1 + 95,
                card_top + 30
            ),
            radius=3,
            fill=TEAL
        )

        text_x = x1 + 30
        text_width_available = card_width - 60

        # -----------------------------------------------------
        # Opportunity title
        # -----------------------------------------------------

        title_y = card_top + 55

        title_end = draw_wrapped(
            opportunity['title'],
            text_x,
            title_y,
            f_card_title,
            NAVY,
            text_width_available,
            38
        )

        # -----------------------------------------------------
        # Description
        # -----------------------------------------------------

        description_y = title_end + 18

        description_end = draw_wrapped(
            opportunity['description'],
            text_x,
            description_y,
            f_description,
            MUTED,
            text_width_available,
            30
        )

        # -----------------------------------------------------
        # Divider between text and image
        # -----------------------------------------------------

        separator_y = description_end + 26

        draw.line(
            (
                text_x,
                separator_y,
                x2 - 30,
                separator_y
            ),
            fill=BORDER,
            width=2
        )

        # -----------------------------------------------------
        # Image placement
        # -----------------------------------------------------

        image_path = opportunity.get('image')

        if image_path:

            path = Path(image_path).expanduser()

            if not path.is_file():
                raise FileNotFoundError(
                    f'Image not found: {image_path}'
                )

            with Image.open(path) as source:
                icon = source.convert('RGBA')

            image_top = separator_y + 18
            image_bottom = card_bottom - 30

            max_image_width = card_width - 90
            max_image_height = image_bottom - image_top

            if max_image_height <= 0:
                raise ValueError(
                    f'No space available for image in card {i + 1}.'
                )

            # Preserve aspect ratio and allow enlargement
            scale = min(
                max_image_width / icon.width,
                max_image_height / icon.height
            ) * 0.80

            new_width = max(
                1,
                int(icon.width * scale)
            )

            new_height = max(
                1,
                int(icon.height * scale)
            )

            icon = icon.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )

            # Centre image horizontally
            image_x = int(
                (x1 + x2) / 2 - new_width / 2
            )

            # Centre image vertically in available space
            image_y = int(
                image_top
                + (max_image_height - new_height) / 2
            )

            img.paste(
                icon,
                (image_x, image_y),
                icon
            )

    # =========================================================
    # 6. Footer
    # =========================================================

    draw.line(
        (95, 835, 1505, 835),
        fill=BORDER,
        width=2
    )

    if left_footer:
        draw.text(
            (95, 851),
            left_footer,
            font=f_footer,
            fill=MUTED
        )

    if right_footer:
        draw.text(
            (
                1505 - text_width(
                    right_footer,
                    f_footer
                ),
                851
            ),
            right_footer,
            font=f_footer,
            fill=MUTED
        )

    # =========================================================
    # 7. Output
    # =========================================================

    if return_value == 'image':
        return img

    if return_value == 'file':

        folder = Path(
            folder_location or '.'
        ).expanduser()

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        output = (
            folder
            / 'opportunities_next_steps_slide.png'
        )

        img.save(output)

        return str(output)

    display(img)

    return None





#############################################


def generate_storytelling_presentation(
        presentation_dict,
        performance_df,
        folder_location,
        df_assessment,
        target_final_assessment
        ):

    frame_the_question(
        question=presentation_dict['frame_the_question']['Question'],
        context=presentation_dict['frame_the_question']['Context'],
        why=presentation_dict['frame_the_question']['Why'],
        scope=presentation_dict['frame_the_question']['Scope'],
        context_image=presentation_dict['frame_the_question']['context_image'],
        why_image=presentation_dict['frame_the_question']['why_image'],
        scope_image=presentation_dict['frame_the_question']['scope_image'],
        folder_location=folder_location
    )
    
    member_population_overview(
        demographics=presentation_dict['member_population_overview']['demographic_dict'],
        activity=presentation_dict['member_population_overview']['activity_dict'],
        observations=presentation_dict['member_population_overview']['observation_list'],
        images=presentation_dict['member_population_overview']['images'],
        folder_location=folder_location,
        left_footer='',
        right_footer=''
    )


    peer_group_introduction(
        groups=presentation_dict['peer_group_introduction']['groups'],
        target_population='Describe the Target Population',
        folder_location=folder_location,
        images=presentation_dict['peer_group_introduction']['images'],
        color_dict=color_dict,
        group_image_size=135,
        concept_image_size=120,
    )

    engagement_measurement_grid(
        questions=presentation_dict['engagement_measurement_grid']['questions'],
        folder_location=folder_location,
        left_footer=presentation_dict['engagement_measurement_grid']['footing_caveat'],
        right_footer=''
    )

    engagement_comparison_matrix(
        df=performance_df,
        title="Comparing Observed Performance Across Groups",
        subtitle="By examining relative and absolute changes across our identified KPIs, we can begin to distinguish measurable differences from meaningful changes and consider what these observations tell us about our original question.",
        left_footer="",
        right_footer=""
    )

    statistical_assessment_matrix(
        questions=presentation_dict['engagement_measurement_grid']['questions'],
        groups=presentation_dict['peer_group_introduction']['groups'],
        df=df_assessment,
        target_final_assessment=target_final_assessment,
        folder_location=folder_location
    )

    key_findings(
        findings=presentation_dict['key_findings'],
    )

    opportunities_next_steps(
        opportunities=presentation_dict['opportunities_next_steps'],
        folder_location=folder_location,
        left_footer="Member Engagement Analysis",
        right_footer="",
    )



def load_presentation_font(size, bold=False, font_path=None):
    '''
    Definition:
        Function Created by ChatGpt to Render Font, for Pillow Usage. Needed as original version of coding did not work in PC environment. 
    Parameters:
        TBD
    Returns:
        TBD
    Date Created:
        04-Nov-26
    Date Last Modified:
        04-Nov-26
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

    if font_path is not None:
        selected = Path(font_path).expanduser()
        if not selected.is_file():
            raise FileNotFoundError(f"Font not found: {selected}")
        return ImageFont.truetype(str(selected), size)

    windows_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    if bold:
        candidates = [
            windows_dir / "arialbd.ttf",
            windows_dir / "calibrib.ttf",
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        ]
    else:
        candidates = [
            windows_dir / "arial.ttf",
            windows_dir / "calibri.ttf",
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    raise RuntimeError(
        "No scalable presentation font found. "
        "Install Arial/DejaVu Sans or supply font_path."
    )



