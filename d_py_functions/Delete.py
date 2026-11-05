# Shared, cross-platform scalable font loader (v1).
# Never silently use Pillow's small default font for presentation text.
def load_presentation_font(size, bold=False, font_path=None):
    from pathlib import Path
    import os
    from PIL import ImageFont

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


from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def frame_the_question(
    question, 
    context, 
    why,
    scope,
    folder_location=None,
    filename='slide_01.png', 
    title='Frame the Question',
    header='',
    left_footer='',
    right_footer=''
):
    
    """Create a 1600x900 presentation PNG in an existing folder; return its Path."""
    folder = Path(folder_location).expanduser() if folder_location is not None else Path.cwd()
    if not folder.is_dir():
        raise FileNotFoundError(f'Folder does not exist: {folder}')
    output = folder / filename
    if output.suffix.lower() != '.png':
        output = output.with_suffix('.png')

    W, H = 1600, 900
    img = Image.new('RGB', (W, H), '#FFFFFF')
    d = ImageDraw.Draw(img)
    navy, teal, muted = '#173247', '#087E92', '#637987'
    pale, line = '#EAF5F7', '#DDE8EC'

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

    def paragraph(text, x, y, width, size=26, fill=navy, bold=False, spacing=10, max_lines=None):
        font = load_font(size, bold)
        lines = wrap(text, font, width)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            while lines[-1] and d.textbbox((0, 0), lines[-1] + '…', font=font)[2] > width:
                lines[-1] = lines[-1][:-1]
            lines[-1] += '…'
        step = size + spacing
        for i, line_text in enumerate(lines):
            d.text((x, y + i * step), line_text, font=font, fill=fill)
        return y + len(lines) * step

    # Header
    d.text((95, 62), header, font=load_font(22, True), fill=teal)
    paragraph(title, 95, 111, 1300, 54, navy, True)
    d.line((95, 215, 1505, 215), fill=line, width=3)


    # Text, Horizontal, Vertical, Maximum Text Width, Text Size, Color, Bold T/F
    
    # Left-side analytical question
    d.rounded_rectangle((95, 264, 870, 746), radius=28, fill='#F1F8FA')
    d.rounded_rectangle((95, 264, 105, 746), radius=4, fill=teal)
    d.text((147, 320), 'Central Analytical Question', font=load_font(21, True), fill=teal)
    paragraph(question, 147, 390, 650, 32, navy, True, spacing=14, max_lines=6)

    # Draw vector-like icons directly with Pillow; shaded backgrounds included.
    def icon_base(cx, cy):
        d.ellipse((cx-45, cy-45, cx+45, cy+45), fill=pale)

    def target(cx, cy):
        icon_base(cx, cy)
        for r in (27, 17, 6):
            d.ellipse((cx-r, cy-r, cx+r, cy+r), outline=teal, width=4)
        d.line((cx+4, cy-4, cx+30, cy-30), fill=teal, width=4)
        d.polygon([(cx+30,cy-30),(cx+16,cy-27),(cx+27,cy-16)], fill=teal)

    def planner(cx, cy):
        icon_base(cx, cy)
        d.rounded_rectangle((cx-26,cy-23,cx+26,cy+28),radius=5,outline=teal,width=4)
        d.line((cx-26,cy-8,cx+26,cy-8),fill=teal,width=4)
        for x in (cx-13,cx+13):
            d.line((x,cy-31,x,cy-17),fill=teal,width=5)
        for x in (cx-12,cx+4):
            for y in (cy+4,cy+17):
                d.ellipse((x-3,y-3,x+3,y+3),fill=teal)

    def chart(cx, cy):
        icon_base(cx, cy)
        d.line((cx-27,cy+26,cx-27,cy-24),fill=teal,width=4)
        d.line((cx-27,cy+26,cx+29,cy+26),fill=teal,width=4)
        pts = [(cx-19,cy+13),(cx-4,cy+1),(cx+7,cy+7),(cx+25,cy-19)]
        d.line(pts,fill=teal,width=5,joint='curve')
        for x,y in pts:
            d.ellipse((x-4,y-4,x+4,y+4),fill=teal)

    sections = [
        ('CONTEXT', context, target, 323),
        ('WHY IT MATTERS', why, chart, 482),
        ('SCOPE', scope, planner, 641),
    ]
    for label, body, icon, cy in sections:
        icon(970, cy)
        d.text((1040, cy-40), label, font=load_font(20, True), fill=teal)
        paragraph(body, 1040, cy-4, 435, 23, navy, spacing=8, max_lines=3)

    d.line((95, 798, 1505, 798), fill=line, width=2)
    d.text((95, 823), left_footer, font=load_font(18), fill=muted)
    d.text((1390, 823),right_footer, font=load_font(18), fill=muted)

    img.save(output)
    return img
    


if __name__ == '__main__':
    result = create_question_slide(
        question='How does performance vary across our business member segments?',
        context='Business members differ in size, activity, and relationship characteristics.',
        why='Understand differences and identify opportunities for further investigation.',
        scope='Active business members over the prior 12 months.',
        folder_location='/mnt/data',
    )
    print(result)



from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont


def member_population_overview(
    demographics,
    activity,
    observations,
    folder_location=None,
    filename='slide_02.png',
    images=None,
    title='Understanding the Affected Population',
    header='STRUCTURE THE EVALUATION',
    left_footer='',
    right_footer=''
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

    output = (folder / filename).with_suffix('.png')

    W, H = 1600, 900

    navy = '#173247'
    teal = '#087E92'
    muted = '#637987'
    pale = '#F1F8FA'
    line = '#DDE8EC'
    white = '#FFFFFF'

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
            size=(90, 90)
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
        cy=688,
        size=(90, 90)
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

    img.save(output)

    return img

def comparison_framework(
    groups,
    target_population,
    folder_location=None,
    filename='slide_03.png',
    images=None,
    title='Establishing the Comparison Framework',
    header='STRUCTURE THE EVALUATION',
    subtitle=(
        'Four complementary reference points help distinguish '
        'potential fee-related changes from normal variation '
        'and broader member trends.'
    ),
    methodology_note=(
        'The four references provide complementary evidence; '
        'none independently establishes that the fee caused a change.'
    ),
    left_footer='',
    right_footer=''
):
    """
    Create a 1600 x 900 infographic-style comparison framework.

    Parameters
    ----------
    groups : list[dict]
        Four comparison groups. Each dictionary contains:
        - title
        - description
        - question
        - image_key (optional)

    target_population : str
        Description of the population being evaluated.

    images : dict, optional
        Dictionary of image paths, keyed by group image_key.
        Optional key 'target' supplies the target icon.

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

    if len(groups) != 4:
        raise ValueError('Exactly four comparison groups are required.')

    output = (folder / filename).with_suffix('.png')

    W, H = 1600, 900

    navy = '#173247'
    teal = '#087E92'
    muted = '#637987'
    pale = '#F1F8FA'
    line = '#DDE8EC'
    white = '#FFFFFF'
    light_teal = '#DDF0F3'

    img = Image.new('RGB', (W, H), white)
    draw = ImageDraw.Draw(img)

    def font(size, bold=False):
        return load_presentation_font(size, bold=bold)

    def text_width(text, fnt):
        bbox = draw.textbbox((0, 0), str(text), font=fnt)
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

    def paragraph(
        text, x, y, width,
        size=22, color=navy,
        bold=False, spacing=6,
        max_lines=None
    ):
        fnt = font(size, bold)
        lines = wrap(text, fnt, width)

        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]

            while last and text_width(last + '…', fnt) > width:
                last = last[:-1]

            lines[-1] = last + '…'

        for i, line_text in enumerate(lines):
            draw.text(
                (x, y + i * (size + spacing)),
                line_text,
                font=fnt,
                fill=color
            )

        return y + len(lines) * (size + spacing)

    def centered_text(text, cx, y, size, color, bold=False):
        fnt = font(size, bold)
        x = cx - text_width(text, fnt) / 2
        draw.text((x, y), text, font=fnt, fill=color)

    def place_image(image_path, cx, cy, size=(70, 70)):
        if image_path is None:
            return

        image_path = Path(image_path).expanduser()

        if not image_path.is_file():
            raise FileNotFoundError(
                f'Image not found: {image_path}'
            )

        icon = Image.open(image_path).convert('RGBA')
        icon.thumbnail(size, Image.Resampling.LANCZOS)

        x = int(cx - icon.width / 2)
        y = int(cy - icon.height / 2)

        img.paste(icon, (x, y), icon)

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    draw.text(
        (95, 42),
        header,
        font=font(22, True),
        fill=teal
    )

    draw.text(
        (95, 82),
        title,
        font=font(51, True),
        fill=navy
    )

    paragraph(
        subtitle,
        95, 156, 1390,
        size=21,
        color=muted,
        max_lines=2
    )

    draw.line(
        (95, 218, 1505, 218),
        fill=line,
        width=3
    )

    # --------------------------------------------------
    # Target population
    # --------------------------------------------------

    target_bounds = (95, 239, 1505, 336)

    draw.rounded_rectangle(
        target_bounds,
        radius=20,
        fill=pale
    )

    place_image(
        images.get('target'),
        cx=165,
        cy=286,
        size=(72, 72)
    )

    target_x = 225 if images.get('target') else 130

    draw.text(
        (target_x, 252),
        'Target Population',
        font=font(27, True),
        fill=navy
    )

    paragraph(
        target_population,
        target_x, 291, 1150,
        size=21,
        color=muted,
        max_lines=1
    )

    # --------------------------------------------------
    # Downward connector
    # --------------------------------------------------

    cx = 800

    draw.line(
        (cx, 336, cx, 365),
        fill=teal,
        width=4
    )

    draw.polygon(
        [(cx-9, 358), (cx+9, 358), (cx, 370)],
        fill=teal
    )

    # --------------------------------------------------
    # Four comparison cards
    # --------------------------------------------------

    card_bounds = [
        (95, 384, 785, 556),
        (815, 384, 1505, 556),
        (95, 572, 785, 744),
        (815, 572, 1505, 744)
    ]

    def group_card(bounds, group, index):
        x1, y1, x2, y2 = bounds

        is_expected = (
            group.get('image_key') == 'expected'
        )

        draw.rounded_rectangle(
            bounds,
            radius=20,
            fill=white,
            outline=line,
            width=2
        )

        image_path = images.get(group.get('image_key'))

        if image_path is not None:
            place_image(
                image_path,
                cx=x1+65,
                cy=y1+45,
                size=(66, 66)
            )

        heading_x = x1+118 if image_path else x1+32

        draw.text(
            (heading_x, y1+24),
            group['title'],
            font=font(27, True),
            fill=navy
        )

        paragraph(
            group['description'],
            x1+32, y1+79, x2-x1-64,
            size=19,
            color=muted,
            max_lines=2,
            spacing=4
        )

        draw.line(
            (x1+32, y2-53, x2-32, y2-53),
            fill=line,
            width=2
        )

        paragraph(
            group['question'],
            x1+32, y2-43, x2-x1-64,
            size=18,
            color=teal,
            bold=True,
            max_lines=1
        )

    for i, (bounds, group) in enumerate(
        zip(card_bounds, groups)
    ):
        group_card(bounds, group, i)

    # --------------------------------------------------
    # Methodology note
    # --------------------------------------------------

    paragraph(
        'INTERPRETATION',
        95, 763, 250,
        size=16,
        color=teal,
        bold=True
    )

    paragraph(
        methodology_note,
        95, 788, 1400,
        size=17,
        color=muted,
        max_lines=1
    )

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------

    draw.line(
        (95, 826, 1505, 826),
        fill=line,
        width=2
    )

    draw.text(
        (95, 844),
        left_footer,
        font=font(17),
        fill=muted
    )

    footer_font = font(17)
    footer_x = 1505 - text_width(right_footer, footer_font)

    draw.text(
        (footer_x, 844),
        right_footer,
        font=footer_font,
        fill=muted
    )

    img.save(output)

    return img



    
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def engagement_measurement_grid(
    questions,
    folder_location=None,
    filename='slide_04.png',
    title='Measuring Changes in Member Engagement',
    header='STRUCTURE THE EVALUATION',
    subtitle=(
        'Key questions guide how we assess changes '
        'in member engagement.'
    ),
    methodology_note=(
        'Metrics are evaluated over time and against '
        'the comparison framework established in Slide 3.'
    ),
    left_footer='',
    right_footer=''
):
    """
    Create a dynamic 1600 x 900 measurement framework slide.

    Supports:
        - 1 to 5 question rows
        - 1 to 5 metrics per question
        - Maximum 6 columns including the question column

    questions : list[dict]
        Each dictionary contains:
            question : str
            metrics  : list[str] or list[dict]

        Metric dictionaries may contain:
            label  : str
            symbol : str (optional)

    Returns
    -------
    PIL.Image.Image
    """

    if not 1 <= len(questions) <= 5:
        raise ValueError('Provide between 1 and 5 questions.')

    max_metrics = max(
        len(q.get('metrics', [])) for q in questions
    )

    if not 1 <= max_metrics <= 5:
        raise ValueError(
            'Provide between 1 and 5 metrics per question.'
        )

    if any(len(q.get('metrics', [])) == 0 for q in questions):
        raise ValueError('Each question must have at least one metric.')

    folder = (
        Path(folder_location).expanduser()
        if folder_location is not None
        else Path.cwd()
    )

    if not folder.is_dir():
        raise FileNotFoundError(
            f'Folder does not exist: {folder}'
        )

    output = (folder / filename).with_suffix('.png')

    W, H = 1600, 900

    navy = '#173247'
    teal = '#087E92'
    muted = '#637987'
    pale = '#F1F8FA'
    line = '#DDE8EC'
    white = '#FFFFFF'

    img = Image.new('RGB', (W, H), white)
    draw = ImageDraw.Draw(img)

    def font(size, bold=False):
        return load_presentation_font(size, bold=bold)

    def text_width(text, fnt):
        bbox = draw.textbbox((0, 0), str(text), font=fnt)
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

    def paragraph(
        text, x, y, width,
        size=22, color=navy,
        bold=False, spacing=5,
        max_lines=None
    ):
        fnt = font(size, bold)
        lines = wrap(text, fnt, width)

        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]

            while last and text_width(last + '…', fnt) > width:
                last = last[:-1]

            lines[-1] = last + '…'

        for i, line_text in enumerate(lines):
            draw.text(
                (x, y + i * (size + spacing)),
                line_text,
                font=fnt,
                fill=color
            )

        return len(lines) * (size + spacing)

    def centered_paragraph(
        text, cx, cy, width,
        size=20, color=navy,
        bold=False, max_lines=3
    ):
        fnt = font(size, bold)
        lines = wrap(text, fnt, width)

        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]

            while last and text_width(last + '…', fnt) > width:
                last = last[:-1]

            lines[-1] = last + '…'

        line_height = size + 5
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
    # Header
    # --------------------------------------------------

    draw.text(
        (95, 42),
        header,
        font=font(22, True),
        fill=teal
    )

    draw.text(
        (95, 82),
        title,
        font=font(51, True),
        fill=navy
    )

    paragraph(
        subtitle,
        95, 156, 1390,
        size=21,
        color=muted,
        max_lines=2
    )

    draw.line(
        (95, 218, 1505, 218),
        fill=line,
        width=3
    )

    # --------------------------------------------------
    # Dynamic grid dimensions
    # --------------------------------------------------

    grid_left = 95
    grid_right = 1505
    grid_top = 250
    grid_bottom = 742

    row_count = len(questions)
    column_count = max_metrics + 1

    row_gap = 12
    column_gap = 12

    grid_width = grid_right - grid_left
    grid_height = grid_bottom - grid_top

    row_height = (
        grid_height - row_gap * (row_count - 1)
    ) / row_count

    # Question column receives 1.65x the width
    # of an individual metric column.
    question_weight = 1.65

    metric_width = (
        grid_width - column_gap * (column_count - 1)
    ) / (max_metrics + question_weight)

    question_width = metric_width * question_weight

    # Font sizes scale slightly with density
    question_font_size = (
        24 if row_count <= 3 else 20
    )

    metric_font_size = (
        20 if max_metrics <= 3 else 17
    )

    # --------------------------------------------------
    # Render rows
    # --------------------------------------------------

    for row_index, question_data in enumerate(questions):

        y1 = grid_top + row_index * (row_height + row_gap)
        y2 = y1 + row_height

        # Question card
        qx1 = grid_left
        qx2 = qx1 + question_width

        draw.rounded_rectangle(
            (qx1, y1, qx2, y2),
            radius=18,
            fill=pale
        )

        draw.text(
            (qx1 + 22, y1 + 15),
            f'{row_index + 1:02d}',
            font=font(19, True),
            fill=teal
        )

        question_text = question_data['question']

        centered_paragraph(
            question_text,
            cx=(qx1 + qx2) / 2,
            cy=(y1 + y2) / 2 + 10,
            width=question_width - 42,
            size=question_font_size,
            bold=True,
            max_lines=4
        )

        # Metric cards
        for metric_index, metric in enumerate(
            question_data['metrics']
        ):

            if isinstance(metric, str):
                label = metric
                symbol = ''
            else:
                label = metric.get('label', '')
                symbol = metric.get('symbol', '')

            mx1 = (
                qx2 + column_gap
                + metric_index * (metric_width + column_gap)
            )
            mx2 = mx1 + metric_width

            draw.rounded_rectangle(
                (mx1, y1, mx2, y2),
                radius=18,
                fill=white,
                outline=line,
                width=2
            )

            metric_cx = (mx1 + mx2) / 2
            metric_cy = (y1 + y2) / 2

            if symbol:
                symbol_size = (
                    35 if row_count <= 3 else 26
                )

                symbol_font = font(symbol_size, True)

                draw.text(
                    (
                        metric_cx - text_width(
                            symbol, symbol_font
                        ) / 2,
                        metric_cy - 42
                    ),
                    symbol,
                    font=symbol_font,
                    fill=teal
                )

                label_cy = metric_cy + 25

            else:
                label_cy = metric_cy

            centered_paragraph(
                label,
                cx=metric_cx,
                cy=label_cy,
                width=metric_width - 24,
                size=metric_font_size,
                bold=True,
                max_lines=3
            )

    # --------------------------------------------------
    # Methodology note
    # --------------------------------------------------

    paragraph(
        methodology_note,
        95, 767, 1400,
        size=17,
        color=muted,
        max_lines=2
    )

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------

    draw.line(
        (95, 826, 1505, 826),
        fill=line,
        width=2
    )

    draw.text(
        (95, 844),
        left_footer,
        font=font(17),
        fill=muted
    )

    footer_font = font(17)

    draw.text(
        (
            1505 - text_width(
                right_footer, footer_font
            ),
            844
        ),
        right_footer,
        font=footer_font,
        fill=muted
    )

    img.save(output)

    return img


from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def engagement_comparison_matrix(
    data,
    folder_location=None,
    filename="slide_05_engagement_comparison.png",
    title="Comparing Changes in Member Engagement",
    header="ASSESS THE EVIDENCE",
    subtitle="Percentage changes across the target population and comparison groups.",
    left_footer="",
    right_footer="",
    font_path=None,
):
    """
    Creates a 1600 x 900 Pillow slide.

    data: Dictionary mapping each question to five percentage values,
          ordered as:
          Target Group, Peer Group, Random Sample,
          All Beem, Expected Target Group.

    Values are percentages, e.g. -4.2 means -4.2%.
    Use None for missing data.
    """

    W, H = 1600, 900

    # Presentation colours
    NAVY = "#173247"
    TEAL = "#087E92"
    MUTED = "#637987"
    PALE = "#F1F8FA"
    BORDER = "#DDE8EC"
    WHITE = "#FFFFFF"

    NEGATIVE = ["#FBEFEC", "#F5D9D2", "#EBA99C", "#D97969"]
    POSITIVE = ["#EAF5F5", "#CBE8E7", "#91CFCD", "#4BA7AA"]

    columns = [
        "Target Group",
        "Peer Group",
        "Random Sample",
        "All Beem",
        "Expected Target Group",
    ]

    questions = [
        "Are members maintaining their financial relationships?",
        "Are members continuing to transact with Beem?",
        "Are members reducing or ending their engagement?",
        "Are members happy?",
    ]

    if set(data) != set(questions):
        raise ValueError("Data must contain exactly the four specified questions.")

    for question in questions:
        if len(data[question]) != 5:
            raise ValueError(f"Expected five values for: {question}")

    # Fonts
    def font(size, bold=False):
        return load_presentation_font(size, bold=bold, font_path=font_path)

    f_header = font(20)
    f_title = font(48)
    f_subtitle = font(23)
    f_column = font(20)
    f_question = font(23)
    f_value = font(33)
    f_legend = font(17)
    f_footer = font(17)

    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    def text_width(value, f):
        return draw.textbbox((0, 0), value, font=f)[2]

    def centered_text(value, center_x, y, f, fill):
        draw.text(
            (center_x - text_width(value, f) / 2, y),
            value,
            font=f,
            fill=fill,
        )

    def wrap_text(value, f, max_width):
        words = value.split()
        lines = []
        current = ""

        for word in words:
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

    def cell_color(value):
        if value is None:
            return "#F0F2F3"

        magnitude = abs(float(value))

        if magnitude < 1:
            band = 0
        elif magnitude < 3:
            band = 1
        elif magnitude < 5:
            band = 2
        else:
            band = 3

        return NEGATIVE[band] if value < 0 else POSITIVE[band]

    def format_value(value):
        if value is None:
            return "N/A"
        return f"{value:+.1f}%" if value != 0 else "0.0%"

    # --------------------------------------------------
    # Slide header
    # --------------------------------------------------
    draw.text((70, 44), header.upper(), font=f_header, fill=TEAL)
    draw.text((70, 91), title, font=f_title, fill=NAVY)
    draw.text((70, 158), subtitle, font=f_subtitle, fill=MUTED)

    draw.line((70, 212, 1530, 212), fill=BORDER, width=2)

    # --------------------------------------------------
    # Matrix geometry
    # --------------------------------------------------
    left = 70
    right = 1530
    matrix_width = right - left

    question_width = 460
    gap = 8
    value_width = (matrix_width - question_width - 5 * gap) / 5

    column_y = 244
    column_height = 65

    row_start = 322
    row_height = 99
    row_gap = 12

    # Column headings
    for j, column in enumerate(columns):
        x = left + question_width + gap + j * (value_width + gap)

        lines = wrap_text(column, f_column, value_width - 12)
        line_height = 25
        total_height = len(lines) * line_height
        start_y = column_y + (column_height - total_height) / 2

        for k, line in enumerate(lines):
            centered_text(
                line,
                x + value_width / 2,
                start_y + k * line_height,
                f_column,
                NAVY,
            )

    # --------------------------------------------------
    # Question rows and coloured percentage blocks
    # --------------------------------------------------
    for i, question in enumerate(questions):
        y = row_start + i * (row_height + row_gap)

        # Question panel
        draw.rounded_rectangle(
            (left, y, left + question_width, y + row_height),
            radius=13,
            fill=PALE,
            outline=BORDER,
            width=1,
        )

        # Question number
        draw.text(
            (left + 18, y + 15),
            f"{i + 1:02d}",
            font=f_legend,
            fill=TEAL,
        )

        # Question text
        lines = wrap_text(question, f_question, question_width - 80)
        line_height = 29
        text_y = y + (row_height - len(lines) * line_height) / 2

        for k, line in enumerate(lines):
            draw.text(
                (left + 56, text_y + k * line_height),
                line,
                font=f_question,
                fill=NAVY,
            )

        # Five comparison cells
        for j, value in enumerate(data[question]):
            x = left + question_width + gap + j * (value_width + gap)

            draw.rounded_rectangle(
                (x, y, x + value_width, y + row_height),
                radius=13,
                fill=cell_color(value),
            )

            centered_text(
                format_value(value),
                x + value_width / 2,
                y + 29,
                f_value,
                NAVY,
            )

    # --------------------------------------------------
    # Colour legend
    # --------------------------------------------------
    legend_y = 790

    draw.text(
        (left, legend_y),
        "PERCENTAGE CHANGE",
        font=f_legend,
        fill=NAVY,
    )

    labels = ["0–1%", "1–3%", "3–5%", "5%+"]

    def draw_legend_group(x, heading, palette):
        draw.text((x, legend_y), heading, font=f_legend, fill=MUTED)

        for i, (label, colour) in enumerate(zip(labels, palette)):
            bx = x + i * 102
            by = legend_y + 29

            draw.rounded_rectangle(
                (bx, by, bx + 82, by + 18),
                radius=4,
                fill=colour,
            )
            centered_text(label, bx + 41, by + 24, f_legend, MUTED)

    draw_legend_group(350, "Negative change", NEGATIVE)
    draw_legend_group(820, "Positive change", POSITIVE)

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------
    draw.line((70, 862, 1530, 862), fill=BORDER, width=1)

    if left_footer:
        draw.text((70, 872), left_footer, font=f_footer, fill=MUTED)

    if right_footer:
        draw.text(
            (1530 - text_width(right_footer, f_footer), 872),
            right_footer,
            font=f_footer,
            fill=MUTED,
        )

    # Save and return
    if folder_location is not None:
        output = Path(folder_location) / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        img.save(output)

    return img


from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def key_findings_slide(
    findings,
    folder_location=None,
    filename="slide_07_key_findings.png",
    title="Key Findings",
    header="INTERPRET THE RESULTS",
    subtitle="What the analysis suggests about changes in member engagement.",
    left_footer="",
    right_footer="07",
    font_path=None,
):
    """
    findings: list of 2 or 3 dictionaries, each containing:
        {
            "headline": "Main finding",
            "support": "One sentence of supporting evidence."
        }

    Returns a 1600 x 900 Pillow image.
    """

    if not 2 <= len(findings) <= 3:
        raise ValueError("Provide 2 or 3 findings.")

    W, H = 1600, 900

    NAVY = "#173247"
    TEAL = "#087E92"
    MUTED = "#637987"
    PALE = "#F1F8FA"
    BORDER = "#DDE8EC"
    WHITE = "#FFFFFF"

    def font(size, bold=False):
        return load_presentation_font(size, bold=bold, font_path=font_path)

    f_header = font(20)
    f_title = font(48)
    f_subtitle = font(23)
    f_number = font(19)
    f_headline = font(31)
    f_support = font(23)
    f_footer = font(17)

    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    def text_width(text, f):
        return draw.textbbox((0, 0), text, font=f)[2]

    def wrap_text(text, f, max_width):
        lines = []
        current = ""

        for word in text.split():
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

    # Header
    draw.text((70, 44), header.upper(), font=f_header, fill=TEAL)
    draw.text((70, 91), title, font=f_title, fill=NAVY)
    draw.text((70, 158), subtitle, font=f_subtitle, fill=MUTED)
    draw.line((70, 212, 1530, 212), fill=BORDER, width=2)

    # Finding cards
    top = 245
    bottom = 830
    gap = 18

    n = len(findings)
    card_height = (bottom - top - (n - 1) * gap) / n

    for i, finding in enumerate(findings):
        y1 = top + i * (card_height + gap)
        y2 = y1 + card_height

        x1, x2 = 70, 1530

        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=18,
            fill=PALE,
            outline=BORDER,
            width=2,
        )

        # Teal accent
        draw.rounded_rectangle(
            (x1, y1, x1 + 7, y2),
            radius=3,
            fill=TEAL,
        )

        text_x = x1 + 45

        draw.text(
            (text_x, y1 + 22),
            f"FINDING {i + 1:02d}",
            font=f_number,
            fill=TEAL,
        )

        headline_lines = wrap_text(
            finding["headline"],
            f_headline,
            x2 - text_x - 45,
        )

        headline_y = y1 + 58

        for line in headline_lines:
            draw.text(
                (text_x, headline_y),
                line,
                font=f_headline,
                fill=NAVY,
            )
            headline_y += 39

        support_lines = wrap_text(
            finding["support"],
            f_support,
            x2 - text_x - 45,
        )

        support_y = headline_y + 12

        for line in support_lines:
            draw.text(
                (text_x, support_y),
                line,
                font=f_support,
                fill=MUTED,
            )
            support_y += 31

        if support_y > y2 - 12:
            raise ValueError(
                f"Finding {i + 1} is too long for the available card height."
            )

    # Footer
    draw.line((70, 862, 1530, 862), fill=BORDER, width=1)

    if left_footer:
        draw.text((70, 872), left_footer, font=f_footer, fill=MUTED)

    if right_footer:
        draw.text(
            (1530 - text_width(right_footer, f_footer), 872),
            right_footer,
            font=f_footer,
            fill=MUTED,
        )

    if folder_location is not None:
        output = Path(folder_location) / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        img.save(output)

    return img


def assumptions_limitations_slide(
    assumptions,
    limitations,
    folder_location=None,
    filename="appendix_assumptions_limitations.png",
    title="Assumptions and Limitations",
    header="APPENDIX",
    subtitle="Key considerations when interpreting the analytical results.",
    interpretation_note=None,
    left_footer="",
    right_footer="APPENDIX",
    font_path=None,
):
    """
    Creates a reusable 1600 x 900 appendix slide.

    assumptions / limitations:
        Lists of dictionaries:
        {
            "title": "Short heading",
            "description": "Explanation in one or two sentences."
        }

    Supports 2–5 items per column.
    """

    if not 2 <= len(assumptions) <= 5:
        raise ValueError("Provide 2–5 assumptions.")

    if not 2 <= len(limitations) <= 5:
        raise ValueError("Provide 2–5 limitations.")

    W, H = 1600, 900

    NAVY = "#173247"
    TEAL = "#087E92"
    MUTED = "#637987"
    PALE = "#F1F8FA"
    BORDER = "#DDE8EC"
    WHITE = "#FFFFFF"

    def font(size, bold=False):
        return load_presentation_font(size, bold=bold, font_path=font_path)

    f_header = font(20)
    f_title = font(48)
    f_subtitle = font(23)
    f_card_title = font(29)
    f_item_title = font(22)
    f_description = font(19)
    f_note = font(20)
    f_footer = font(17)

    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    def text_width(text, f):
        return draw.textbbox((0, 0), text, font=f)[2]

    def wrap_text(text, f, max_width):
        lines = []
        current = ""

        for word in text.split():
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

    def draw_wrapped(text, x, y, f, fill, width, spacing):
        lines = wrap_text(text, f, width)

        for line in lines:
            draw.text((x, y), line, font=f, fill=fill)
            y += spacing

        return y

    # --------------------------------------------
    # Header
    # --------------------------------------------
    draw.text((70, 44), header.upper(), font=f_header, fill=TEAL)
    draw.text((70, 91), title, font=f_title, fill=NAVY)
    draw.text((70, 158), subtitle, font=f_subtitle, fill=MUTED)

    draw.line((70, 212, 1530, 212), fill=BORDER, width=2)

    # --------------------------------------------
    # Two-column layout
    # --------------------------------------------
    card_top = 245
    card_bottom = 730 if interpretation_note else 820

    left_x1, left_x2 = 70, 790
    right_x1, right_x2 = 810, 1530

    def draw_column(x1, x2, heading, items):
        draw.rounded_rectangle(
            (x1, card_top, x2, card_bottom),
            radius=18,
            fill=PALE,
            outline=BORDER,
            width=2,
        )

        # Accent
        draw.rounded_rectangle(
            (x1, card_top, x1 + 7, card_bottom),
            radius=3,
            fill=TEAL,
        )

        draw.text(
            (x1 + 32, card_top + 24),
            heading,
            font=f_card_title,
            fill=NAVY,
        )

        draw.line(
            (x1 + 32, card_top + 76, x2 - 32, card_top + 76),
            fill=BORDER,
            width=2,
        )

        content_top = card_top + 100
        content_bottom = card_bottom - 20
        slot_height = (content_bottom - content_top) / len(items)

        for i, item in enumerate(items):
            y = content_top + i * slot_height

            draw.text(
                (x1 + 32, y),
                f"{i + 1:02d}",
                font=f_description,
                fill=TEAL,
            )

            text_x = x1 + 77
            available_width = x2 - text_x - 28

            title_end = draw_wrapped(
                item["title"],
                text_x,
                y,
                f_item_title,
                NAVY,
                available_width,
                28,
            )

            description_end = draw_wrapped(
                item["description"],
                text_x,
                title_end + 7,
                f_description,
                MUTED,
                available_width,
                25,
            )

            if description_end > y + slot_height - 4:
                raise ValueError(
                    f"Content in '{heading}', item {i + 1}, "
                    "is too long for the available space."
                )

    draw_column(left_x1, left_x2, "Key Assumptions", assumptions)
    draw_column(right_x1, right_x2, "Analytical Limitations", limitations)

    # --------------------------------------------
    # Optional interpretation note
    # --------------------------------------------
    if interpretation_note:
        note_y1, note_y2 = 750, 835

        draw.rounded_rectangle(
            (70, note_y1, 1530, note_y2),
            radius=12,
            fill=WHITE,
            outline=BORDER,
            width=2,
        )

        draw.rectangle(
            (70, note_y1 + 12, 76, note_y2 - 12),
            fill=TEAL,
        )

        draw.text(
            (95, note_y1 + 12),
            "INTERPRETATION NOTE",
            font=f_footer,
            fill=TEAL,
        )

        note_end = draw_wrapped(
            interpretation_note,
            95,
            note_y1 + 39,
            f_note,
            NAVY,
            1400,
            26,
        )

        if note_end > note_y2 - 5:
            raise ValueError("Interpretation note is too long.")

    # --------------------------------------------
    # Footer
    # --------------------------------------------
    draw.line((70, 862, 1530, 862), fill=BORDER, width=1)

    if left_footer:
        draw.text((70, 872), left_footer, font=f_footer, fill=MUTED)

    if right_footer:
        draw.text(
            (1530 - text_width(right_footer, f_footer), 872),
            right_footer,
            font=f_footer,
            fill=MUTED,
        )

    if folder_location is not None:
        output = Path(folder_location) / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        img.save(output)

    return img



from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def opportunities_next_steps_slide(
    opportunities,
    folder_location=None,
    filename="slide_08_opportunities.png",
    title="Opportunities and Next Steps",
    header="LOOKING FORWARD",
    subtitle="Potential opportunities to strengthen the analysis and inform future decisions.",
    left_footer="Member Engagement Analysis",
    right_footer="08",
    font_path=None,
):
    """
    Create a 1600 x 900 Pillow presentation slide.

    opportunities: List of 2 or 3 dictionaries.

    Each dictionary:
        {
            "title": "Opportunity title",
            "description": "Short supporting paragraph.",
            "image": "/absolute/path/to/icon.png"
        }

    Images should ideally be transparent PNGs.

    Returns a PIL Image.
    """

    if not 2 <= len(opportunities) <= 3:
        raise ValueError("Provide 2 or 3 opportunities.")

    W, H = 1600, 900

    NAVY = "#173247"
    TEAL = "#087E92"
    MUTED = "#637987"
    PALE = "#F1F8FA"
    BORDER = "#DDE8EC"
    WHITE = "#FFFFFF"

    # --------------------------------------------------
    # Fonts
    # --------------------------------------------------
    def font(size, bold=False):
        return load_presentation_font(size, bold=bold, font_path=font_path)

    f_header = font(20)
    f_title = font(48)
    f_subtitle = font(23)
    f_card_title = font(30)
    f_description = font(23)
    f_footer = font(17)

    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    def text_width(text, f):
        return draw.textbbox((0, 0), text, font=f)[2]

    def wrap_text(text, f, max_width):
        lines = []
        current = ""

        for word in text.split():
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

    def draw_wrapped(text, x, y, f, fill, width, line_height):
        lines = wrap_text(text, f, width)

        for line in lines:
            draw.text((x, y), line, font=f, fill=fill)
            y += line_height

        return y

    # --------------------------------------------------
    # Slide header
    # --------------------------------------------------
    draw.text((70, 44), header.upper(), font=f_header, fill=TEAL)
    draw.text((70, 91), title, font=f_title, fill=NAVY)
    draw.text((70, 158), subtitle, font=f_subtitle, fill=MUTED)

    draw.line((70, 212, 1530, 212), fill=BORDER, width=2)

    # --------------------------------------------------
    # Card layout
    # --------------------------------------------------
    card_top = 255
    card_bottom = 820
    card_gap = 28

    left = 70
    right = 1530

    n = len(opportunities)

    card_width = (
        (right - left) - (n - 1) * card_gap
    ) / n

    card_height = card_bottom - card_top

    for i, opportunity in enumerate(opportunities):

        x1 = left + i * (card_width + card_gap)
        x2 = x1 + card_width

        # Main card
        draw.rounded_rectangle(
            (x1, card_top, x2, card_bottom),
            radius=20,
            fill=PALE,
            outline=BORDER,
            width=2,
        )

        # Teal left accent
        draw.rounded_rectangle(
            (x1, card_top, x1 + 10, card_bottom),
            radius=4,
            fill=TEAL,
        )

        text_x = x1 + 35
        text_width_available = card_width - 70

        # Card title
        title_y = card_top + 36

        title_end = draw_wrapped(
            opportunity["title"],
            text_x,
            title_y,
            f_card_title,
            NAVY,
            text_width_available,
            38,
        )

        # Card description
        description_y = title_end + 18

        description_end = draw_wrapped(
            opportunity["description"],
            text_x,
            description_y,
            f_description,
            NAVY,
            text_width_available,
            32,
        )

        # --------------------------------------------------
        # Image placement
        # --------------------------------------------------

        image_path = opportunity.get("image")
        
        if image_path:
            path = Path(image_path)
        
            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )
        
            icon = Image.open(path).convert("RGBA")
        
            # Available horizontal space within the card
            max_image_width = card_width - 60
        
            # Available vertical space below description
            image_top = description_end + 20
            image_bottom = card_bottom - 25
        
            max_image_height = image_bottom - image_top
        
            if max_image_height <= 0:
                raise ValueError(
                    f"No space available for image in card {i + 1}."
                )
        
            # Calculate scale, allowing enlargement
            scale = min(
                max_image_width / icon.width,
                max_image_height / icon.height
            )* 0.75
        
            new_width = int(icon.width * scale)
            new_height = int(icon.height * scale)
        
            # Resize image
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
        
            # Paste image with transparency
            img.paste(icon, (image_x, image_y), icon)




    # --------------------------------------------------
    # Footer
    # --------------------------------------------------
    draw.line((70, 862, 1530, 862), fill=BORDER, width=1)

    if left_footer:
        draw.text(
            (70, 872),
            left_footer,
            font=f_footer,
            fill=MUTED,
        )

    if right_footer:
        draw.text(
            (
                1530 - text_width(right_footer, f_footer),
                872,
            ),
            right_footer,
            font=f_footer,
            fill=MUTED,
        )

    # --------------------------------------------------
    # Save and return
    # --------------------------------------------------
    if folder_location is not None:
        output = Path(folder_location) / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        img.save(output)

    return img


