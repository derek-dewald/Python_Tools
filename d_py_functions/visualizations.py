'''
module_word: visualizations
module_definition: Functions and tools used to visual data.

'''

from matplotlib.ticker import MaxNLocator, PercentFormatter,FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, HTML, Markdown
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
from numbers import Number
import seaborn as sns
import pandas as pd
import numpy as np
import html


from df_column_engineering import calculate_boundaries
from objects_manual import object_dict


color_dict = object_dict['color_dict']

fig_size = (10,5)
dpi = 300

def set_report_style():

    plt.style.use('default')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,

        'axes.titlesize': 14,
        'axes.titleweight': 'bold',

        'axes.labelsize': 11,

        'axes.edgecolor': color_dict['line'],

        'xtick.color': color_dict['muted'],
        'ytick.color': color_dict['muted'],

        'text.color': color_dict['navy'],

        'grid.color': color_dict['border'],
        'grid.alpha': 1.0,

        'figure.facecolor': color_dict['white'],
        'axes.facecolor': color_dict['white']
    })
    

def set_axis_number_format(
    ax,
    values,
    axis='y',
    format_type='auto',
    decimals=1
):
    """
    Definition:
        Standardizes numeric axis labels using automatic
        magnitude detection (K, M, B) or explicit formatting.

    Parameters:
        ax: Matplotlib axis to format.
        values: Numeric values used to determine magnitude.
        axis (str): 'x' or 'y'.
        format_type (str): 'auto', 'commas', or 'percent'.
        decimals (int): Maximum decimal places for compact values.

    Returns:
        formatter: Matplotlib FuncFormatter.

    Required Functions:
        numpy
        matplotlib
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    max_value = np.max(np.abs(values)) if len(values) else 0

    if format_type == 'auto':
        if max_value >= 1_000_000_000:
            divisor, suffix = 1_000_000_000, 'B'
        elif max_value >= 1_000_000:
            divisor, suffix = 1_000_000, 'M'
        elif max_value >= 1_000:
            divisor, suffix = 1_000, 'K'
        else:
            divisor, suffix = 1, ''

        def format_number(x, pos=None):
            scaled = x / divisor
            formatted = f'{scaled:,.{decimals}f}'
            if decimals > 0:
                formatted = formatted.rstrip('0').rstrip('.')
            return formatted + suffix

    elif format_type == 'commas':
        def format_number(x, pos=None):
            return f'{x:,.0f}'

    elif format_type == 'percent':
        def format_number(x, pos=None):
            return f'{x:.0%}'

    else:
        raise ValueError(
            "format_type must be 'auto', 'commas', or 'percent'"
        )

    formatter = FuncFormatter(format_number)

    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    elif axis == 'x':
        ax.xaxis.set_major_formatter(formatter)
    else:
        raise ValueError("axis must be 'x' or 'y'")

    return formatter

def presentation_histogram(
    df,
    column_name,
    bins=30,
    title=None,
    xlabel=None,
    ylabel='Frequency',
    show_mean=False,
    show_median=False,
    show_normal=False,
    show_kde=False,
    ax=None,
    show=True,
    dpi=250,
    file_name=None,
    fig_size=fig_size
):
    """
    Definition:
        Creates a standardized presentation histogram showing the distribution of a numeric DataFrame column, with optional statistical reference lines and distribution curves.

    Parameters:
        df (pd.DataFrame): Source dataset.
        column_name (str): Numeric column to visualize.
        bins (int): Number of histogram bins.
        title (str): Optional chart title.
        xlabel (str): Optional horizontal axis label.
        ylabel (str): Vertical axis label.
        show_mean (bool): Display mean reference line.
        show_median (bool): Display median reference line.
        show_normal (bool): Overlay theoretical normal distribution.
        show_kde (bool): Overlay actual distribution using KDE.
        ax (matplotlib.axes.Axes): Optional existing axis.
        show (bool): Display chart when True.
        dpi (int): Resolution of saved image.
        file_name (str): Optional output filename, including path.

    Returns:
        tuple: (fig, ax), allowing reuse in chart matrices.

    Date Created:
        14-Nov-26

    Date Last Modified:
        14-Nov-26

    Categorization:
        Data Visualization / Presentation Charts

    Usage:
        fig, ax = presentation_histogram(
            df=activity_df,
            column_name='RENT_M',
            bins=40,
            title='Distribution of Monthly Rent',
            show_mean=True,
            show_median=True,
            show_normal=True,
            show_kde=True
        )

    Notes:
        Uses standardized set_report_style() and color_dict.
        Normal distribution is fitted using the sample mean and standard deviation.
        KDE estimates the observed probability density.
        Distribution curves are scaled to histogram frequency.
        Optional ax allows integration into subplot matrices.
        Returned fig and ax permit additional customization.

    Required Functions:
        set_report_style()
        color_dict
        numpy
        pandas
        matplotlib
        scipy
    """

    set_report_style()

    values = pd.to_numeric(
        df[column_name],
        errors='coerce'
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        raise ValueError(
            f"No valid numeric values in {column_name}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------

    counts, bin_edges, patches = ax.hist(
        values,
        bins=bins,
        color=color_dict['teal'],
        edgecolor=color_dict['white'],
        linewidth=0.8
    )

    # --------------------------------------------------
    # Distribution Curves
    # --------------------------------------------------

    valid_distribution = (
        len(values) > 1
        and values.std() > 0
    )

    if (show_normal or show_kde) and valid_distribution:

        x = np.linspace(
            bin_edges[0],
            bin_edges[-1],
            300
        )

        bin_width = np.diff(bin_edges).mean()
        scale = len(values) * bin_width

        if show_normal:

            y_normal = norm.pdf(
                x,
                loc=values.mean(),
                scale=values.std()
            ) * scale

            ax.plot(
                x,
                y_normal,
                color=color_dict['navy'],
                linestyle='--',
                linewidth=2,
                label='Normal Distribution'
            )

        if show_kde:

            y_kde = gaussian_kde(values)(x) * scale

            ax.plot(
                x,
                y_kde,
                color=color_dict['red'],
                linewidth=2,
                label='Actual Distribution (KDE)'
            )

    # --------------------------------------------------
    # Statistical Reference Lines
    # --------------------------------------------------

    if show_mean:
        ax.axvline(
            values.mean(),
            color=color_dict['navy'],
            linestyle='--',
            linewidth=1.8,
            label=f"Mean: {values.mean():,.2f}"
        )

    if show_median:
        ax.axvline(
            values.median(),
            color=color_dict['pale'],
            linestyle=':',
            linewidth=2,
            label=f"Median: {values.median():,.2f}"
        )

    # --------------------------------------------------
    # Titles and Labels
    # --------------------------------------------------

    if title:
        ax.set_title(
            title,
            loc='left',
            pad=16
        )

    ax.set_xlabel(
        xlabel if xlabel is not None else column_name
    )

    ax.set_ylabel(ylabel)

    # --------------------------------------------------
    # Standardized Formatting
    # --------------------------------------------------

    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.6)
    ax.grid(axis='x', visible=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if (
        show_mean
        or show_median
        or (valid_distribution and (show_normal or show_kde))
    ):
        ax.legend(frameon=False)

    # --------------------------------------------------
    # Save and Display
    # --------------------------------------------------

    if file_name:
        fig.savefig(
            file_name,
            dpi=dpi,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig, ax

def presentation_distribution_analysis(
    df,
    column_name,
    use_log=False,
    breaks_df=None,
    title=None,
    figsize=(18, 5),
    show=True,
    dpi=250,
    file_name=None,
    print_=False,
    ylabel=None
):
    """
    Definition:
        Creates a standardized 1 x 3 presentation showing:
        1. Ranked distribution of observations.
        2. Cumulative concentration of values.
        3. Segment boundary comparison.

    Parameters:
        df (pd.DataFrame): Source dataset.
        column_name (str): Numeric column to analyze.
        use_log (bool): Apply log1p transformation to charts 1 and 3.
        breaks_df (pd.DataFrame): Optional segment boundaries.
            Defaults to calculate_boundaries().
        title (str): Optional overall figure title.
        figsize (tuple): Overall figure dimensions.
        show (bool): Display figure when True.
        dpi (int): Resolution of saved image.
        file_name (str): Optional output filename, including path.
        print_ (bool): Display brief chart descriptions.

    Returns:
        tuple: (fig, axes), where axes contains the three charts.

    Date Created:
        14-Nov-26

    Date Last Modified:
        14-Nov-26

    Categorization:
        Data Visualization / Presentation Charts

    Usage:
        fig, axes = presentation_distribution_analysis(
            df=activity_df,
            column_name='RENT_M',
            use_log=False,
            show=True
        )

    Notes:
        Maintains a fixed 1 x 3 chart arrangement.
        Chart 1 highlights observations above the 90th percentile.
        Chart 2 compares actual concentration to equal distribution.
        Chart 3 compares boundaries including and excluding zeros.
        Uses standardized set_report_style() and color_dict.
        Returned axes allow customization of individual charts.

    Required Functions:
        set_report_style()
        calculate_boundaries()
        color_dict
        numpy
        pandas
        matplotlib
    """

    set_report_style()

    # --------------------------------------------------
    # Data Preparation
    # --------------------------------------------------

    s = pd.to_numeric(
        df[column_name],
        errors='coerce'
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        raise ValueError(
            f"No valid numeric observations in {column_name}"
        )

    if use_log and (s < 0).any():
        raise ValueError(
            "Log transformation requires non-negative values."
        )

    if breaks_df is None:
        breaks_df = calculate_boundaries(df, column_name)

    # --------------------------------------------------
    # Figure Setup
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=True
    )

    primary = color_dict['navy']
    secondary = color_dict['teal']
    tertiary = color_dict['pale']
    muted = color_dict['muted']

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.6)
        ax.grid(axis='x', visible=False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --------------------------------------------------
    # Chart 1 - Ranked Distribution
    # --------------------------------------------------

    ax = axes[0]

    ranked = s.sort_values().reset_index(drop=True)

    y = np.log1p(ranked) if use_log else ranked
    x = np.arange(1, len(y) + 1)

    upper_threshold = ranked.quantile(0.90)
    upper_mask = ranked > upper_threshold

    ax.scatter(
        x,
        y,
        color=tertiary,
        s=8,
        alpha=0.65,
        rasterized=True
    )

    ax.scatter(
        x[upper_mask.to_numpy()],
        y[upper_mask.to_numpy()],
        color=primary,
        s=10,
        alpha=0.9,
        rasterized=True
    )

    zero_pct = (s == 0).mean()

    ax.text(
        0.03,
        0.96,
        f'{zero_pct:.1%} zero values',
        transform=ax.transAxes,
        va='top',
        fontsize=9,
        color=muted
    )

    ax.set_title(
        'Distribution',
        loc='left',
        pad=16
    )

    ax.set_xlabel('Ranked Observations')

    ax.set_ylabel(f'Log(1 + {ylabel or column_name})' if use_log else (ylabel or column_name))


    ax.xaxis.set_major_locator(MaxNLocator(5))

    # --------------------------------------------------
    # Chart 2 - Cumulative Concentration
    # --------------------------------------------------

    ax = axes[1]

    if (s < 0).any():

        ax.text(
            0.5, 0.5,
            'Concentration requires\nnon-negative values',
            ha='center',
            va='center',
            transform=ax.transAxes,
            color=muted
        )

        ax.set_xticks([])
        ax.set_yticks([])

    elif s.sum() == 0:

        ax.text(
            0.5, 0.5,
            'No positive value to concentrate',
            ha='center',
            va='center',
            transform=ax.transAxes,
            color=muted
        )

        ax.set_xticks([])
        ax.set_yticks([])

    else:

        values = np.sort(s.to_numpy())

        cumulative = np.insert(
            np.cumsum(values) / values.sum(),
            0, 0
        )

        population_pct = np.linspace(
            0, 1, len(values) + 1
        )

        ax.plot(
            population_pct,
            cumulative,
            color=primary,
            linewidth=2,
            label='Actual Concentration'
        )

        ax.plot(
            [0, 1],
            [0, 1],
            color=muted,
            linestyle='--',
            linewidth=1.2,
            alpha=0.6,
            label='Equal Distribution'
        )

        ax.fill_between(
            population_pct,
            cumulative,
            population_pct,
            color=tertiary,
            alpha=0.15
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))

        ax.legend(
            loc='upper left',
            frameon=False,
            fontsize=9
        )

    ax.set_title(
        'Concentration',
        loc='left',
        pad=16
    )

    ax.set_xlabel('Cumulative % of Members')
    ax.set_ylabel('Cumulative % of Value')

    # --------------------------------------------------
    # Chart 3 - Segment Boundary Comparison
    # --------------------------------------------------

    ax = axes[2]

    seg = breaks_df[f'{column_name}_SEG'].copy()
    seg1 = breaks_df[f'{column_name}_SEG1'].copy()

    if use_log:
        seg = np.log1p(seg)
        seg1 = np.log1p(seg1)

    x = np.arange(1, len(breaks_df) + 1)

    ax.plot(
        x,
        seg,
        color=secondary,
        linewidth=2,
        marker='o',
        markersize=5,
        label='Including Zeros'
    )

    ax.plot(
        x,
        seg1,
        color=primary,
        linewidth=2,
        marker='s',
        markersize=5,
        label='Excluding Zeros'
    )

    ax.set_title(
        'Segment Boundaries',
        loc='left',
        pad=16
    )

    ax.set_xlabel('Segment')

    ax.set_ylabel(f'Log(1 + {ylabel or column_name})' if use_log else (ylabel or column_name))

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    ax.legend(
        loc='upper left',
        frameon=False,
        fontsize=9
    )

    # --------------------------------------------------
    # Overall Figure Title
    # --------------------------------------------------

    fig.suptitle(
        title if title is not None
        else f'Concentration Analysis  |  {column_name}',
        fontsize=16,
        fontweight='bold',
        color=primary
    )

    # --------------------------------------------------
    # Optional Descriptions
    # --------------------------------------------------

    if print_:
        print(
            "Chart 1: Ranked distribution of values "
            "across the population."
        )
        print(
            "Chart 2: Cumulative concentration of "
            "value across the population."
        )
        print(
            "Chart 3: Segment boundaries including "
            "and excluding zeros."
        )

    # --------------------------------------------------
    # Save and Display
    # --------------------------------------------------

    if file_name:
        fig.savefig(
            file_name,
            dpi=dpi,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig, axes


def presentation_bar_chart(
    df,
    column_name,
    value_column=None,
    title=None,
    xlabel=None,
    ylabel=None,
    orientation='vertical',
    xrotation=0,
    sort=True,
    ascending=False,
    show_values=True,
    rotation=0,
    value_format='auto',
    agg='sum',
    ax=None,
    show=True,
    dpi=250,
    file_name=None,
    fig_size=fig_size
):
    """
    Definition:
        Creates a standardized presentation bar chart showing categorical frequencies or aggregated numerical values.

    Parameters:
        df (pd.DataFrame): Source dataset.
        column_name (str): Categorical column to visualize.
        value_column (str): Optional numeric column. If None,
            displays frequency counts.
        title (str): Optional chart title.
        xlabel (str): Optional horizontal axis label.
        ylabel (str): Optional vertical axis label.
        orientation (str): 'vertical' or 'horizontal'.
        sort (bool): Sort bars by value.
        ascending (bool): Sort direction.
        show_values (bool): Display values on bars.
        rotation (int): Rotation of category labels.
        value_format (str): 'auto', 'commas', or 'percent'.
        agg (str): Aggregation for value_column, e.g.
            'sum', 'mean', 'median', 'count'.
        figsize (tuple): Figure dimensions for standalone charts.
        ax (matplotlib.axes.Axes): Optional existing axis.
        show (bool): Display chart when True.
        dpi (int): Resolution of saved image.
        file_name (str): Optional output filename, including path.

    Returns:
        tuple: (fig, ax), allowing reuse in chart matrices.

    Date Created:
        14-Nov-26

    Date Last Modified:
        14-Nov-26

    Categorization:
        Data Visualization / Presentation Charts

    Usage:
        fig, ax = presentation_bar_chart(
            df=member_df,
            column_name='BRANCH',
            value_column='TOTAL_DEPOSIT',
            title='Total Deposits by Branch',
            xlabel='Branch',
            ylabel='Total Deposits ($)',
            rotation=25
        )

    Notes:
        Automatically formats large values using K, M, or B.
        Supports counts or aggregation of a numeric column.
        Uses standardized set_report_style() and color_dict.
        Uses set_axis_number_format() for numeric formatting.
        Supports standalone figures and subplot integration.
        Returns fig and ax for further customization.

    Required Functions:
        set_report_style()
        set_axis_number_format()
        color_dict
        numpy
        pandas
        matplotlib
    """

    set_report_style()

    # --------------------------------------------------
    # Data Preparation
    # --------------------------------------------------

    if orientation not in ('vertical', 'horizontal'):
        raise ValueError(
            "orientation must be 'vertical' or 'horizontal'"
        )

    if value_column is None:

        data = df[column_name].value_counts(dropna=False)
        default_ylabel = 'Frequency'

    else:

        numeric_values = pd.to_numeric(
            df[value_column],
            errors='coerce'
        )

        data = (
            df.assign(_chart_value=numeric_values)
            .groupby(column_name, dropna=False)['_chart_value']
            .agg(agg)
        )

        default_ylabel = value_column

    data = data.dropna()

    if data.empty:
        raise ValueError("No valid observations to plot.")

    if len(data) > 100:
        raise ValueError(
            f"Too many categories ({len(data):,}). "
            "Check whether column_name and value_column are reversed."
        )

    if sort:
        data = data.sort_values(ascending=ascending)

    categories = data.index.astype(str)
    values = data.to_numpy(dtype=float)

    # --------------------------------------------------
    # Figure Setup
    # --------------------------------------------------

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    else:
        fig = ax.figure

    # --------------------------------------------------
    # Bar Chart
    # --------------------------------------------------

    if orientation == 'vertical':

        bars = ax.bar(
            categories,
            values,
            color=color_dict['teal'],
            edgecolor=color_dict['white'],
            linewidth=0.8,
            width=0.7
        )

        ax.set_xlabel(
            xlabel if xlabel is not None else column_name
        )

        ax.set_ylabel(
            ylabel if ylabel is not None else default_ylabel
        )

        ax.grid(axis='y', alpha=0.6)
        ax.grid(axis='x', visible=False)

        plt.setp(
            ax.get_xticklabels(),
            rotation=rotation,
            ha='right' if rotation else 'center'
        )

    else:

        bars = ax.barh(
            categories,
            values,
            color=color_dict['teal'],
            edgecolor=color_dict['white'],
            linewidth=0.8,
            height=0.7
        )

        if sort:
            ax.invert_yaxis()

        ax.set_xlabel(
            xlabel if xlabel is not None else default_ylabel
        )

        ax.set_ylabel(
            ylabel if ylabel is not None else column_name
        )

        ax.grid(axis='x', alpha=0.6)
        ax.grid(axis='y', visible=False)

        plt.setp(
            ax.get_yticklabels(),
            rotation=rotation
        )

    # --------------------------------------------------
    # Automatic Number Formatting
    # --------------------------------------------------

    formatter = set_axis_number_format(
        ax=ax,
        values=values,
        axis='y' if orientation == 'vertical' else 'x',
        format_type=value_format
    )

    # --------------------------------------------------
    # Bar Value Labels
    # --------------------------------------------------

    if show_values:

        ax.bar_label(
            bars,
            labels=[formatter(v) for v in values],
            padding=3,
            fontsize=9,
            color=color_dict['navy']
        )

        # Leave space for labels
        ax.margins(
            y=0.12 if orientation == 'vertical' else 0.05,
            x=0.12 if orientation == 'horizontal' else 0.05
        )

    # --------------------------------------------------
    # Standardized Formatting
    # --------------------------------------------------

    if title:
        ax.set_title(
            title,
            loc='left',
            pad=16
        )

    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if orientation == 'vertical':
        ax.tick_params(axis='x', rotation=xrotation)

    # --------------------------------------------------
    # Save and Display
    # --------------------------------------------------

    if file_name:
        fig.savefig(
            file_name,
            dpi=dpi,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig, ax


def generate_data_header_box(
    value_dict,
    leading_text=None,
    trailing_text=None,
    columns=4,
    width=1200
):
    """
    Definition:
        Creates a standardized KPI dashboard using the global
        color_dict, with a configurable number of columns.

    Parameters:
        value_dict (dict): Dictionary containing KPI labels and values.
        leading_text (str): Optional Markdown before dashboard.
        trailing_text (str): Optional Markdown after dashboard.
        columns (int): Number of columns in the dashboard.
            Rows are calculated automatically.
        width (int): Maximum dashboard width in pixels.

    Returns:
        Rendered HTML dashboard.

    Date Created:
        21-Oct-26

    Date Last Modified:
        14-Nov-26

    Categorization:
        Data Visualization / Presentation Dashboards

    Usage:
        generate_data_header_box(
            value_dict=kpi_dict,
            columns=4
        )

    Notes:
        Uses color_dict for consistent presentation styling.
        Supports 1 x 4, 2 x 4, 2 x 3 and other arrangements.
        Numeric values automatically receive comma formatting.
        Preformatted strings are displayed unchanged.

    Required Functions:
        color_dict
        html
        IPython.display
    """

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if columns < 1:
        raise ValueError("columns must be at least 1.")

    if not value_dict:
        raise ValueError("value_dict cannot be empty.")

    # --------------------------------------------------
    # Standardized Colours
    # --------------------------------------------------

    primary = color_dict['teal']
    navy = color_dict['navy']
    muted = color_dict['muted']
    border = color_dict['line']
    background = color_dict['neutral']
    white = color_dict['white']
    black = color_dict['black']

    # --------------------------------------------------
    # KPI Cards
    # --------------------------------------------------

    cards = ""

    for item in value_dict.values():

        value = item['Value']
        label = html.escape(str(item['Text']))

        if isinstance(value, Number) and not isinstance(value, bool):
            formatted = f"{value:,.0f}"
        else:
            formatted = str(value)

        cards += f"""
        <div style="
            min-width: 0;
            padding: 20px 12px;
            text-align: center;
            background: {white};
            border: 1px solid {border};
            border-radius: 10px;
            box-sizing: border-box;
        ">
            <div style="
                font-size: 30px;
                font-weight: 700;
                color: {navy};
                overflow-wrap: anywhere;
            ">
                {html.escape(formatted)}
            </div>

            <div style="
                font-size: 14px;
                color: {navy};
                margin-top: 8px;
            ">
                {label}
            </div>
        </div>
        """

    # --------------------------------------------------
    # Dashboard Layout
    # --------------------------------------------------

    output = f"""
    <div style="
        width: {width}px;
        max-width: 100%;
        display: grid;
        grid-template-columns: repeat({columns}, minmax(0, 1fr));
        gap: 14px;
        padding: 18px;
        background: {navy};
        border: 1px solid {border};
        border-radius: 14px;
        box-sizing: border-box;
        font-family: Arial, Helvetica, sans-serif;
    ">
        {cards}
    </div>
    """

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    if leading_text:
        display(Markdown(leading_text))

    display(HTML(output))

    if trailing_text:
        display(Markdown(trailing_text))

def presentation_heatmap(
    df,
    title=None,
    xlabel=None,
    ylabel=None,
    show_values=False,
    value_format='.1f',
    figsize=fig_size,
    dpi=dpi,
    ax=None,
    show=True,
    file_name=None
):
    """
    Definition:
        Creates a standardized heatmap from an existing pivot table.

    Parameters:
        df (DataFrame): Pivot table to visualize.
        title (str): Optional chart title.
        xlabel (str): Optional X-axis label.
        ylabel (str): Optional Y-axis label.
        show_values (bool): Display cell values.
        value_format (str): Number formatting for annotations.
        figsize (tuple): Optional figure dimensions.
        dpi (int): Optional figure resolution.
        ax (Axes): Optional existing Matplotlib axis.
        show (bool): Whether to display the figure.
        file_name (str): Optional file path for saving.

    Returns:
        tuple: (fig, ax)

    Date Created:
        14-Nov-26

    Categorization:
        Data Visualization / Heatmap

    Required Functions:
        set_report_style()
        color_dict
        chart_format
    """

    set_report_style()

    cmap = LinearSegmentedColormap.from_list(
        'presentation_blues',
        [
            color_dict['light_teal'],
            color_dict['pale'],
            color_dict['teal'],
            color_dict['navy']
        ],
        N=256
    )


    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    heatmap = sns.heatmap(
        df,
        cmap=cmap,
        annot=show_values,
        fmt=value_format,
        linewidths=0.5,
        linecolor=color_dict['white'],
        cbar=True,
        ax=ax
    )
    
    # Apply consistent K / M / B formatting
    values = df.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    
    if finite_values.size:
        max_value = np.max(np.abs(finite_values))
    
        if max_value >= 1_000_000_000:
            divisor, suffix = 1_000_000_000, 'B'
        elif max_value >= 1_000_000:
            divisor, suffix = 1_000_000, 'M'
        elif max_value >= 1_000:
            divisor, suffix = 1_000, 'K'
        else:
            divisor, suffix = 1, ''
    
        def format_number(x, pos=None):
            scaled = x / divisor
            return f'{scaled:,.1f}'.rstrip('0').rstrip('.') + suffix
    
        colorbar = heatmap.collections[0].colorbar
        colorbar.formatter = FuncFormatter(format_number)
        colorbar.update_ticks()

    if title:
        ax.set_title(title, loc='left', pad=16)

    ax.set_xlabel(xlabel if xlabel is not None else df.columns.name)
    ax.set_ylabel(ylabel if ylabel is not None else df.index.name)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    ax.tick_params(axis='both', length=0)

    if file_name:
        fig.savefig(file_name, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax
