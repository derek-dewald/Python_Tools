'''
module_word: visualizations
module_definition: Functions and tools used to visual data.

'''
from matplotlib.ticker import PercentFormatter, MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from df_column_engineering import calculate_boundaries
from objects_manual import object_dict

chart_format = object_dict['chart_format']

def apply_chart_format(ax, chart_format=chart_format):
    """
    
    Definition:
        Intermediary function created solely for the purposes of applying formating to charts. Created by GPT
        
    Parameters:
        ax (ax): Chart Axis
        chart_format(dict): Predefined Formating structure. 
    
    Returns:
        ax
    Date Created:
        11-Nov-26
    Date Last Modified:
        11-Nov-26
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
    """

    ax.set_facecolor(chart_format['background'])

    ax.spines['top'].set_visible(
        chart_format['show_top_spine']
    )
    ax.spines['right'].set_visible(
        chart_format['show_right_spine']
    )
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(chart_format['grid'])

    ax.tick_params(
        axis='both',
        labelsize=chart_format['tick_size'],
        length=0,
        pad=6
    )

    ax.grid(
        axis=chart_format['grid_axis'],
        color=chart_format['grid'],
        alpha=chart_format['grid_alpha']
    )

    ax.set_axisbelow(True)

    ax.title.set_fontsize(chart_format['subtitle_size'])
    ax.title.set_fontweight('bold')
    ax.title.set_color(chart_format['primary'])

    ax.xaxis.label.set_fontsize(chart_format['label_size'])
    ax.yaxis.label.set_fontsize(chart_format['label_size'])

    return ax


def plot_column_distribution_analysis(
    df,
    column_name,
    use_log=False,
    breaks_df=None,
    figsize=None,
    chart_format=chart_format,
    print_=False
):
    """
    Definition:
        Visualization Function, which plots 3 Graphs to help understand the distribution and values of a column.
        
        Chart 1: Visualizes a Distirbution of values Across the population.
        Chart 2: Plots a concentration analysis, showing the distribution of cummulative value over the population.
        Chart 3: Plots the Segment Points (Deciles).

    Parameters:
        df(df): Dataframe
        column_name(str): Name of Column to be analyzed.
        use_log(binary): Indicator to determine whether absolute value or Log should be applied.
        breaks_df(list): Optional List to determne segmentation for chart 3. If False, returns Decile.
        fig_size(tuple): Allows for determination of figsize, used default chart_format option when False
        chart_format(dict): Default CHart format for consistency.
        print(binary): Option to return brief text description of each chart.
        
    Returns:
        fig
    Date Created:
        11-Nov-26
    Date Last Modified:
        11-Nov-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        from synthetic_member import member_df,activity_df,mbr_profile
        a = plot_column_distribution_analysis(activity_df,'RENT_M')
    Notes:
        None
    Required Functions:
        None
        
    
    
    """

    if breaks_df is None:
        breaks_df = calculate_boundaries(df, column_name)

    if figsize is None:
        figsize = chart_format['figsize']

    s = pd.to_numeric(df[column_name], errors='coerce').dropna()

    if s.empty:
        raise ValueError("No valid numeric observations.")

    if use_log and (s < 0).any():
        raise ValueError("Log transformation requires non-negative values.")

    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        dpi=chart_format['dpi'],
        constrained_layout=True
    )

    fig.patch.set_facecolor(chart_format['background'])

    for ax in axes:
        apply_chart_format(ax, chart_format)

    primary = chart_format['primary']
    secondary = chart_format['secondary']
    tertiary = chart_format['tertiary']
    muted = chart_format['muted']

    ####################################################################
    # Plot 1 - Ranked Distribution
    ####################################################################

    ax = axes[0]

    ranked = s.sort_values().reset_index(drop=True)

    if use_log:
        y = np.log1p(ranked)
    else:
        y = ranked

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
        0.03, 0.96,
        f'{zero_pct:.1%} zero values',
        transform=ax.transAxes,
        va='top',
        fontsize=chart_format['tick_size'],
        color=muted
    )

    ax.set_title(
        '01  |  Distribution',
        loc='left',
        pad=chart_format['title_pad']
    )

    ax.set_xlabel('Ranked Observations')
    ax.set_ylabel(
        f'Log(1 + {column_name})'
        if use_log else column_name
    )

    ax.xaxis.set_major_locator(MaxNLocator(5))

    ####################################################################
    # Plot 2 - Cumulative Concentration
    ####################################################################

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
            linewidth=chart_format['line_width'],
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
            fontsize=chart_format['legend_size']
        )

    ax.set_title(
        '02  |  Concentration',
        loc='left',
        pad=chart_format['title_pad']
    )

    ax.set_xlabel('Cumulative % of Members')
    ax.set_ylabel('Cumulative % of Value')

    ####################################################################
    # Plot 3 - Segment Boundary Comparison
    ####################################################################

    ax = axes[2]

    seg = breaks_df[f'{column_name}_SEG']
    seg1 = breaks_df[f'{column_name}_SEG1']

    if use_log:
        seg = np.log1p(seg)
        seg1 = np.log1p(seg1)

    x = np.arange(1, len(breaks_df) + 1)

    ax.plot(
        x,
        seg,
        color=secondary,
        linewidth=chart_format['line_width'],
        marker='o',
        markersize=chart_format['marker_size'],
        label='Including Zeros'
    )

    ax.plot(
        x,
        seg1,
        color=primary,
        linewidth=chart_format['line_width'],
        marker='s',
        markersize=chart_format['marker_size'],
        label='Excluding Zeros'
    )

    ax.set_title(
        '03  |  Segment Boundaries',
        loc='left',
        pad=chart_format['title_pad']
    )

    ax.set_xlabel('Segment')
    ax.set_ylabel(
        f'Log(1 + {column_name})'
        if use_log else column_name
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(
        loc='upper left',
        frameon=False,
        fontsize=chart_format['legend_size']
    )

    ####################################################################
    # Figure Title
    ####################################################################

    if print_:
        display(Markdown("""
        Chart 1: Visualizes a Distirbution of values Across the population.
        Chart 2: Plots a concentration analysis, showing the distribution of cummulative value over the population.
        Chart 3: Plots the Segment Points (Deciles).
        """))
        
    fig.suptitle(
        f'Concentration Analysis  |  {column_name}',
        fontsize=chart_format['title_size'],
        fontweight='bold',
        color=primary
    )

    return fig
