'''
module_word: eda_functions
module_definition: Repo for functions required to implement Exploratory Data Analysis in Machine Learning Lifecycle

'''

from IPython.display import Markdown,display
from object_rendering import visualize_dataframe_in_notebook
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def review_dataset_lvl0( df):
    '''
    Definition:
        Function used to provide a level 0 assessment of the data to understand a mimimum state of the dataset. 
        Used as Step 1 in Exploratory Data Analysis.
    Parameters:
        df(dataframe): Dataset.
    Returns:
        rendered_object
    Date Created:
        09-Oct-26
    Date Last Modified:
        03-Nov-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        test = review_dataset_lvl0(df)
    Notes:
        None
    Required Functions:
        visualize_dataframe_in_notebook
        display
        Markdown
    
    '''
    records = []
    
    for column in df.columns:
    
        s = df[column]
        numeric = pd.api.types.is_numeric_dtype(s)
    
        is_empty_record = s[(s.isna())|((s==""))]
    
        record = {
            'VARIABLE': column,
            'DTYPE': str(s.dtype),
            'COUNT': s.count(),
            'UNIQUE': s.nunique(dropna=True),
            'NULL_COUNT':s.isna().sum(),           
            'ZERO_COUNT':(s==0).sum(),
            'BLANK_COUNT':(s=='').sum(),
            'NULL_BLANK_COUNT': len(is_empty_record),
            'NULL_BLANK_PCT': (len(is_empty_record) / len(s))*100 if s.count() > 0 else np.nan,
        }
        text_columns = list(record.keys())

        if s.dtype == 'object':
            types = s.dropna().map(type).nunique()
            if types > 1:
                record['MULTIPLE_DATA_TYPES'] = 1
        else:
            record['MULTIPLE_DATA_TYPES'] = 0
    
        records.append(record)
        
    result = pd.DataFrame(records)

    display(Markdown('## Results of Level 0 Data Quality Review'))
    display(Markdown('''### 
    Please Review each of the Dataframes. The following unique dataframes have been created, please review each before proceeding
    Columns with Missing Data: Must be Imputed, Deleted, or Acknowledged that Missing Data is not a problem.
    Columns with Multiple Data Types: Clean up, or accept.
    Columns with a Single Value: Must validate to make sure that only a single value is acceptable. Almost certainly it can be removed.
    Columns with Zero Values: Do not explicitly need to change, but should review and validate.
    
    '''))

    is_null_blank = result[result['NULL_BLANK_COUNT']>0]
    temp_df = result[result['NULL_BLANK_COUNT']==0]

    multiple_dt = temp_df[temp_df['MULTIPLE_DATA_TYPES']==1]
    temp_df = temp_df[temp_df['MULTIPLE_DATA_TYPES']==0]

    one_val = temp_df[temp_df['UNIQUE']==1]
    temp_df = temp_df[temp_df['UNIQUE']!=1]
    
    zero_df = temp_df[temp_df['ZERO_COUNT']>0]
    temp_df = temp_df[temp_df['ZERO_COUNT']==0]
    
    if len(is_null_blank)>0:
        display(Markdown('## Columns with Missing Data'))
        visualize_dataframe_in_notebook(is_null_blank)

    if len(multiple_dt)>0:
        display(Markdown('## Columns with Multiple Data Types'))
        visualize_dataframe_in_notebook(multiple_dt)

    if len(zero_df)>0:
        display(Markdown('## Columns with a Single Value'))
        visualize_dataframe_in_notebook(one_val)
    
    if len(zero_df)>0:
        display(Markdown('## Columns with Zero Valued Records'))
        visualize_dataframe_in_notebook(zero_df)
    
    display(Markdown('## All Remaining Records Not Flagged Order based on Number of Unqiue Observations'))
    visualize_dataframe_in_notebook(temp_df.sort_values('UNIQUE'))
    

    return result



def review_dataset_lvl1( df, top_n=5, min_top_count=10 ):
    '''
    Definition:
        Function used to provide a level 1 assessment of the data to understand a mimimum state of the dataset. 
        Used as Step 1 in Exploratory Data Analysis.
    Parameters:
        df(dataframe): Dataset.
        top_n(int): Number of Variables to include the Top Observation in right most column of Dataset.
        min_top_count(int): Minimum Number of records which must be present in dataset in order to include in output.
    Returns:
        dict, df
    Date Created:
        09-Oct-26
    Date Last Modified:
        09-Oct-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        test = review_dataset_lvl1(df)
    Notes:
        None
    Required Functions:
        visualize_dataframe_in_notebook
        display
        Markdown
    
    '''
    records = []
    
    for column in df.columns:
    
        s = df[column]
        numeric = pd.api.types.is_numeric_dtype(s)
    
        is_empty_record = s[(s.isna())|((s==""))]
    
        record = {
            'VARIABLE': column,
            'DTYPE': str(s.dtype),
            'COUNT': s.count(),
            'UNIQUE': s.nunique(dropna=True),
            'NULL_COUNT':s.isna().sum(),           
            'ZERO_COUNT':(s==0).sum(),
            'BLANK_COUNT':(s=='').sum(),
            'NULL_BLANK_COUNT': len(is_empty_record),
            'NULL_BLANK_PCT': (len(is_empty_record) / len(s))*100 if s.count() > 0 else np.nan,
        }
        text_columns = list(record.keys())
    
        if numeric:
            q01 = s.quantile(.01)
            q05 = s.quantile(.05)
            q25 = s.quantile(.25)
            q50 = s.quantile(.50)
            q75 = s.quantile(.75)
            q95 = s.quantile(.95)
            q99 = s.quantile(.99)
    
            record.update({
                # Central tendency
                'MEAN': s.mean(),
                'MEDIAN': q50,
    
                # Variation
                'STD': s.std(),
    
                # Range / Percentiles
                'Q1': q25,
                'Q2': q50,
                'Q3': q75,            
                'MIN': s.min(),
                'MAX': s.max(),
                'P01': q01,
                'P05': q05,
                'P95': q95,
                'P99': q99
            })
                
        vc = s.value_counts(dropna=False)

        vc = vc[vc >= min_top_count]
        
        for i in range(top_n):
            record[f'TOP_{i+1}'] = np.nan
    
        for i, (value, count) in enumerate(vc.head(top_n).items()):
            pct = count / len(s) * 100
            record[f'TOP_{i+1}'] = f'{value}: {count:,} ({pct:.2f}%)'
        records.append(record)
        
    result = pd.DataFrame(records)

    try:
        cols = list(result.drop([ 'TOP_1', 'TOP_2','TOP_3', 'TOP_4', 'TOP_5'],axis=1).columns)
        cols.extend([ 'TOP_1', 'TOP_2','TOP_3', 'TOP_4', 'TOP_5'])
        result = result[cols]
    except:
        pass
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    numerical_col_df = result[result['VARIABLE'].isin(numeric_columns)]
    text_columns.extend([f'TOP_{i+1}' for i in range(top_n)])
    text_col_df      = result[~result['DTYPE'].isin(['int64','float64'])][text_columns]
    
    display(Markdown('## Results'))
    display(Markdown('## Numeric Columns'))
    visualize_dataframe_in_notebook(
        numerical_col_df,
        show_as_int=['ZERO_COUNT','NULL_COUNT','UNIQUE','COUNT'],
        show_as_number=['MEAN','MEDIAN','STD','IQR','SKEW','MIN','MAX','P01','P05','Q1','Q2','Q3','P95','P99']
    )

    numeric_graph_objects = []
    for column in numerical_col_df['VARIABLE'].unique().tolist():
        numeric_graph_objects.append(plot_numeric_histogram(df,column))
    plot_visualization_grid(numeric_graph_objects)
    
    display(Markdown('## Non Numeric Columns'))
    
    visualize_dataframe_in_notebook(
        text_col_df)

    # Visualize Graphs
    # Numeric

    # Non Numerci
    text_graph_objects = []
    for column in text_col_df['VARIABLE'].unique().tolist():
        text_graph_objects.append(plot_categorical_bar_frequency_distribution(df,column))
    plot_visualization_grid(text_graph_objects)

    
    return {"Numerical Columns":numerical_col_df,'Text Columns':text_col_df}

def plot_categorical_bar_frequency_distribution(df, column, top_n=10, ax=None):

    counts = (
        df[column]
        .astype('object')
        .fillna('Missing')
        .astype(str)
        .value_counts()
    )

    if len(counts) > top_n:
        other_count = counts.iloc[top_n:].sum()
        counts = counts.head(top_n).sort_index()
        counts = pd.concat([counts, pd.Series({'Other': other_count})])
    else:
        counts = counts.sort_index()

    percentages = counts / len(df) * 100 if len(df) else counts.astype(float)

    def draw(ax):
        bars = ax.barh(
            percentages.index,
            percentages.values,
            color='steelblue'
        )

        for bar, pct in zip(bars, percentages.values):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%',
                va='center',
                fontsize=9
            )

        ax.invert_yaxis()
        ax.set_title(f'Distribution of Records in Column: {column}', fontsize=11)
        ax.set_xlabel('')
        ax.set_xlim(
            0,
            max(percentages.max() * 1.15, 5) if len(percentages) else 5
        )
        ax.grid(axis='x', alpha=0.2)
        ax.set_axisbelow(True)

        return ax

    if ax is not None:
        return draw(ax)

    return draw

def plot_numeric_histogram(df, column, bins=30, ax=None):

    values = pd.to_numeric(df[column], errors='coerce').dropna()

    def draw(ax):
        counts, bin_edges = np.histogram(values, bins=bins)

        percentages = (
            counts / len(df) * 100
            if len(df)
            else counts.astype(float)
        )

        ax.bar(
            bin_edges[:-1],
            percentages,
            width=np.diff(bin_edges),
            align='edge',
            color='steelblue',
            edgecolor='white',
            alpha=0.85
        )

        ax.set_title(
            f'Distribution of Records in Column: {column}',
            fontsize=11
        )
        #ax.set_xlabel(column)
        ax.set_ylabel('Percentage (%)')

        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)

        return ax

    if ax is not None:
        return draw(ax)

    return draw



def plot_visualization_grid(visualizations, ncols=3, figsize=(6, 4)):

    nplots = len(visualizations)

    if nplots == 0:
        raise ValueError("No visualizations supplied.")

    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        squeeze=False,
        layout='constrained'  # Automatically accommodates axis labels
    )

    for ax, visualization in zip(axes.flat, visualizations):
        visualization(ax)

    # Hide unused subplots
    for ax in axes.flat[nplots:]:
        ax.set_visible(False)

    # Additional spacing between charts
    fig.get_layout_engine().set(
        w_pad=0.5,
        h_pad=0.3,
        wspace=0.15,
        hspace=0.15
    )

    plt.show()

    return fig, axes



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, Markdown


# ============================================================
# HELPER: CLASSIFY DISTRIBUTION
# ============================================================

def classify_distribution(skew):
    """
    Classify the approximate shape of a numeric distribution
    using skewness.
    """

    if pd.isna(skew):
        return 'Undefined'
    elif skew < -1:
        return 'Highly Left Skewed'
    elif skew < -0.5:
        return 'Moderately Left Skewed'
    elif skew <= 0.5:
        return 'Approximately Symmetric'
    elif skew <= 1:
        return 'Moderately Right Skewed'
    else:
        return 'Highly Right Skewed'


# ============================================================
# HELPER: RECOMMEND LOGARITHMIC SCALE
# ============================================================

def recommend_log_scale(s, threshold=20):
    """
    Recommend logarithmic scaling when the positive values
    span a sufficiently wide range.

    Uses the ratio of the 99th percentile to the median
    of positive observations.

    Does not recommend standard log scaling for variables
    containing negative values.
    """

    values = pd.to_numeric(
        s, errors='coerce'
    ).dropna()

    values = values[np.isfinite(values)]

    if len(values) < 2:
        return False

    if (values < 0).any():
        return False

    positive = values[values > 0]

    if len(positive) < 2:
        return False

    p50 = positive.quantile(.50)
    p99 = positive.quantile(.99)

    if p50 <= 0:
        return False

    return bool((p99 / p50) > threshold)


# ============================================================
# LEVEL 1 DATASET REVIEW
# ============================================================

def review_dataset_lvl1(df, top_n=5, min_top_count=10):
    '''
    Definition:
        Function used to provide a Level 1 assessment of
        the dataset through descriptive statistics,
        distribution analysis and visualizations.

    Parameters:
        df(dataframe): Dataset.
        top_n(int): Number of top observations to include.
        min_top_count(int): Minimum frequency required
                           for top observations.

    Returns:
        dict:
            Numerical Columns
            Text Columns

    Process:
        1. Calculate descriptive statistics.
        2. Classify numeric distribution shape.
        3. Recommend logarithmic visualization.
        4. Display numeric and categorical summaries.
        5. Plot standard distributions (N x 3).
        6. Plot linear/log comparisons (N x 2).
        7. Plot categorical distributions (N x 3).

    Required Functions:
        visualize_dataframe_in_notebook
        classify_distribution
        recommend_log_scale
        plot_numeric_histogram
        plot_categorical_bar_frequency_distribution
        plot_visualization_grid
    '''

    records = []

    for column in df.columns:

        s = df[column]
        numeric = pd.api.types.is_numeric_dtype(s)

        record = {
            'VARIABLE': column,
            'DTYPE': str(s.dtype),
            'COUNT': s.count(),
            'UNIQUE': s.nunique(dropna=True)
        }

        # ----------------------------------------------------
        # Numeric descriptive statistics
        # ----------------------------------------------------

        if numeric:

            q01 = s.quantile(.01)
            q05 = s.quantile(.05)
            q25 = s.quantile(.25)
            q50 = s.quantile(.50)
            q75 = s.quantile(.75)
            q95 = s.quantile(.95)
            q99 = s.quantile(.99)

            skew = s.skew()

            record.update({

                # Central tendency
                'MEAN': s.mean(),
                'MEDIAN': q50,

                # Variation
                'STD': s.std(),
                'IQR': q75 - q25,
                'SKEW': skew,
                'KURTOSIS': s.kurt(),

                # Distribution characteristics
                'DISTRIBUTION': classify_distribution(skew),
                'LOG_RECOMMENDED': recommend_log_scale(s),
                'ZERO_PCT': s.eq(0).mean() * 100,
                'ZERO_INFLATED': s.eq(0).mean() > .30,

                # Range / Percentiles
                'Q1': q25,
                'Q2': q50,
                'Q3': q75,
                'MIN': s.min(),
                'MAX': s.max(),
                'P01': q01,
                'P05': q05,
                'P95': q95,
                'P99': q99
            })

        # ----------------------------------------------------
        # Top observations
        # ----------------------------------------------------

        vc = s.value_counts(dropna=False)
        vc = vc[vc >= min_top_count]

        for i in range(top_n):
            record[f'TOP_{i+1}'] = np.nan

        for i, (value, count) in enumerate(
            vc.head(top_n).items()
        ):
            pct = count / len(s) * 100
            record[f'TOP_{i+1}'] = (
                f'{value}: {count:,} ({pct:.2f}%)'
            )

        records.append(record)

    result = pd.DataFrame(records)

    # --------------------------------------------------------
    # Organize columns
    # --------------------------------------------------------

    top_columns = [
        f'TOP_{i+1}' for i in range(top_n)
    ]

    result = result[
        [c for c in result.columns if c not in top_columns]
        + top_columns
    ]

    numeric_columns = [
        column for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    numerical_col_df = result[
        result['VARIABLE'].isin(numeric_columns)
    ].drop(columns=['DTYPE', 'COUNT'])

    text_columns = [
        'VARIABLE', 'DTYPE', 'COUNT', 'UNIQUE'
    ] + top_columns

    text_col_df = result[
        ~result['VARIABLE'].isin(numeric_columns)
    ][text_columns]

    # --------------------------------------------------------
    # Display numeric summary
    # --------------------------------------------------------

    display(Markdown('## Results'))
    display(Markdown('## Numeric Columns'))

    visualize_dataframe_in_notebook(
        numerical_col_df,
        show_as_int=['UNIQUE'],
        show_as_number=[
            'MEAN', 'MEDIAN', 'STD', 'IQR',
            'SKEW', 'KURTOSIS', 'ZERO_PCT',
            'MIN', 'MAX', 'P01', 'P05',
            'Q1', 'Q2', 'Q3', 'P95', 'P99'
        ]
    )

    # --------------------------------------------------------
    # Separate numeric visualizations
    # --------------------------------------------------------

    standard_graphs = []
    log_comparison_graphs = []

    for column in numerical_col_df['VARIABLE'].unique():

        if recommend_log_scale(df[column]):

            # Linear graph
            log_comparison_graphs.append(
                plot_numeric_histogram(
                    df,
                    column,
                    log_scale=False
                )
            )

            # Logarithmic graph
            log_comparison_graphs.append(
                plot_numeric_histogram(
                    df,
                    column,
                    log_scale=True
                )
            )

        else:

            standard_graphs.append(
                plot_numeric_histogram(
                    df,
                    column,
                    log_scale=False
                )
            )

    # --------------------------------------------------------
    # Standard distributions: N x 3
    # --------------------------------------------------------

    if standard_graphs:

        display(Markdown(
            '### Standard Distributions'
        ))

        plot_visualization_grid(
            standard_graphs,
            ncols=3,
            figsize=(6, 4)
        )

    # --------------------------------------------------------
    # Logarithmic comparisons: N x 2
    # --------------------------------------------------------

    if log_comparison_graphs:

        display(Markdown(
            '### Logarithmic Distribution Comparisons'
        ))

        plot_visualization_grid(
            log_comparison_graphs,
            ncols=2,
            figsize=(7, 4)
        )

    # --------------------------------------------------------
    # Non-numeric columns
    # --------------------------------------------------------

    display(Markdown('## Non Numeric Columns'))


    text_graph_objects = []

    for column in [x for x in text_col_df['VARIABLE'].unique() if x not in ['MEMBERNBR']]:

        text_graph_objects.append(
            plot_categorical_bar_frequency_distribution(
                df,
                column
            )
        )

    if text_graph_objects:

        plot_visualization_grid(
            text_graph_objects,
            ncols=3,
            figsize=(6, 4)
        )

    return {
        'Numerical Columns': numerical_col_df,
        'Text Columns': text_col_df
    }


# ============================================================
# CATEGORICAL FREQUENCY DISTRIBUTION
# ============================================================

def plot_categorical_bar_frequency_distribution(
    df, column, top_n=10, ax=None
):

    counts = (
        df[column]
        .astype('object')
        .fillna('Missing')
        .astype(str)
        .value_counts()
    )

    if len(counts) > top_n:

        other_count = counts.iloc[top_n:].sum()

        counts = counts.head(top_n).sort_index()

        counts = pd.concat([
            counts,
            pd.Series({'Other': other_count})
        ])

    else:
        counts = counts.sort_index()

    percentages = (
        counts / len(df) * 100
        if len(df)
        else counts.astype(float)
    )

    def draw(ax):

        bars = ax.barh(
            percentages.index,
            percentages.values,
            color='steelblue'
        )

        for bar, pct in zip(
            bars, percentages.values
        ):

            ax.text(
                bar.get_width() + .5,
                bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%',
                va='center',
                fontsize=9
            )

        ax.invert_yaxis()

        ax.set_title(
            f'Distribution of Records in Column: {column}',
            fontsize=11
        )

        ax.set_xlabel('')

        ax.set_xlim(
            0,
            max(percentages.max() * 1.15, 5)
            if len(percentages) else 5
        )

        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)

        return ax

    if ax is not None:
        return draw(ax)

    return draw


# ============================================================
# NUMERIC HISTOGRAM
# ============================================================

def plot_numeric_histogram(
    df, column, bins=30, ax=None, log_scale=False
):

    values = pd.to_numeric(
        df[column], errors='coerce'
    ).dropna()

    values = values[np.isfinite(values)]

    if log_scale:
        values = values[values > 0]

    def draw(ax):

        if len(values) == 0:

            ax.set_title(
                f'{column} - No Valid Observations',
                fontsize=11
            )

            ax.text(
                .5, .5,
                'No valid observations',
                ha='center',
                va='center',
                transform=ax.transAxes
            )

            return ax

        # ----------------------------------------------------
        # Determine histogram bins
        # ----------------------------------------------------

        if log_scale and values.min() < values.max():

            bin_edges = np.geomspace(
                values.min(),
                values.max(),
                bins + 1
            )

        else:
            bin_edges = bins

        # ----------------------------------------------------
        # Draw histogram
        # ----------------------------------------------------

        ax.hist(
            values,
            bins=bin_edges,
            weights=np.full(
                len(values),
                100 / len(df)
            ),
            color='steelblue',
            edgecolor='white',
            alpha=.85
        )

        if log_scale and values.min() < values.max():
            ax.set_xscale('log')

        scale_label = (
            'Log' if log_scale else 'Linear'
        )

        ax.set_title(
            f'{column} ({scale_label})',
            fontsize=11
        )

        ax.set_ylabel('Percentage (%)')

        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)

        return ax

    if ax is not None:
        return draw(ax)

    return draw


# ============================================================
# VISUALIZATION GRID
# ============================================================

def plot_visualization_grid_(
    visualizations, ncols=3, figsize=(6, 4)
):

    nplots = len(visualizations)

    if nplots == 0:
        raise ValueError(
            "No visualizations supplied."
        )

    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            figsize[0] * ncols,
            figsize[1] * nrows
        ),
        squeeze=False,
        layout='constrained'
    )

    for ax, visualization in zip(
        axes.flat, visualizations
    ):
        visualization(ax)

    # Hide unused subplots
    for ax in axes.flat[nplots:]:
        ax.set_visible(False)

    # Additional spacing between charts
    fig.get_layout_engine().set(
        w_pad=.5,
        h_pad=.3,
        wspace=.15,
        hspace=.15
    )

    plt.show()

    return fig, axes


# ============================================================
# USAGE
# ============================================================

def plot_visualization_grid(
    visualizations, ncols=3, figsize=(6, 4), max_rows=4
):

    nplots = len(visualizations)

    if nplots == 0:
        raise ValueError("No visualizations supplied.")

    plots_per_figure = ncols * max_rows

    for start in range(0, nplots, plots_per_figure):

        batch = visualizations[start:start + plots_per_figure]

        nrows = int(np.ceil(len(batch) / ncols))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                figsize[0] * ncols,
                figsize[1] * nrows
            ),
            squeeze=False,
            layout='constrained'
        )

        for ax, visualization in zip(axes.flat, batch):
            visualization(ax)

        for ax in axes.flat[len(batch):]:
            ax.set_visible(False)

        fig.get_layout_engine().set(
            w_pad=.5,
            h_pad=.3,
            wspace=.15,
            hspace=.15
        )

        plt.show()
        plt.close(fig)