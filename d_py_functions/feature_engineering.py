'''
module_word: feature_engineering
module_definition: Repo for functions which create data elements within the context of an existing Data Frame. Does not include the creation of New Data Sets.

'''
from object_rendering import visualize_dataframe_in_notebook

import numpy as np
import pandas as pd

def binary_complex_equivalency(
    df,
    column_name,
    column_name1,
    new_column_name='EQ_FLAG',
    eq=1,
    ne=0,
    tolerance=.001,
    include_difference=False
):

    '''
    Definition:
        Function which tests the Equivalence of 2 Columns in a dataframe to determine whether they are Equal or Not.
    Parameters:
        df(dataframe): DataFrame
        column_name(str): Name of Column to be tested against column_name1.
        column_name1(str): Name of Column to be tested against column_name.
        new_column_name(str): Name of Created Column, representing where values are either Equal, or Not Equal. Default Name is EQ_FLAG
        eq(str): Any Value which will represent condition Matching.
        ne(str): Any value, representing Not Equal Condition
        tolerance(float): Degree of grace to be applied by np.is_close() to minimize python rounding errors.
        include_difference(bool): Boolean flag to determine whether to include arthematic difference in addition to binary flag.
        
    Returns:
        df
    Date Created:
        28-Aug-26
    Date Last Modified:
        28-Aug-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        binary_complex_equivalency(df,'SCORE1','SCORE2')
    Notes:
        None
    Required Functions:
        None
    
    
    
    '''
    
    s1 = df[column_name]
    s2 = df[column_name1]

    both_null = s1.isna() & s2.isna()

    if (
        pd.api.types.is_numeric_dtype(s1)
        and pd.api.types.is_numeric_dtype(s2)
    ):
        equal = np.isclose(
            s1,
            s2,
            atol=tolerance,
            rtol=0,
            equal_nan=True
        )
        if include_difference:
            df['COLUMN_DIFF'] = s1 - s2
            
    else:
        equal = s1.eq(s2) | both_null

        if include_difference:
            df['COLUMN_DIFF'] = np.nan

    df[new_column_name] = np.where(equal, eq, ne)

    return df

def create_segment_range(
    df,
    column_name,
    segment_range,
    new_column_name=None,
    print_=False
):
    """
    """
    if not new_column_name:
        new_column_name = f"{column_name}_SEGMENT"

    segment_range = sorted(segment_range)

    labels = [f"1) LTE {segment_range[0]}"]
    bins = [-float('inf')] + segment_range + [float('inf')]

    for i, (low, high) in enumerate(zip(segment_range[:-1], segment_range[1:]), start=2):
        labels.append(f"{i}) GT {low} - LTE {high}")

    labels.append(f"{len(segment_range) + 1}) GT {segment_range[-1]}")

    df[new_column_name] = pd.cut(
        df[column_name],
        bins=bins,
        labels=labels,
        right=True
    )
    if print_:
        display(df[new_column_name].value_counts().sort_index())



def lognormal_distribution_generator(base, std_dev=0.30, rng=None):
    """
    Takes a Number 

    Parameters
    ----------
    base : float or array-like
        Expected expense (arithmetic mean).
    std_dev : float
        Standard deviation as a proportion of the mean.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    float or ndarray
        Generated expense amounts.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    base = np.asarray(base, dtype=float)

    sigma = np.sqrt(np.log1p(std_dev**2))
    mu = -0.5 * sigma**2

    return base * rng.lognormal(
        mean=mu,
        sigma=sigma,
        size=base.shape
    )



def consolidate_relationship_grouper(
    df,
    parent_col,
    child_col
):

    """
        
    Definition:
        Identifies groups of interconnected members based on direct and indirect relationships using the Union-Find algorithm. Members are assigned to the same group when a relationship exists between them, either directly or through other members.

    Parameters:
        df(df):DataFrame containing member relationship pairs.
        parent_col (str): Column identifying the primary member.
        child_col (str): Column identifying the related member.

    Returns:
        df

    Process:
        TBD

    Categorization:
        TBD

    Usage:
        identify_connected_member_groups(df,parent,child)

    Notes:
        Relationships are treated as bidirectional.
        Indirect relationships are included (transitive grouping).
        Duplicate and circular relationships are handled.
        Members without valid relationships are excluded.
        Group IDs may change when the underlying data changes.
        Groups represent connected networks, not necessarily households or families.

    Required Functions:
        pandas
        
    """
    parent = {}
    size = {}

    def find(x):
        if x not in parent:
            parent[x] = x
            size[x] = 1

        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):
        ra, rb = find(a), find(b)

        if ra == rb:
            return

        if size[ra] < size[rb]:
            ra, rb = rb, ra

        parent[rb] = ra
        size[ra] += size[rb]

    # Build connected groups
    for a, b in df[[parent_col, child_col]].dropna().itertuples(
        index=False, name=None
    ):
        union(a, b)

    # Assign sequential group IDs
    roots = {x: find(x) for x in parent}
    root_ids = {
        root: f'GROUP_{i:06d}'
        for i, root in enumerate(sorted(set(roots.values()), key=str), 1)
    }

    return pd.DataFrame({
        'MEMBERNBR': list(roots),
        'GROUP_ID': [root_ids[r] for r in roots.values()]
    })


def df_column_map_text_patterns(
    df,
    column_name,
    new_column_name,
    mapping_dict,
    print_=False,
    default_value=None
):
    """
    Definition:
        Function which text mines a dataframe column and allows the application of a provisional dictionary to replace values into a new column, using .str.contains(na=False).
    Parameters:
        df(df): DataFrame
        column_name(str): Name of column which will have the .map applied to
        new_column_name(str): Name of New column which is mapped.
        mapping_dict(dict): Dictionary, which will be mapped to column and create new column name
        print_(binary): Determine whether a summary value count will be provided to validate Output. Default to Suppress.
        default_value(str): If you would like to override the default value where conditions are not Met. None will return the value from Column
        
    Returns:
        TBD
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
        TBD
    Required Functions:
        visualize_dataframe_in_notebook
        numpy

    
    """
    if not default_value:
        default_value = df[column_name]
    
    df = df.copy()
    conditions = [df[column_name].str.contains(key, na=False) for key in mapping_dict]
    df[new_column_name] = np.select(conditions,mapping_dict.values(),default=default_value)

    if print_:
        visualize_dataframe_in_notebook(pd.DataFrame(df[new_column_name].value_counts()),hide_index=False)
    
    return df


def log_transformation(
    df,
    column_name,
    new_column_name=None,
    include_negative=False
):
    '''
    Definition:
        Applies a logarithmic transformation to a numeric column, reducing the influence of extreme values while preserving relative differences between observations.

    Parameters:
        df(df): DataFrame containing the values to transform.
        column_name(str): Name of the numeric column to transform.
        new_column_name(str): Name of the transformed column. Defaults to the original column name with '_LOG' appended.
        include_negative(binary): Determines whether negative values are preserved using a signed logarithmic transformation. Defaults to False.

    Returns:
        None

    Date Created:
        13-Nov-26

    Date Last Modified:
        13-Nov-26

    Process:
        TBD

    Categorization:
        TBD

    Usage:
        og_transformation(df,column_name='TRANSACTION_VALUE',include_negative=True)

    Notes:
        TBD

    Required Functions:
        numpy
    '''

    if new_column_name is None:
        new_column_name = f"{column_name}_LOG"

    if include_negative:
        col = np.abs(df[column_name])
        sign = np.sign(df[column_name])
        df.loc[:, new_column_name] = np.log1p(col) * sign

    else:
        col = df[column_name].clip(lower=0)
        df.loc[:, new_column_name] = np.log1p(col)


def clip_distribution(
    df,
    column_name,
    new_column_name=None,
    head=None,
    tail=None
):
    '''
    Definition:
        Applies optional upper and lower boundaries to a numeric column, limiting extreme values without removing records. Values outside the specified boundaries are replaced with the nearest boundary value.

    Parameters:
        df(df): DataFrame containing the values to clip.
        column_name(str): Name of the numeric column to transform.
        new_column_name(str): Name of the resulting column. Defaults to the original column name with '_CLIPPED'.
        head(float): Upper boundary (ceiling). Values exceeding this threshold are replaced with the threshold. Defaults to None.
        tail(float): Lower boundary (floor). Values below this threshold are replaced with the threshold. Defaults to None.

    Returns:
        None

    Date Created:
        13-Nov-26

    Date Last Modified:
        13-Nov-26

    Process:
        TBD
    Categorization:
        TBD

    Usage:
        clip_distribution(df=member_df,column_name='TRANSACTION_VALUE',head=10000,tail=0)

    Notes:
        TBD

    Required Functions:
        pandas
    '''

    if new_column_name is None:
        new_column_name = f"{column_name}_CLIPPED"

    col = df[column_name]

    if head is not None:
        col = col.clip(upper=head)

    if tail is not None:
        col = col.clip(lower=tail)

    df[new_column_name] = col