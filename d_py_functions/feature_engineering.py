'''
module_word: feature_engineering
module_definition: Repo for functions which create data elements within the context of an existing Data Frame. Does not include the creation of New Data Sets.

'''

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