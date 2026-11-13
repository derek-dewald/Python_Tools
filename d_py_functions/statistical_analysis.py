'''
module_word: statistical_analysis
module_definition: Repository for stand alone statistical analysis functions, or those supporting statistical analysis.

'''

from object_rendering import visualize_dataframe_in_notebook
from IPython.display import Markdown,display

from scipy import stats
import pandas as pd
import numpy as np


def t_value_test(
    df,
    observed_col,
    mean_col,
    std_col,
    observations,
    confidence_level=0.95,
    alternative="two-sided"
):

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    result = df.copy()

    n = result[observations]
    dfree = n - 1

    # Standard error of the sample mean
    result["STANDARD_ERROR"] = (
        result[std_col] / np.sqrt(n)
    )

    # T statistic
    result["T_STAT"] = (
        (result[observed_col] - result[mean_col])
        / result["STANDARD_ERROR"]
    )

    # Identify invalid statistical inputs
    invalid = (
        (n < 2)
        | (result[std_col] <= 0)
        | n.isna()
        | result[std_col].isna()
        | result[observed_col].isna()
        | result[mean_col].isna()
        | ~np.isfinite(result["T_STAT"])
    )

    result.loc[invalid, ["STANDARD_ERROR", "T_STAT"]] = np.nan

    # P-value
    if alternative == "two-sided":
        result["P_VALUE"] = 2 * stats.t.sf(
            np.abs(result["T_STAT"]), df=dfree
        )

    elif alternative == "greater":
        result["P_VALUE"] = stats.t.sf(
            result["T_STAT"], df=dfree
        )

    else:
        result["P_VALUE"] = stats.t.cdf(
            result["T_STAT"], df=dfree
        )

    result.loc[invalid, "P_VALUE"] = np.nan

    # Statistical significance
    alpha = 1 - confidence_level

    result["NULL_REJECTED"] = (
        result["P_VALUE"] < alpha
    ).astype(int)

    # Direction
    result["OBS_GT_MEAN"] = (
        result[observed_col] > result[mean_col]
    ).astype(int)

    # Business flag
    result["FLAG"] = result["NULL_REJECTED"]

    # Preserve your original rule:
    # Missing reference means are flagged for review
    result["FLAG"] = np.where(
        result[mean_col].isna(), 1, result["FLAG"]
    )

    # Summary
    flagged = result["FLAG"].sum()

    print(
        f"Number of Flagged Records: {flagged}, "
        f"Percentage of Flagged Records: {flagged / len(result):.2%}"
        if len(result) else "No records to evaluate."
    )

    return result
    
def welch_t_test(
    df,
    observed_mean_col,
    observed_std_col,
    observed_n_col,
    reference_mean_col,
    reference_std_col,
    reference_n_col,
    confidence_level=0.95,
    alternative="two-sided"
):

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    result = df.copy()

    # Observed group
    mean1 = result[observed_mean_col]
    std1 = result[observed_std_col]
    n1 = result[observed_n_col]

    # Reference group
    mean2 = result[reference_mean_col]
    std2 = result[reference_std_col]
    n2 = result[reference_n_col]

    # Variance components
    var1 = (std1 ** 2) / n1
    var2 = (std2 ** 2) / n2

    # Standard error of the difference in means
    result["STANDARD_ERROR"] = np.sqrt(var1 + var2)

    # Welch T statistic
    result["T_STAT"] = (
        (mean1 - mean2) / result["STANDARD_ERROR"]
    )

    # Welch-Satterthwaite degrees of freedom
    numerator = (var1 + var2) ** 2

    denominator = (
        (var1 ** 2) / (n1 - 1)
        + (var2 ** 2) / (n2 - 1)
    )

    result["DEGREES_OF_FREEDOM"] = numerator / denominator

    # Invalid statistical inputs
    invalid = (
        (n1 < 2)
        | (n2 < 2)
        | (std1 < 0)
        | (std2 < 0)
        | mean1.isna()
        | mean2.isna()
        | std1.isna()
        | std2.isna()
        | n1.isna()
        | n2.isna()
        | (result["STANDARD_ERROR"] <= 0)
        | ~np.isfinite(result["T_STAT"])
        | ~np.isfinite(result["DEGREES_OF_FREEDOM"])
    )

    result.loc[
        invalid,
        ["STANDARD_ERROR", "T_STAT", "DEGREES_OF_FREEDOM"]
    ] = np.nan

    # P-value
    if alternative == "two-sided":
        result["P_VALUE"] = 2 * stats.t.sf(
            np.abs(result["T_STAT"]),
            df=result["DEGREES_OF_FREEDOM"]
        )

    elif alternative == "greater":
        result["P_VALUE"] = stats.t.sf(
            result["T_STAT"],
            df=result["DEGREES_OF_FREEDOM"]
        )

    else:
        result["P_VALUE"] = stats.t.cdf(
            result["T_STAT"],
            df=result["DEGREES_OF_FREEDOM"]
        )

    result.loc[invalid, "P_VALUE"] = np.nan

    # Statistical significance
    alpha = 1 - confidence_level

    result["NULL_REJECTED"] = (
        result["P_VALUE"] < alpha
    ).astype(int)

    # Direction
    result["OBS_GT_MEAN"] = (
        mean1 > mean2
    ).astype(int)

    # Business flag
    result["FLAG"] = result["NULL_REJECTED"]

    # Flag missing reference means for review
    result["FLAG"] = np.where(
        mean2.isna(), 1, result["FLAG"]
    )

    # Summary
    flagged = result["FLAG"].sum()

    print(
        f"Number of Flagged Records: {flagged}, "
        f"Percentage of Flagged Records: {flagged / len(result):.2%}"
        if len(result) else "No records to evaluate."
    )

    return result



def predictive_t_test(
    df,
    observed_col,
    mean_col,
    std_col,
    observations,
    confidence_level=0.95,
    alternative="two-sided"
):
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("Invalid alternative")

    result = df.copy()

    n = result[observations]
    dfree = n - 1

    # Predictive standard error
    result["STANDARD_ERROR"] = (
        result[std_col] * np.sqrt(1 + 1 / n)
    )

    result["T_STAT"] = (
        (result[observed_col] - result[mean_col])
        / result["STANDARD_ERROR"]
    )

    if alternative == "two-sided":
        result["P_VALUE"] = 2 * stats.t.sf(
            np.abs(result["T_STAT"]), df=dfree
        )
    elif alternative == "greater":
        result["P_VALUE"] = stats.t.sf(
            result["T_STAT"], df=dfree
        )
    else:
        result["P_VALUE"] = stats.t.cdf(
            result["T_STAT"], df=dfree
        )

    invalid = (
        (n < 2)
        | (result[std_col] <= 0)
        | n.isna()
        | result[std_col].isna()
        | result[observed_col].isna()
        | result[mean_col].isna()
    )

    result.loc[
        invalid, ["T_STAT", "P_VALUE"]
    ] = np.nan

    alpha = 1 - confidence_level

    result["NULL_REJECTED"] = (
        result["P_VALUE"] < alpha
    ).astype(int)

    result["OBS_GT_MEAN"] = (
        result[observed_col] > result[mean_col]
    ).astype(int)

    if alternative == "two-sided":
        result["FLAG"] = result["NULL_REJECTED"]
    elif alternative == "greater":
        result["FLAG"] = (
            result["NULL_REJECTED"] *
            result["OBS_GT_MEAN"]
        )
    else:
        result["FLAG"] = (
            result["NULL_REJECTED"] *
            (1 - result["OBS_GT_MEAN"])
        )

    # Preserve your business rule
    result["FLAG"] = np.where(
        result[mean_col].isna(),
        1,
        result["FLAG"]
    )

    flagged = result["FLAG"].sum()

    print(
        f"Number Flagged: {flagged}, "
        f"Percentage Flagged: {flagged / len(result):.2%}"
        if len(result) else "No observations to evaluate."
    )

    return result

def create_stratified_sample(
    df,
    column_list,
    flag_name=None,
    target_flag="TARGET_MBR",
    dropna=False,
    rng=None,
    review_distribution=False,
):
  
    '''
    
    Definition:
        TBD
    Parameters:
        df(df): Dataframe
        column_list(list): List of Columns to Match Target Group
        flag_name(str): Name of New Segment. If blank default to SAMPLE_FLAG.
        target_flag(str): Name of column in DF which is Target Group. MUST be included. Must be Binary.
        dropna(binary): Condition for Group By.
        rng(object): Random Number Generator. By Default will create with seed 42.
        review_distribution(Binary): OPtional argument to display some visualizations after processing, to help assess impact of creation.

    Returns:
        df

    Date Created:
        12-Nov-26
    
    Date Last Modified:
        12-Nov-26
    
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        member_df = create_stratified_sample(member_df,column_list = ['AGE_SEGMENT','DURATION_YEARS_SEGMENT'])
    Notes:
        None
    Required Functions:
        None
    '''

    if not rng:
        rng = np.random.default_rng(42)
    
    df = df.copy() # Create Copy of DF
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True) # Randomly Sort
    
    if not flag_name:
        flag_name = 'SAMPLE_FLAG'
        df[flag_name] = (df[target_flag]==0).astype(int)
    
    col_list = column_list + [target_flag]
    
    # Count target distribution
    target_counts = (
        df[df[target_flag] == 1]
        .groupby(col_list, dropna=dropna)
        .size()
        .reset_index(name="MAX_RECORDS")
    )
    
    # Always Want Target to be the Opposite
    target_counts[target_flag] = np.where(target_counts[target_flag]==1,0,1)
    
    df = df.merge(target_counts,on=list(target_counts.drop('MAX_RECORDS',axis=1).columns),how='left')
    df['MAX_RECORDS'] = np.where(df[flag_name].fillna(0)==1,df['MAX_RECORDS'].fillna(0),0)
    
    col_list = col_list + [flag_name]
    
    df['n'] = df.groupby(col_list, dropna=dropna).cumcount()+ 1
    df[flag_name] = (df['MAX_RECORDS']>=df['n']).astype(int)
    
    df['COUNT'] = 1
    
    pt =df.pivot_table(index=col_list,aggfunc='sum')[['COUNT']].reset_index().sort_values(['COUNT',flag_name],ascending=[False,False])

    if len(pt[(pt[target_flag]==1)&(pt[flag_name]==1)]):
        display(Markdown('### ERROR Members in Target and Qualification'))
        visualize_dataframe_in_notebook(pt[(pt[target_flag]==1)&(pt[flag_name]==1)])
        
    display(Markdown('### Summary of Total Records Flagged in Sample'))
    visualize_dataframe_in_notebook(pt.head(20))
    
    display(Markdown('### Summary of Total Records Flagged in Target'))
    visualize_dataframe_in_notebook(pt[pt['TARGET_MBR']==1].sort_values('COUNT',ascending=False).head(20))

    return df.drop(columns=["MAX_RECORDS", "n",'COUNT'])


def create_target_population(
    df,
    include_condition,
    percent_include=.5,
    new_column_name='TARGET_MBR',
    rng=None
):
    '''
    Definition:
        Creates a randomly selected target population based on
        specified eligibility conditions and inclusion probability.

    Parameters:
        df(df): DataFrame containing the population.
        include_condition(Series): Boolean condition identifying members eligible for selection.
        percent_include(float): Probability of including each eligible member. Defaults to 0.5 (50%).
        new_column_name(str): Name of the new binary target flag. Defaults to TARGET_MBR.
        rng(object): Random Number Generator. By default, creates a generator with seed 42.

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
        create_target_population(df=member_df,include_condition=(member_df['CITY'].isin(['Burnaby', 'Vancouver']) &(member_df['LEGACY_CU'] == 'Fishermans Friend')),percent_include=0.5)

    Notes:
        TBD
    Required Functions:
        numpy
        
    '''

    if rng is None:
        rng = np.random.default_rng(42)

    df[new_column_name] = (
        (percent_include > rng.uniform(size=len(df))) *
        include_condition
    ).astype(int)

    print(f"Total Number of Members in Randomly Selected Target Group: {sum(df[new_column_name])}")
