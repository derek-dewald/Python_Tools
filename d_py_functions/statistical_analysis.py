'''
module_word: statistical_analysis
module_definition: Repository for stand alone statistical analysis functions.

'''


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

    # Standard error
    result['STANDARD_ERROR'] = result[std_col] / np.sqrt(n)

    # T statistic
    result['T_STAT'] = (
        (result[observed_col] - result[mean_col])
        / result['STANDARD_ERROR']
    )

    # P-value
    if alternative == "two-sided":
        result['P_VALUE'] = 2 * stats.t.sf(
            np.abs(result['T_STAT']), df=dfree
        )

    elif alternative == "greater":
        result['P_VALUE'] = stats.t.sf(
            result['T_STAT'], df=dfree
        )

    else:
        result['P_VALUE'] = stats.t.cdf(
            result['T_STAT'], df=dfree
        )

    # Invalid statistical inputs
    invalid = (
        (n < 2)
        | (result[std_col] <= 0)
        | n.isna()
        | result[std_col].isna()
    )

    result.loc[invalid, ['T_STAT', 'P_VALUE']] = np.nan

    # Statistical significance
    alpha = 1 - confidence_level
    result['NULL_REJECTED'] = (
        result['P_VALUE'] < alpha
    ).astype(int)

    # Direction
    result['OBS_GT_MEAN'] = (
        result[observed_col] > result[mean_col]
    ).astype(int)

    # Business flag
    if alternative == "two-sided":
        result['FLAG'] = result['NULL_REJECTED']

    elif alternative == "greater":
        result['FLAG'] = (
            result['NULL_REJECTED'] * result['OBS_GT_MEAN']
        )

    else:
        result['FLAG'] = (
            result['NULL_REJECTED'] * (1 - result['OBS_GT_MEAN'])
        )

    # Flag missing reference means
    result['FLAG'] = np.where(
        result[mean_col].isna(), 1, result['FLAG']
    )

    # Summary
    flagged = result['FLAG'].sum()

    print(
        f"Number of Flagged Transactions: {flagged}, "
        f"Percentage of Flagged Transactions: {flagged / len(result):.2%}"
        if len(result) else "No transactions to evaluate."
    )

    return result