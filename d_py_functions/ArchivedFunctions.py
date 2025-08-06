def CreateCalculatedField(df, primary_key, calc_instructions, include_all=1):

    '''
    calc_instructions = [
    {'type': 'sum', 'value1': 'LENDING', 'name': 'TOTAL_LENDING'},
    {'type': 'weighted_average', 'value1': 'LENDING', 'value2': 'INTEREST_RATE', 'name': 'WEIGHTED_INTEREST'},
    {'type': 'ratio', 'value1': 'RENEWED_AMOUNT', 'value2': 'MATURED_AMOUNT', 'name': 'RENEWAL_RATE'}
    ]

    output = CreateCalculatedField(final_df, ['BRANCHNAME', 'CITY', 'LOB', 'DURATION'], calc_instructions)

    

    Create as initially Used in Data Management Dashboard Accumulation. Was not pursed Overtly Complex.

    '''

    base_aggs = {}
    
    # Collect all fields we need
    for calc in calc_instructions:
        if calc['type'] == 'sum':
            base_aggs[calc['name']] = (calc['value1'], 'sum')
        elif calc['type'] == 'weighted_average':
            base_aggs[f"__{calc['name']}_NUM"] = (
                calc['value2'], lambda x, col=calc['value1']: (df.loc[x.index, col] * x).sum()
            )
            base_aggs[f"__{calc['name']}_DEN"] = (calc['value1'], 'sum')
        elif calc['type'] == 'ratio':
            base_aggs[f"__{calc['name']}_NUM"] = (calc['value1'], 'sum')
            base_aggs[f"__{calc['name']}_DEN"] = (calc['value2'], 'sum')

    # Base groupby
    grouped = df.groupby(primary_key, dropna=False).agg(**base_aggs).reset_index()

    # Compute post-aggregates
    for calc in calc_instructions:
        if calc['type'] == 'sum':
            continue
        elif calc['type'] == 'weighted_average':
            num = grouped[f"__{calc['name']}_NUM"]
            den = grouped[f"__{calc['name']}_DEN"]
            grouped[calc['name']] = np.where(den != 0, num / den, np.nan)
            grouped.drop(columns=[f"__{calc['name']}_NUM", f"__{calc['name']}_DEN"], inplace=True)
        elif calc['type'] == 'ratio':
            num = grouped[f"__{calc['name']}_NUM"]
            den = grouped[f"__{calc['name']}_DEN"]
            grouped[calc['name']] = np.where(den != 0, num / den, np.nan)
            grouped.drop(columns=[f"__{calc['name']}_NUM", f"__{calc['name']}_DEN"], inplace=True)

    result_frames = [grouped.copy()]

    # Rollup combinations
    if include_all:
        for r in range(1, len(primary_key)):
            for group_cols in combinations(primary_key, r):
                temp = df.copy()
                for col in primary_key:
                    if col not in group_cols:
                        temp[col] = 'All'
                temp_group = CreateCalculatedField(temp, primary_key, calc_instructions, include_all=0)
                result_frames.append(temp_group)

        # Full 'All' row
        temp = df.copy()
        for col in primary_key:
            temp[col] = 'All'
        temp_group = CreateCalculatedField(temp, primary_key, calc_instructions, include_all=0)
        result_frames.append(temp_group)

    return pd.concat(result_frames, ignore_index=True)