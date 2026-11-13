import pandas as pd
import numpy as np

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from feature_engineering import create_segment_range,lognormal_distribution_generator
from utility_functions import net_present_value_calculator

age_dict = {'mean_age':59,'max_age':99,'min_age':12,'std_dev':20}

duration_dict = {'mean':14,'std_dev':5}

geography_dict = {
    'Vancouver': 0.19,'Surrey': 0.185,'Burnaby': 0.080,'Richmond': 0.065,'Abbotsford': 0.048,'Coquitlam': 0.047,
    'Kelowna': 0.045,'Langley': 0.044,'Delta': 0.034,'Saanich': 0.034,'Nanaimo': 0.030,'Kamloops': 0.029,'Chilliwack': 0.028,
    'Victoria': 0.027,'Maple Ridge': 0.025,'New Westminster': 0.024,'Prince George': 0.021,'Port Coquitlam': 0.018,
    'Vernon': 0.014,'Penticton': 0.012
}

branches = {
    'Vancouver': 7,'Surrey': 6,'Burnaby': 5,'Richmond': 4,'Abbotsford': 4,'Coquitlam': 3,'Kelowna': 5,
    'Langley': 3,'Delta': 2,'Saanich': 1,'Nanaimo': 1,'Kamloops': 1,'Chilliwack': 1,'Victoria': 1,
    'Maple Ridge': 1,'New Westminster': 1,'Prince George': 1,'Port Coquitlam': 1,'Vernon': 1,'Penticton':1
}

entity_map = {
    'Abbotsford Branch 1': 'Fishermans Friend',
    'Abbotsford Branch 2': 'Fishermans Friend',
    'Abbotsford Branch 3': 'Fishermans Friend',
    'Abbotsford Branch 4': 'Spa City',
    'Burnaby Branch 1': 'Fishermans Friend',
    'Burnaby Branch 2': 'Fishermans Friend',
    'Burnaby Branch 3': 'Fishermans Friend',
    'Burnaby Branch 4': 'Spa City',
    'Burnaby Branch 5': 'Spa City',
    'Chilliwack Branch 1': 'Fishermans Friend',
    'Coquitlam Branch 1': 'Fishermans Friend',
    'Coquitlam Branch 2': 'Fishermans Friend',
    'Coquitlam Branch 3': 'Spa City',
    'Delta Branch 1': 'Fishermans Friend',
    'Delta Branch 2': 'Spa City',
    'Kamloops Branch 1': 'K-Town',
    'Kelowna Branch 1': 'K-Town',
    'Kelowna Branch 2': 'K-Town',
    'Kelowna Branch 3': 'K-Town',
    'Kelowna Branch 4': 'K-Town',
    'Kelowna Branch 5': 'K-Town',
    'Langley Branch 1': 'Fishermans Friend',
    'Langley Branch 2': 'Fishermans Friend',
    'Langley Branch 3': 'Spa City',
    'Maple Ridge Branch 1': 'Fishermans Friend',
    'Nanaimo Branch 1': 'K-Town',
    'New Westminster Branch 1': 'Fishermans Friend',
    'Penticton Branch 1': 'K-Town',
    'Port Coquitlam Branch 1': 'Fishermans Friend',
    'Prince George Branch 1': 'K-Town',
    'Richmond Branch 1': 'Fishermans Friend',
    'Richmond Branch 2': 'Fishermans Friend',
    'Richmond Branch 3': 'Fishermans Friend',
    'Richmond Branch 4': 'Spa City',
    'Saanich Branch 1': 'K-Town',
    'Surrey Branch 1': 'Fishermans Friend',
    'Surrey Branch 2': 'Spa City',
    'Surrey Branch 3': 'Fishermans Friend',
    'Surrey Branch 4': 'Fishermans Friend',
    'Surrey Branch 5': 'Fishermans Friend',
    'Surrey Branch 6': 'Spa City',
    'Vancouver Branch 1': 'Fishermans Friend',
    'Vancouver Branch 2': 'Fishermans Friend',
    'Vancouver Branch 3': 'Fishermans Friend',
    'Vancouver Branch 4': 'Spa City',
    'Vancouver Branch 5': 'Fishermans Friend',
    'Vancouver Branch 6': 'Spa City',
    'Vancouver Branch 7': 'Fishermans Friend',
    'Vernon Branch 1': 'K-Town',
    'Victoria Branch 1': 'K-Town'
}

geography_multiplier = {
    'Vancouver': 1.4,'Surrey': 1.2,'Burnaby': 1.3,'Richmond': 1.2,'Abbotsford':1,'Coquitlam': 1.2,
    'Kelowna': .9,'Langley': 1,'Delta': .95,'Saanich': .9,'Nanaimo': 0.85,'Kamloops': 0.8,'Chilliwack': 0.9,
    'Victoria': 1,'Maple Ridge': .95,'New Westminster': 1,'Prince George': .85,'Port Coquitlam': 0.95,
    'Vernon': .8,'Penticton': 0.8
}

housing_price = {
    'mean':1250000,
    'min':500000,
    'std_dev':300000
    }

income_distribution = {
    '1) LTE 20': (8000, 0.65),
    '2) GT 20 - LTE 25': (28000, 0.50),
    '3) GT 25 - LTE 35': (65000, 0.45),
    '4) GT 35 - LTE 45': (80000, 0.45),
    '5) GT 45 - LTE 55': (85000, 0.50),
    '6) GT 55 - LTE 65': (100000, 0.55),
    '7) GT 65 - LTE 85': (42000, 0.55),
    '8) GT 85': (35000, 0.60)
}
homeownership_probability = {
    '1) LTE 20': 0.01,
    '2) GT 20 - LTE 25': 0.08,
    '3) GT 25 - LTE 35': 0.30,
    '4) GT 35 - LTE 45': 0.55,
    '5) GT 45 - LTE 55': 0.68,
    '6) GT 55 - LTE 65': 0.75,
    '7) GT 65 - LTE 85': 0.78,
    '8) GT 85': 0.72
}

car_ownership_probability = {
    '1) LTE 20': 0.05,
    '2) GT 20 - LTE 25': 0.15,
    '3) GT 25 - LTE 35': 0.6,
    '4) GT 35 - LTE 45': 0.85,
    '5) GT 45 - LTE 55': 0.95,
    '6) GT 55 - LTE 65': 0.85,
    '7) GT 65 - LTE 85': 0.65,
    '8) GT 85': 0.3
}

marital_status_distribution = {
    '1) LTE 20':          [0.99, 0.01, 0.00, 0.00],
    '2) GT 20 - LTE 25': [0.85, 0.14, 0.01, 0.00],
    '3) GT 25 - LTE 35': [0.50, 0.45, 0.05, 0.00],
    '4) GT 35 - LTE 45': [0.25, 0.60, 0.14, 0.01],
    '5) GT 45 - LTE 55': [0.18, 0.60, 0.20, 0.02],
    '6) GT 55 - LTE 65': [0.16, 0.58, 0.21, 0.05],
    '7) GT 65 - LTE 85': [0.07, 0.53, 0.18, 0.22],
    '8) GT 85':          [0.02, 0.28, 0.12, 0.58]
}

marital_statuses = [
    'Single',
    'Married',
    'Divorced',
    'Widowed'
]

marital_stats_multiplier = { 'Single':1,'Married':1.75,'Divorced':1,'Widowed':1.35}

zero_debt_probability = {
    '1) LTE 20': 0.85,
    '2) GT 20 - LTE 25': 0.40,
    '3) GT 25 - LTE 35': 0.20,
    '4) GT 35 - LTE 45': 0.15,
    '5) GT 45 - LTE 55': 0.20,
    '6) GT 55 - LTE 65': 0.35,
    '7) GT 65 - LTE 85': 0.60,
    '8) GT 85': 0.80
}

stock_vs_cash = {
    '1) LTE 20': 0.85,
    '2) GT 20 - LTE 25': 0.55,
    '3) GT 25 - LTE 35': 0.45,
    '4) GT 35 - LTE 45': 0.35,
    '5) GT 45 - LTE 55': 0.25,
    '6) GT 55 - LTE 65': 0.15,
    '7) GT 65 - LTE 85': 0.1,
    '8) GT 85': 0.05
}



car_ownership_probability = {
    '1) LTE 20': 0.05,
    '2) GT 20 - LTE 25': 0.15,
    '3) GT 25 - LTE 35': 0.6,
    '4) GT 35 - LTE 45': 0.85,
    '5) GT 45 - LTE 55': 0.95,
    '6) GT 55 - LTE 65': 0.85,
    '7) GT 65 - LTE 85': 0.65,
    '8) GT 85': 0.3
}

expenses_dict = {
    'baseline_rent':1750,
    'home_insurance_perc':.0015,
    'home_maint_cost':.00025,
    'property_tax_perc':.003,
    'utility_base_line_perc':.000095,
    'grocery_base_perc':.0075,
    'phone_internet':100,
    'dining_out_perc':.0025,
    'entertainment_perc':.005,
    'gas_perc':.002,
    'car_insurance_maint_perc':.004,
    'other_perc':.015,
}


market_interest_rates = {
    'MORTGAGE_DEBT':.04,                                 
    'LOC_DEBT':.055,
    'PERSONAL_LOAN_DEBT':.08,
    'CC_DEBT':.21,
    'CAR_LOAN_DEBT':.03
}

def calculate_member_profile(df):

    df = df.copy()

    debt_to_income = (
        df['TOTAL_DEBT'] /
        df['HOUSEHOLD_INCOME'].clip(lower=1)
    )

    condition = [
        df['NET_WORTH'] > 5_000_000,
        df['NET_WORTH'] > 2_000_000,
        debt_to_income > 3, # High leverage
        (df['NET_WORTH'] < 0) &(df['MONTHLY_CF'] < 0),# Financial vulnerability
        (df['NET_WORTH'] > 0) &(df['MONTHLY_CF'] <= 0),# Financially stretched

        # Accumulating
        (df['NET_WORTH'] > 100_000) &(df['MONTHLY_CF'] > 0),
        (df['NET_WORTH'] > 0) &(df['MONTHLY_CF'] > 0),
        (df['NET_WORTH'] < 0) &(df['MONTHLY_CF'] > 0),
        (df['NET_WORTH'] <= 0) &(df['MONTHLY_CF'] <= 0),
    
    
    ]

    values = [
        'Ultra Wealthy',
        'Affluent',
        'Highly Leveraged',
        'Financially Vulnerable',
        'Financially Stretched',
        'Accumulating',
        'Emerging',
        'Stabilizing',
        'Financially Precarious'
        
    ]

    df['MEMBER_PROFILE'] = np.select(
        condition,
        values,
        default='Not Defined'
    )

    return df

def distribute_member_debt(df, market_interest_rates,rng, concentration=15):

    df = df.copy()

    debt_columns = [
        'MORTGAGE_DEBT',
        'LOC_DEBT',
        'PERSONAL_LOAN_DEBT',
        'CC_DEBT',
        'CAR_LOAN_DEBT'
    ]

    # Quadrant allocation weights
    weights = np.array([
        [0.85, 0.10, 0.00, 0.00, 0.05],  # Q1
        [0.65, 0.12, 0.10, 0.08, 0.05],  # Q2
        [0.00, 0.25, 0.55, 0.05, 0.15],  # Q3
        [0.00, 0.15, 0.40, 0.35, 0.10]   # Q4
    ])

    homeowner = df['HOME_OWNER'].to_numpy() == 1
    positive_nw = df['NET_WORTH'].to_numpy() > 0
    has_vehicle = df['HAS_VEHICLE'].to_numpy() == 1

    # Determine quadrant
    quadrant = np.select(
        [
            homeowner & positive_nw,
            homeowner & ~positive_nw,
            ~homeowner & positive_nw,
            ~homeowner & ~positive_nw
        ],
        [0, 1, 2, 3]
    )

    # Member-specific allocation parameters
    alpha = weights[quadrant].copy()

    # No car loans without vehicle ownership
    alpha[~has_vehicle, 4] = 0

    # Convert weights into Dirichlet parameters
    alpha *= concentration

    # Gamma draws permit zero-weight categories
    draws = rng.gamma(
        shape=np.maximum(alpha, 0.001),
        scale=1.0
    )

    draws[alpha == 0] = 0

    proportions = draws / draws.sum(axis=1, keepdims=True)

    # Allocate existing debt
    total_debt = df['TOTAL_DEBT'].to_numpy()

    allocations = proportions * total_debt[:, None]

    df[debt_columns] = np.round(allocations, 2)

    # Reconcile rounding to preserve TOTAL_DEBT
    difference = total_debt - df[debt_columns].sum(axis=1)

    df['PERSONAL_LOAN_DEBT'] += difference
    df['MORTGAGE_RATE'] = lognormal_distribution_generator(pd.Series(market_interest_rates['MORTGAGE_DEBT'], index=df.index),std_dev=.01,rng=rng).clip(.02,.06)               
    df['LOC_RATE']      = lognormal_distribution_generator(pd.Series(market_interest_rates['LOC_DEBT'], index=df.index),std_dev=.015,rng=rng).clip(.03,.09)
    df['PL_RATE'] = lognormal_distribution_generator(pd.Series(market_interest_rates['PERSONAL_LOAN_DEBT'], index=df.index),std_dev=.02,rng=rng).clip(.04,.12)
    df['CC_RATE'] = lognormal_distribution_generator(pd.Series(market_interest_rates['CC_DEBT'], index=df.index),std_dev=.05,rng=rng).clip(.05,.3)
    df['CL_RATE'] = lognormal_distribution_generator(pd.Series(market_interest_rates['CAR_LOAN_DEBT'], index=df.index),std_dev=.02,rng=rng).clip(.02,.1)

    df['EAR'] = ((
        df['MORTGAGE_DEBT']*df['MORTGAGE_RATE'] +
        df['LOC_DEBT']*df['LOC_RATE'] +
        df['PERSONAL_LOAN_DEBT']*df['PL_RATE'] +
        df['CC_DEBT']*df['CC_RATE'] +
        df['CAR_LOAN_DEBT']*df['CL_RATE'] 
    )/df['TOTAL_DEBT']).fillna(0)

    df['MONTHLY_DEBT_PAYMENT'] = net_present_value_calculator(
       annual_interest=df['EAR'],
       total_amount=df['TOTAL_DEBT'],
       payments=300
    )
    
    return df

def car_ownership(df,car_ownership_probability,rng):
    probability = lognormal_distribution_generator(df['AGE_SEGMENT'].map(car_ownership_probability),rng=rng).clip(0,1)
    return (rng.random(len(df)) < probability).astype(int)

def generate_branch_distribution(branches,rng):
    
    branch_dict = {}
    
    for x in list(branches.keys()):
        temp_list = []
        for count in range(branches[x]):
            text = f"{x} Branch {count+1}"
            temp_list.append(text)
        branch_dict[x] = temp_list
    
    branch_prop_dict = {}
    
    for city in list(branches.keys()):
        temp_list = []
        for count in range(len(branch_dict[city])):
            temp_list.append(rng.integers(20,100))
        temp_list = [x/sum(temp_list) for x in temp_list]
        branch_prop_dict[city] = temp_list

    return branch_dict, branch_prop_dict



def generate_member_income(df, income_distribution, rng):
    df = df.copy()

    median_map = {
        k: v[0] for k, v in income_distribution.items()
    }

    sigma_map = {
        k: v[1] for k, v in income_distribution.items()
    }

    median = df['AGE_SEGMENT'].map(median_map).astype(float)
    sigma = df['AGE_SEGMENT'].map(sigma_map).astype(float)

    if median.isna().any() or sigma.isna().any():
        raise ValueError("Some AGE_SEGMENT values are missing income parameters.")

    df['ANNUAL_INCOME'] = rng.lognormal(
        mean=np.log(median.to_numpy()),
        sigma=sigma.to_numpy()
    ).round(0)

    return df

def generate_homeownership(
    df,
    homeownership_probability,
    geography_multiplier,
    rng
):
    df = df.copy()

    # Base probability determined by age
    base_probability = df['AGE_SEGMENT'].map(
        homeownership_probability
    ).astype(float)

    # Geographic adjustment
    geography_factor = df['CITY'].map(
        geography_multiplier
    ).astype(float)

    # Income adjustment
    income_factor = np.clip(
        df['ANNUAL_INCOME'] / 80000,
        0.20,
        1.0
    )

    # Combined probability
    adjusted_probability = (
        base_probability
        * income_factor
        / geography_factor**0.25
    ).clip(0, 1)

    # Randomly assign ownership
    df['HOME_OWNER'] = (
        rng.uniform(size=len(df)) < adjusted_probability
    ).astype(int)

    return df

def generate_marital_status(
    df,
    marital_status_distribution,
    rng
):
    df = df.copy()

    statuses = ['Single', 'Married', 'Divorced', 'Widowed']

    df['MARITAL_STATUS'] = None

    for age_segment, probabilities in marital_status_distribution.items():

        mask = df['AGE_SEGMENT'] == age_segment

        df.loc[mask, 'MARITAL_STATUS'] = rng.choice(
            statuses,
            size=mask.sum(),
            p=probabilities
        )

    return df



def generate_member_financials(df, income_distribution, zero_debt_probability,rng):

    df = df.copy()

    # Existing member attributes
    age = df['AGE'].to_numpy()
    income = df['ANNUAL_INCOME'].to_numpy()
    household_income = df['HOUSEHOLD_INCOME'].to_numpy()
    house_value = df['HOUSE_VALUE'].to_numpy()

    # Reference income for each age segment
    median_income = (
        df['AGE_SEGMENT']
        .map(lambda x: income_distribution[x][0])
        .to_numpy()
    )

    # Income relative to members in the same age segment
    income_factor = np.maximum(
        income / np.maximum(median_income, 1),
        0.1
    )

    # --------------------------------------------------
    # ASSETS
    # --------------------------------------------------

    # Wealth accumulation increases with age
    age_factor = np.clip(
        (age - 18) / 45,
        0.1,
        1.5
    )

    # Generate financial assets with substantial variation
    financial_assets = (
        median_income
        * income_factor**0.8
        * age_factor**2
        * rng.lognormal(mean=0, sigma=1.5, size=len(df))
    )

    df['TOTAL_ASSETS'] = (
        house_value + financial_assets
    ).round(0)

    # --------------------------------------------------
    # DEBT
    # --------------------------------------------------

    # Borrowing propensity generally declines with age
    debt_age_factor = np.clip(
        (80 - age) / 50,
        0.1,
        1.2
    )

    # Income influences borrowing capacity
    borrowing_capacity = (
        household_income
        * debt_age_factor
        * rng.lognormal(mean=0.5, sigma=0.7, size=len(df))
    )

    # Homeowners may carry substantial mortgage debt
    housing_debt = (
        house_value
        * np.clip(
            debt_age_factor
            * rng.uniform(0.3, 0.9, len(df)),
            0,
            0.95
        )
    )

    df['TOTAL_DEBT'] = (
        borrowing_capacity + housing_debt
    ).round(0)
    
    # Apply after TOTAL_DEBT has been generated
    probability = df['AGE_SEGMENT'].map(zero_debt_probability).to_numpy()
    
    zero_debt_mask = rng.random(len(df)) < probability
    
    df.loc[zero_debt_mask, 'TOTAL_DEBT'] = 0
    
    # Recalculate net worth
    df['NET_WORTH'] = df['TOTAL_ASSETS'] - df['TOTAL_DEBT']

    return df


def generate_mbr_expenses(
    df,
    rng,
    expenses_dict
):
    start_cols = set(df.columns)
    
    df['RENT_M'] = np.where(
        df['HOME_OWNER']==1,0,
        df['CITY'].map(geography_multiplier)*expenses_dict['baseline_rent']
    )
    df['RENT_M'] = lognormal_distribution_generator(df['RENT_M'],rng=rng)
    
    df['HOME_INSURNACE_A'] = np.where(df['HOME_OWNER']!=1,0,
        lognormal_distribution_generator((df['HOUSE_VALUE']*lognormal_distribution_generator(expenses_dict['home_insurance_perc']))))

    df['PROPERTY_TAX_A'] = np.where(df['HOME_OWNER']!=1,0,
        lognormal_distribution_generator((df['HOUSE_VALUE']*expenses_dict['property_tax_perc'])))

    df['HOME_MAINTAINENCE_M'] = np.where(df['HOME_OWNER']!=1,0,
        lognormal_distribution_generator((df['HOUSE_VALUE']*expenses_dict['home_maint_cost'])))

    
    df['UTILITIES_M'] = np.where(
        df['HOME_OWNER']!=1,lognormal_distribution_generator(np.ones(len(df))*75),lognormal_distribution_generator(df['HOUSE_VALUE']*expenses_dict['utility_base_line_perc']))    

    df['GROCERIES_M'] = lognormal_distribution_generator(df['ANNUAL_INCOME']*expenses_dict['grocery_base_perc'],rng=rng)
    df['PHONE_INTERNET_M'] = lognormal_distribution_generator(np.full(len(df), expenses_dict['phone_internet']),rng=rng)
    df['DINING_OUT_M'] = lognormal_distribution_generator(df['ANNUAL_INCOME']*expenses_dict['dining_out_perc'],rng=rng)
    df['ENTERTAINMENT_M'] = lognormal_distribution_generator(df['ANNUAL_INCOME']*expenses_dict['entertainment_perc'],rng=rng)
    df['GAS_M'] = lognormal_distribution_generator(df['ANNUAL_INCOME']*expenses_dict['gas_perc'],rng=rng)
    df['CAR_INSURANCE_M'] = lognormal_distribution_generator(df['ANNUAL_INCOME']*expenses_dict['car_insurance_maint_perc'],rng=rng)
    df['OTHER_M'] = lognormal_distribution_generator(df['ANNUAL_INCOME']*expenses_dict['other_perc'],rng=rng)

    cols = [x for x in df.columns if x.find('_M')!=-1]
    
    df['TOTAL_NON_LOAN_MONTHLY_EXPENSES'] =df[cols].sum(axis=1)
    
    return df


def generate_kids_count(df, random_state=42):

    rng = np.random.default_rng(random_state)
    df = df.copy()

    def assign_kids(age, marital_status):

        if age < 25:
            probabilities = [0.90, 0.08, 0.02, 0.00]
        elif age < 35:
            probabilities = [0.55, 0.25, 0.17, 0.03]
        elif age < 45:
            probabilities = [0.30, 0.25, 0.35, 0.10]
        elif age < 55:
            probabilities = [0.40, 0.25, 0.28, 0.07]
        elif age < 65:
            probabilities = [0.75, 0.15, 0.08, 0.02]
        else:
            probabilities = [0.95, 0.04, 0.01, 0.00]

        # Adjust likelihood for marital status
        if marital_status == 'Married':
            probabilities[0] *= 0.75

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        return rng.choice([0, 1, 2, 3], p=probabilities)

    df['KIDS_COUNT'] = [
        assign_kids(age, status)
        for age, status in zip(df['AGE'], df['MARITAL_STATUS'])
    ]

    return df


def generate_member_behaviour_profiles(df, random_state=42):

    rng = np.random.default_rng(random_state)

    required = [
        'MEMBERNBR', 'AGE', 'HOUSEHOLD_INCOME',
        'TOTAL_LIQUID_ASSETS', 'TOTAL_DEBT',
        'MONTHLY_CF', 'DURATION_YEARS',
        'BEEM_IS_PRIMARY', 'KIDS_COUNT',
        'GROCERIES_M', 'DINING_OUT_M',
        'ENTERTAINMENT_M', 'OTHER_M'
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df['MEMBERNBR'].duplicated().any():
        raise ValueError("MEMBERNBR must be unique.")

    profile = df[required].copy()
    n = len(profile)

    # -----------------------------------------
    # 1. Existing financial characteristics
    # -----------------------------------------

    income = profile['HOUSEHOLD_INCOME'].clip(lower=1)
    monthly_income = income / 12

    debt_ratio = profile['TOTAL_DEBT'] / income

    liquidity_months = (
        profile['TOTAL_LIQUID_ASSETS'].clip(lower=0)
        / monthly_income
    )

    cash_flow_ratio = (
        profile['MONTHLY_CF'] / monthly_income
    )

    discretionary_ratio = (
        profile['DINING_OUT_M']
        + profile['ENTERTAINMENT_M']
        + 0.5 * profile['OTHER_M']
    ) / monthly_income

    kids = profile['KIDS_COUNT'].clip(lower=0, upper=4)
    age = profile['AGE']
    tenure = profile['DURATION_YEARS'].clip(lower=0)

    primary = profile['BEEM_IS_PRIMARY'].clip(0, 1)

    # Normalize selected characteristics
    debt_score = np.clip(debt_ratio / 5, 0, 1)
    liquidity_score = np.clip(liquidity_months / 12, 0, 1)
    cash_flow_score = np.clip(cash_flow_ratio, -0.5, 0.5)
    spending_score = np.clip(discretionary_ratio / 0.4, 0, 1)
    tenure_score = np.clip(tenure / 20, 0, 1)

    # Shared individual variation
    # Creates some correlation between behaviours
    financial_discipline = rng.normal(0, 1, n)
    risk_preference = rng.normal(0, 1, n)

    # -----------------------------------------
    # 2. Income growth and volatility
    # -----------------------------------------

    # Annual expected income growth
    growth_base = np.select(
        [
            age < 30,
            age < 45,
            age < 60
        ],
        [
            0.035,
            0.025,
            0.015
        ],
        default=0.005
    )

    profile['INCOME_GROWTH_RATE'] = np.clip(
        growth_base
        + rng.normal(0, 0.015, n),
        -0.05, 0.08
    )

    # Monthly income volatility
    profile['INCOME_VOLATILITY'] = np.clip(
        0.025
        + 0.015 * (age < 35)
        + 0.01 * (kids > 0)
        + rng.normal(0, 0.012, n),
        0.005, 0.15
    )

    # -----------------------------------------
    # 3. Spending growth and volatility
    # -----------------------------------------

    profile['SPENDING_GROWTH_RATE'] = np.clip(
        0.025
        + 0.025 * spending_score
        + 0.004 * kids
        + 0.008 * risk_preference
        - 0.008 * financial_discipline
        + rng.normal(0, 0.012, n),
        -0.04, 0.12
    )

    profile['SPENDING_VOLATILITY'] = np.clip(
        0.035
        + 0.045 * spending_score
        + 0.006 * kids
        + 0.01 * risk_preference
        + rng.normal(0, 0.015, n),
        0.01, 0.20
    )

    # -----------------------------------------
    # 4. Savings and credit behaviour
    # -----------------------------------------

    # Score between 0 and 1
    profile['SAVINGS_PROPENSITY'] = np.clip(
        0.45
        + 0.20 * cash_flow_score
        + 0.15 * liquidity_score
        - 0.12 * spending_score
        + 0.08 * financial_discipline
        + rng.normal(0, 0.08, n),
        0, 1
    )

    # Score between 0 and 1
    profile['CREDIT_RELIANCE'] = np.clip(
        0.25
        + 0.35 * debt_score
        - 0.15 * liquidity_score
        - 0.15 * cash_flow_score
        + 0.08 * risk_preference
        + rng.normal(0, 0.08, n),
        0, 1
    )

    # -----------------------------------------
    # 5. Financial resilience
    # -----------------------------------------

    # Score between 0 and 1
    profile['FINANCIAL_RESILIENCE'] = np.clip(
        0.30
        + 0.40 * liquidity_score
        + 0.20 * cash_flow_score
        - 0.15 * debt_score
        + rng.normal(0, 0.07, n),
        0, 1
    )

    # -----------------------------------------
    # 6. Relationship stability
    # -----------------------------------------

    # Score between 0 and 1
    profile['RELATIONSHIP_STABILITY'] = np.clip(
        0.30
        + 0.25 * tenure_score
        + 0.25 * primary
        + rng.normal(0, 0.12, n),
        0, 1
    )

    # -----------------------------------------
    # 7. Return behavioural attributes only
    # -----------------------------------------

    behaviour_columns = [
        'MEMBERNBR',
        'INCOME_GROWTH_RATE',
        'INCOME_VOLATILITY',
        'SPENDING_GROWTH_RATE',
        'SPENDING_VOLATILITY',
        'SAVINGS_PROPENSITY',
        'CREDIT_RELIANCE',
        'FINANCIAL_RESILIENCE',
        'RELATIONSHIP_STABILITY'
    ]

    return profile[behaviour_columns]


def create_mbr_base_df(
    geography_dict,   
    age_distribution,
    branches,
    duration_dict,
    entity_map,
    income_distribution,
    geography_multiplier,
    seed=42,
    members=180000
):
    df = pd.DataFrame(
        [f'Member {x}' for x in range(1,members+1)],columns=['MEMBERNBR']
    )
    # Randomly Assign Age With Distribution as Defined


    rng = np.random.default_rng(seed)
    
    df['AGE']= (
        np.clip(rng.normal(
            age_distribution['mean_age'], 
            age_distribution['std_dev'], 
            members),
                age_distribution['min_age'],
                age_distribution['max_age']
               ).round().astype(int)
    )

    # Randomly Assign City Based on Distribution
    
    df['CITY'] = rng.choice(
        list(geography_dict.keys()),
        p=list(geography_dict.values()),
        size=members
)
    # Create  a Branch Distribution
    branch_dict, branch_prop_dict = generate_branch_distribution(branches,rng)

    df['BRANCH'] = df['CITY'].map(
        lambda city: rng.choice(
            branch_dict[city],
            p=branch_prop_dict[city])
    )

    branch_list = df['BRANCH'].unique()

    reassign_mask = rng.uniform(size=members) < 0.025
    
    df.loc[reassign_mask, 'BRANCH'] = rng.choice(
        branch_list,
        size=reassign_mask.sum()
    )

    df['DURATION_YEARS'] = (
      np.clip(rng.normal(
          duration_dict['mean'],
          duration_dict['std_dev'],
          members),
              min=0)
    )
    
    df['DURATION_YEARS'] = np.minimum(
        df['DURATION_YEARS'],
        np.maximum(df['AGE'] - 12, 0)
    )

    df['LEGACY_CU'] = df['BRANCH'].map(entity_map)
    
    create_segment_range(
        df,
        column_name='AGE',
        segment_range=[20,25,35,45,55,65,85]
    )

    create_segment_range(
        df,
        column_name='DURATION_YEARS',
        segment_range=[0,1,3,5,10,15,20,25]
    )

    df = generate_member_income(df,income_distribution,rng)
    df = generate_marital_status(df,marital_status_distribution,rng)

    df['HOUSEHOLD_INCOME'] = df['ANNUAL_INCOME']*df['MARITAL_STATUS'].map(marital_stats_multiplier)


    random_house_price = np.clip(
            rng.normal( 
                housing_price['mean'],
                housing_price['std_dev'], 
                members
            ),
            min=housing_price['min']).round().astype(int)
    
    df = generate_homeownership(df,homeownership_probability,geography_multiplier,rng)
    
    df['HOUSE_VALUE'] = np.where(df['HOME_OWNER']==1,df['CITY'].map(geography_multiplier) * random_house_price,0)

    df = generate_member_financials(df,income_distribution,zero_debt_probability,rng)

    df['TOTAL_LIQUID_ASSETS'] = np.clip(df['TOTAL_ASSETS'] - df['HOUSE_VALUE'],min=0)

    ######
    df['STOCK_INVESTMENT_PERC'] = lognormal_distribution_generator(df['AGE_SEGMENT'].map(stock_vs_cash))
    df['EXCESS_EQUITY_IN_REAL_ESTATE'] = (np.clip(df['HOUSE_VALUE'] - df['TOTAL_DEBT'],min=0))
    df['LIQUID_TNW'] = (np.clip(df['NET_WORTH'] - df['EXCESS_EQUITY_IN_REAL_ESTATE'],min=0))
    
    df['TOTAL_CASH_EQUIVALENTS'] = (np.clip(df['LIQUID_TNW']*(1-df['STOCK_INVESTMENT_PERC']),min=0))
    df['TOTAL_PORTFOLIO'] = df['LIQUID_TNW']*df['STOCK_INVESTMENT_PERC']

    df['HAS_VEHICLE'] = car_ownership(df,car_ownership_probability,rng)

    df = distribute_member_debt(df,market_interest_rates,rng)

    
    df['BEEM_IS_PRIMARY'] = (rng.uniform(size=len(df))<lognormal_distribution_generator(.65,rng=rng)).astype(int)
    df['TOTAL_DEPOSIT'] = df['BEEM_IS_PRIMARY']*df['TOTAL_CASH_EQUIVALENTS']*rng.uniform(low=.85,high=1,size=len(df))
    df['TOTAL_LOANS'] =   (
        np.random.choice(
        [0,.25,.5,.75,.8,.85,.9,.95,1],
            p=[.1,.1,.2,.1,.1,.1,.1,.1,.1]
        )*(df['MORTGAGE_DEBT']+df['LOC_DEBT']+df['PERSONAL_LOAN_DEBT'])
                          )
        
    df = generate_mbr_expenses(df,rng=rng,expenses_dict=expenses_dict)
        
    df['MONTHLY_CF_AFTER_DEBT'] = (df['HOUSEHOLD_INCOME']*.7)/12 - df['MONTHLY_DEBT_PAYMENT'] # HOUSEHOLD. Less Tax.
    df['MONTHLY_CF'] = df['MONTHLY_CF_AFTER_DEBT'] - df['TOTAL_NON_LOAN_MONTHLY_EXPENSES']

    df = calculate_member_profile(df)
    df = generate_kids_count(df,seed) 
    mbr_profile = generate_member_behaviour_profiles(df)


    activity_cols = [
        'MEMBERNBR','RENT_M', 'HOME_INSURNACE_A', 'PROPERTY_TAX_A','HOME_MAINTAINENCE_M', 'UTILITIES_M', 'GROCERIES_M', 'PHONE_INTERNET_M','DINING_OUT_M', 
        'ENTERTAINMENT_M', 'GAS_M','CAR_INSURANCE_M','OTHER_M', 'TOTAL_NON_LOAN_MONTHLY_EXPENSES', 'MONTHLY_CF_AFTER_DEBT','MONTHLY_CF','MONTHLY_DEBT_PAYMENT'
          ]
    
    activity_df = df[activity_cols]
    member_df = df.drop([x for x in activity_cols if x not in ['MEMBERNBR']],axis=1)
                                                                 
    return member_df,activity_df,mbr_profile

member_df,activity_df,mbr_profile = create_mbr_base_df(
    age_distribution=age_dict,
    geography_dict=geography_dict,
    branches=branches,
    duration_dict=duration_dict,
    entity_map=entity_map,
    income_distribution=income_distribution,
    geography_multiplier=geography_multiplier    
)


