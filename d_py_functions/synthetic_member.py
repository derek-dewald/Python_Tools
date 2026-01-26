from dateutil.relativedelta  import relativedelta
import numpy as np
import pandas as pd
import random
import datetime


try:
    rng = np.random.default_rng(seed)
except:
    rng = np.random.default_rng()

monthly_payment_dict = {
    'TV_INTERNET':[.05,.15],
    'CELL_PHONE':[.05,.15],
    'INSURANCE_HEALTH':[.05,.15],
    'INSURANCE_HOUSE':[.05,.11],
    'PROPERTY_TAX_PROV':[.05,.11],
    'PROPERTY_TAX_MUNI':[.05,.11],
    'UTILITIES_GAS':[.05,.11],
    'UTILITIES_HYDRO':[.05,.11],
}

pos_txn_dict = {
    'GROCERY':[0,.25],
    'GAS_TRANSPORTATION':[.05,.15],
    'FAST_FOOD':[0,.1],
    'OTHER_FOOD':[0,.1],
    'CASH':[0,.2],
    'ENTERTAINMENT_OTHER':[0,.1],
    'CLOTHES':[0,.1],
}

mbr_profile_dict = {
    "EARLY_CAREER_PROFESSIONAL": {
        "age": [22, 30],
        "liquid_assets": [5000, 40000],
        "weight": 0.05,
        "primary_is_beem": [0.25, 0.4],
        'kids':[0,.25],
        "home_owner": [0.10, 0.25],
        "annual_income": [55000, 90000],
        "duration": [0, 5],
        "consumer_debt": [0, 30000],
        "mortgage_debt": [250000, 750000],
        "bill_pay_debit": [0.10, 0.35],
        "pos_txn_debit": [0.30, 0.45],
        "investment_debit": [0.05, 0.12],
        "payroll_deposit": [0.80, 0.95],
        "other_deposit": [0.00, 0.10],
    },

    "MID_CAREER_PROFESSIONAL": {
        "age": [30, 50],
        "liquid_assets": [50000, 400000],
        "weight": 0.05,
        "primary_is_beem": [0.30, 0.5],
        'kids':[0,.5],
        "home_owner": [0.50, 0.7],
        "annual_income": [80000, 160000],
        "duration": [0, 7],
        "consumer_debt": [0, 15000],
        "mortgage_debt": [250000, 1500000],
        "bill_pay_debit": [0.15, 0.40],
        "pos_txn_debit": [0.25, 0.40],
        "investment_debit": [0.10, 0.20],
        "payroll_deposit": [0.85, 0.98],
        "other_deposit": [0.00, 0.10],
    },

    "LATE_CAREER_PROFESSIONAL": {
        "age": [45, 70],
        "liquid_assets": [250000, 2000000],
        "weight": 0.02,
        "primary_is_beem": [0.20, 0.6],
        'kids':[0,1],
        "home_owner": [0.70, 0.9],
        "annual_income": [110000, 250000],
        "duration": [0, 15],
        "consumer_debt": [0, 30000],
        "mortgage_debt": [100000, 750000],
        "bill_pay_debit": [0.10, 0.25],
        "pos_txn_debit": [0.20, 0.35],
        "investment_debit": [0.15, 0.30],
        "payroll_deposit": [0.80, 0.95],
        "other_deposit": [0.05, 0.20],
    },

    "FIXED_INCOME_SENIOR": {
        "age": [65, 90],
        "liquid_assets": [5000, 80000],
        "weight": 0.20,
        "primary_is_beem": [0.40, 0.7],
        'kids':[0,1],
        "home_owner": [0.60, 1.0],
        "annual_income": [40000, 60000],
        "duration": [0, 20],
        "consumer_debt": [0, 30000],
        "mortgage_debt": [100000, 500000],
        "bill_pay_debit": [0.10, 0.25],
        "pos_txn_debit": [0.25, 0.40],
        "investment_debit": [0.05, 0.12],
        "payroll_deposit": [0.00, 0.10],  # payroll uncommon
        "other_deposit": [0.60, 0.90],   # pensions, transfers
    },

    "LOW_INCOME_WORKER": {
        "age": [25, 60],
        "liquid_assets": [0, 15000],
        "weight": 0.15,
        "primary_is_beem": [0.10, 0.3],
        'kids':[0,1],
        "home_owner": [0.10, 0.35],
        "annual_income": [40000, 60000],
        "duration": [0, 10],
        "consumer_debt": [0, 50000],
        "mortgage_debt": [100000, 500000],
        "bill_pay_debit": [0.10, 0.20],
        "pos_txn_debit": [0.35, 0.55],
        "investment_debit": [0.02, 0.07],
        "payroll_deposit": [0.80, 0.95],
        "other_deposit": [0.00, 0.10]
    },

    "RECENT_GRADUATE": {
        "age": [21, 27],
        "liquid_assets": [1000, 15000],
        "weight": 0.05,
        "primary_is_beem": [0.10, 0.4],
        'kids':[0,0],
        "home_owner": [0.00, 0.15],
        "annual_income": [40000, 65000],
        "duration": [0, 5],
        "consumer_debt": [0, 50000],
        "mortgage_debt": [250000, 500000],
        "bill_pay_debit": [0.20, 0.40],
        "pos_txn_debit": [0.35, 0.55],
        "investment_debit": [0.03, 0.10],
        "payroll_deposit": [0.85, 0.98],
        "other_deposit": [0.00, 0.08]
    },

    "STUDENT": {
        "age": [18, 25],
        "liquid_assets": [0, 5000],
        "weight": 0.05,
        "primary_is_beem": [0.20, 0.5],
        'kids':[0,0],
        "home_owner": [0.00, 0.10],
        "annual_income": [10000, 25000],
        "duration": [0, 5],
        "consumer_debt": [0, 50000],
        "mortgage_debt": [250000, 500000],
        "bill_pay_debit": [0.10, 0.30],
        "pos_txn_debit": [0.40, 0.60],
        "investment_debit": [0.00, 0.05],
        "payroll_deposit": [0.10, 0.40],  
        "other_deposit": [0.20, 0.60],
    },

    "RETIREMENT_READY": {
        "age": [55, 80],
        "liquid_assets": [400000, 1500000],
        "weight": 0.15,
        "primary_is_beem": [0.50, 0.7],
        'kids':[0,1],
        "home_owner": [0.50, 1.0],
        "annual_income": [70000, 150000],
        "duration": [0, 20],
        "consumer_debt": [0, 30000],
        "mortgage_debt": [250000, 750000],
        "bill_pay_debit": [0.10, 0.25],
        "pos_txn_debit": [0.20, 0.35],
        "investment_debit": [0.15, 0.30],
        "payroll_deposit": [0.60, 0.85],  
        "other_deposit": [0.10, 0.30],
    },

    "PAYCHECK_TO_PAYCHECK": {
        "age": [22, 65],
        "liquid_assets": [0, 5000],
        "weight": 0.16,
        "primary_is_beem": [0.30, 0.5],
        'kids':[0,1],
        "home_owner": [0.20, 0.5],
        "annual_income": [40000, 75000],
        "duration": [0, 10],
        "consumer_debt": [0, 50000],
        "mortgage_debt": [100000, 500000],
        "bill_pay_debit": [0.10, 0.25],
        "pos_txn_debit": [0.40, 0.60],
        "investment_debit": [0.00, 0.05],
        "payroll_deposit": [0.85, 0.98],
        "other_deposit": [0.00, 0.10]
    },

    "FINANCIALLY_STRESSED": {
        "age": [25, 65],
        "liquid_assets": [0, 2000],
        "weight": 0.12,
        "primary_is_beem": [0.30, 0.7],
        'kids':[0,1],
        "home_owner": [0.30, 0.6],
        "annual_income": [50000, 80000],
        "duration": [0, 7],
        "consumer_debt": [0, 50000],
        "mortgage_debt": [250000, 750000],
        "bill_pay_debit": [0.10, 0.25],
        "pos_txn_debit": [0.45, 0.65],
        "investment_debit": [0.00, 0.05],
        "payroll_deposit": [0.80, 0.95],
        "other_deposit": [0.00, 0.10]
    },
}

branch_mbr_composition_dict_1 = {
    'EARLY_CAREER_PROFESSIONAL': 0.15,
    'MID_CAREER_PROFESSIONAL': 0.08,
    'LATE_CAREER_PROFESSIONAL': 0.05,
    'FIXED_INCOME_SENIOR': 0.25,
    'LOW_INCOME_WORKER': 0.05,
    'RECENT_GRADUATE': 0.05,
    'STUDENT': 0.05,
    'RETIREMENT_READY': 0.25,
    'PAYCHECK_TO_PAYCHECK': 0.05,
    'FINANCIALLY_STRESSED': 0.02
}

branch_mbr_composition_dict_2 = {
    'EARLY_CAREER_PROFESSIONAL': 0.10,
    'MID_CAREER_PROFESSIONAL': 0.05,
    'LATE_CAREER_PROFESSIONAL': 0.05,
    'FIXED_INCOME_SENIOR': 0.3,
    'LOW_INCOME_WORKER': 0.1,
    'RECENT_GRADUATE': 0.01,
    'STUDENT': 0.02,
    'RETIREMENT_READY': 0.35,
    'PAYCHECK_TO_PAYCHECK': 0.01,
    'FINANCIALLY_STRESSED': 0.01
}

branch_mbr_composition_dict_3 = {
    'EARLY_CAREER_PROFESSIONAL': 0.05,
    'MID_CAREER_PROFESSIONAL': 0.1,
    'LATE_CAREER_PROFESSIONAL': 0.1,
    'FIXED_INCOME_SENIOR': 0.1,
    'LOW_INCOME_WORKER': 0.15,
    'RECENT_GRADUATE': 0.1,
    'STUDENT': 0.1,
    'RETIREMENT_READY': 0.05,
    'PAYCHECK_TO_PAYCHECK': 0.1,
    'FINANCIALLY_STRESSED': 0.15
}

branch_mbr_composition_dict_4 = {
    'EARLY_CAREER_PROFESSIONAL': 0,
    'MID_CAREER_PROFESSIONAL': 0,
    'LATE_CAREER_PROFESSIONAL': 0,
    'FIXED_INCOME_SENIOR': 0.35,
    'LOW_INCOME_WORKER': 0.1,
    'RECENT_GRADUATE': 0,
    'STUDENT': 0,
    'RETIREMENT_READY': 0.35,
    'PAYCHECK_TO_PAYCHECK': 0.15,
    'FINANCIALLY_STRESSED': 0.05
}

branch_mbr_composition_dict_5 = {
    'EARLY_CAREER_PROFESSIONAL': 0.2,
    'MID_CAREER_PROFESSIONAL': 0.15,
    'LATE_CAREER_PROFESSIONAL': 0.1,
    'FIXED_INCOME_SENIOR': 0.1,
    'LOW_INCOME_WORKER': 0.1,
    'RECENT_GRADUATE': 0.1,
    'STUDENT': 0.05,
    'RETIREMENT_READY': 0.1,
    'PAYCHECK_TO_PAYCHECK': 0.05,
    'FINANCIALLY_STRESSED': 0.05
}

branch_mbr_composition_dict = {
    'dict_1':{'profile':branch_mbr_composition_dict_1,'perc_':.35},
    'dict_2':{'profile':branch_mbr_composition_dict_2,'perc_':.35},
    'dict_3':{'profile':branch_mbr_composition_dict_3,'perc_':.05},
    'dict_4':{'profile':branch_mbr_composition_dict_4,'perc_':.2},
    'dict_5':{'profile':branch_mbr_composition_dict_5,'perc_':.05}
}

general_assumptions = {
    'OPERATIONAL_ATTRITION':{
        'Description':"Attrition which occurs as the result of Daily Operation. Evenly distributed throughout the Business. Monthly Percentage.",
        'value':[.00025,.0025],
        'model_status':1
    },
    'MEMBER_HEALTH':{
        'Description':"Attrition which occurs as the result of inherentence and unfortunate Member Passing. Only applies to members over 70. Monthly Percentage.",
        'value':[.001,.005],
        'model_status':1
    },
    'INTEGRATION_IMPACT':{
        'Description':"Attrition which occurs as the result of frustration, negative impact or general disagreement with merger",
        'value':[.0001,.0005],
    'model_status':0
    },
    'PERFORMANCE_RELATED':{
        'Description':"Attrition which occurs as the result of impact of a specific event, can add as many event based, just need to understand",
        'value':[.0001,.0005],
    'model_status':0},
    'NUMBER_OF_KIDS':{
        'Description': "Allocation as to the Number of Kids a Member has based on the Random Distribution. Used a explict parameter, not generally distributed parameter",
        'value':{0:.4,1:.8,2:.9,3:1},
        'model_status':0 
    },
    'MTG_Multiplier':{
        'Description': "Multiplier to apply to Random Mortgage Valuation based on Geography and Mortgage Pricing.",
        'value':{'LEGACY 1':1.25,'LEGACY 2':1.25,'LEGACY 3':.75,'LEGACY 4':.6},
        'model_status':0 
    },
    'Legacy_efficieny_factor_dict':{
        'Description': "Random Value Created to Implement slightly Less Randomness across distribution and apply ability to note different growth and performance across legacy institution",
        'value':{'LEGACY 1':.98,'LEGACY 2':.97,'LEGACY 3':1.03,'LEGACY 4':1.08},
        'model_status':0 
    },
    'NUMBER_OF_MEMBERS':{
        'Description': "Number of members to be included in Dataframe",
        'value':[140000,200000],
        'model_status':0 
    },
    
}

legacy1 = {
    'Vancouver': 0.15,'Burnaby': 0.05,'Richmond': 0.05,'Surrey': 0.25,'Langley': 0.05,'New Westminster': 0.05,
    'North Vancouver': 0.05,'West Vancouver': 0.05,'Port Coquitlam': 0.05,'Abbotsford': 0.15,'Chilliwack': 0.05,
    'Grand Forks': 0.05
}
legacy2 = {
    "Vancouver":.15, "Burnaby":.1,"North Vancouver":.5, "West Vancouver":.25
}
legacy3 = {
    "Kelowna":.4, "West Kelowna":.2, "Vernon":.2, "Penticton":.1, "Salmon Arm":.025,"Nelson":.025,"Cranbrook":.05
}
legacy4 = {
    "Prince George":.5, "Fort St. John":.25, "Dawson Creek":.25
}

legacy_city_dict = {
    'LEGACY 1':legacy1,
    'LEGACY 2':legacy2,
    'LEGACY 3':legacy3,
    'LEGACY 4':legacy4
}



mbr_growth_attrition_profiles={
    'DEPOSIT_GROWTH_RATE_MAJOR':{
        'Description': "Increase of Member Deposit Growth Rate and Income. Reflect of member with growing prospects",
        'distribution':[0,.005],
        'deposit_growth':[1.05,1.20],
        'loan_growth':[1,1],
        'income_growth':[1.05,1.20],
        'txn_growth':[1,1],
        'deposit_impact':[0,0],
        'loan_impact':[0,0],
        'income_impact':[0,0],
        'txn_impact':[0,0],
        'quarters_remaining':4
    },
    'DEPOSIT_GROWTH_RATE_MINOR':{
        'Description': "Growth Profile to be Applied to Members, indicating they will experience a 1 time deposit increase",
        'distribution':[0,.005],
        'deposit_growth':[1.025,1.10],
        'loan_growth':[1,1],
        'income_growth':[1.025,1.10],
        'txn_growth':[1,1],
        'deposit_impact':[0,0],
        'loan_impact':[0,0],
        'income_impact':[0,0],
        'txn_impact':[0,0],
        'quarters_remaining':12
    },
    'MATERIAL_LOAN':{
        'Description': "Growth Profile to be Applied to Members, indicating they will experience 1 time loan increase. Increase Income to Support.",
        'distribution':[0,.005],
        'deposit_growth':[1,1],
        'loan_growth':[1,1],
        'income_growth':[1,1],
        'txn_growth':[1,1],
        'deposit_impact':[0,0],
        'loan_impact':[250000,1000000],
        'income_impact':[25000,75000],
        'txn_impact':[0,0],
        'quarters_remaining':1
    },
    'STEADY_TRANSACTION_INCREASE':{
        'Description': "Growth Profile to be Applied to Members, indicating they will experience a Increase to Transaction Value (and perhaps Volume)",
        'distribution':[0,.005],
        'deposit_growth':[1,1],
        'loan_growth':[1,1],
        'income_growth':[1.01,1.03],
        'txn_growth':[1.025,1.075],
        'deposit_impact':[0,0],
        'loan_impact':[0,0],
        'income_impact':[0,75000],
        'txn_impact':[0,0],
        'quarters_remaining':18
    },
    'SPARKED_ENGAGEMENT':{
        'Description': "Growth Profile to be Applied to Members, indicating they will become Beem Primary Members and Get a New Deposit and Loan Balance and Transaction Activity",
        'distribution':[0,.005],
        'deposit_growth':[1.025,1.075],
        'loan_growth':[1,1],
        'income_growth':[1.02,1.04],
        'txn_growth':[1.025,1.075],
        'deposit_impact':[10000,50000],
        'loan_impact':[0,0],
        'income_impact':[0,25000],
        'txn_impact':[250,1000],
        'quarters_remaining':1
    },
    'SLOW_DISENGAGEMENT':{
        'Description': "Attrition Profile to be Applied to Members, indicating they will become Less steadily across all Metrics",
        'distribution':[0,.005],
        'deposit_growth':[.975,1],
        'loan_growth':[1,1],
        'income_growth':[1,1],
        'txn_growth':[.9,1],
        'deposit_impact':[0,0],
        'loan_impact':[0,0],
        'income_impact':[0,0],
        'txn_impact':[0,0],
        'quarters_remaining':30
    },
    'RAPID_DISENGAGEMENT':{
        'Description': "Attrition Profile to be Applied to Members, indicating they will become Less steadily across all Metrics",
        'distribution':[0,.005],
        'deposit_growth':[.9,1],
        'loan_growth':[1,1],
        'income_growth':[1,1],
        'txn_growth':[.85,.95],
        'deposit_impact':[0,0],
        'loan_impact':[0,0],
        'income_impact':[0,0],
        'txn_impact':[0,0],
        'quarters_remaining':24
    },
    'IMMEDIATE_DISENGAGEMENT':{
        'Description': "Attrition Profile to be Applied to Members, indicating they will cease being a member with Immediate Effect",
        'distribution':[0,.005],
        'deposit_growth':[0,.75],
        'loan_growth':[0,.5],
        'income_growth':[1,1],
        'txn_growth':[.5,.75],
        'deposit_impact':[0,0],
        'loan_impact':[0,0],
        'income_impact':[0,0],
        'txn_impact':[0,0],
        'quarters_remaining':24
    },
    'PARTIAL_DISENGAGEMENT_LOAN':{
        'Description': "Attrition Profile to be Applied to Members, indicating they will cease being a product holder of Loans",
        'distribution':[0,.005],
        'deposit_growth':[.85,1],
        'loan_growth':[0,0],
        'income_growth':[1,1],
        'txn_growth':[.85,.95],
        'deposit_impact':[0,0],
        'loan_impact':[0,0],
        'income_impact':[0,0],
        'txn_impact':[0,0],
        'quarters_remaining':1
    },
    'PARTIAL_DISENGAGEMENT_DEPOSIT':{
        'Description': "Attrition Profile to be Applied to Members, indicating they will cease being a product holder of Deposits",
        'distribution':[0,.005],
        'deposit_growth':[.5,.75],
        'loan_growth':[1,1],
        'income_growth':[1,1],
        'txn_growth':[.85,.95],
        'deposit_impact':[0,0],
        'loan_impact':[0,0],
        'income_impact':[0,0],
        'txn_impact':[0,0],
        'quarters_remaining':24
    }
} 

# Add a Default which takes residual Distribution Value.
mbr_growth_attrition_profiles['NO_CHANGE'] = {
    'Description': "Default Profile, which has member making only Randomly Generated Change",
    'distribution':[sum([mbr_growth_attrition_profiles[x]['distribution'][0] for x in mbr_growth_attrition_profiles.keys()]),
                    1-sum([mbr_growth_attrition_profiles[x]['distribution'][1] for x in mbr_growth_attrition_profiles.keys()])],
    'deposit_growth':[1,1],
    'loan_growth':[1,1],
    'income_growth':[1,1],
    'txn_growth':[1,1],
    'deposit_impact':[0,0],
    'loan_impact':[0,0],
    'income_impact':[0,0],
    'txn_impact':[0,0],
    'quarters_remaining':1
}


# List of Potential Base line Branch Composition of Members 
branch_mbr_composition_templates = list(branch_mbr_composition_dict.keys())

# Default "Suggested" distribution of Potential Base Line Branch Compositions
branch_mbr_composition_templates_perc = [branch_mbr_composition_dict[x]['perc_'] for x in branch_mbr_composition_templates]

# Total Number of Members, Randomly Selected
total_mbrs = random.randint(general_assumptions['NUMBER_OF_MEMBERS']['value'][0],general_assumptions['NUMBER_OF_MEMBERS']['value'][1])


def pick_from_dict(dict_,text):  
    dict_record = dict_.get(text)
    list_ = list(dict_record.keys())
    perc_ = list(dict_record.values())
    return random.choices(list_,weights=perc_)[0]

def calculate_rng_from_df_low_high(df, 
                                   new_column_name,
                                   low, 
                                   high,
                                   decimal=2,
                                   distribution='continuous',
                                   skew=1):
    """
    Generate a column of random values between df[low] and df[high].
    - distribution: 'continuous' or 'discrete'
    - skew: 1 = uniform (no skew), >1 bias low, <1 bias high
    - seed: optional for reproducibility
    """
    
    low_arr = df[low].to_numpy()
    high_arr = df[high].to_numpy()

    # Draw uniform [0,1]
    u = rng.uniform(0.0, 1.0, size=len(df))

    # Apply skew if not 1
    if skew != 1:
        u = u ** skew

    if distribution == 'continuous':
        vals = low_arr + (high_arr - low_arr) * u
        df[new_column_name] = np.round(vals, decimal)

    elif distribution == 'discrete':
        vals = low_arr + (high_arr - low_arr) * u
        ints = np.floor(vals).astype(int)
        # Clip to bounds (exclusive upper)
        ints = np.clip(ints, low_arr, high_arr - 1)
        df[new_column_name] = ints

    else:
        raise ValueError("distribution must be 'continuous' or 'discrete'")

    return df


def simplistic_engagement_calculation(df,new_column_name='ENGAGEMENT_SCORE'):
    
    '''
    
    Is Absolute Value an indication of Rate Shopping.
    Can we look at certain types of Activity - Refinances, New TDs with Nothing Else. Etc.
    Are Almost Loyal Members more/less/equally suseptible to engagement.
    What other types of metrics could be included, contact, complaints, referrals.
    # Update duration with Primary is Beem. 
    # Can we look for members which have flags, first mortgage, student funded, etc...
    # Does Legacy Entity Impact Loyalty.
    
    
    '''
    
    engagement_dict = {
        'PRIMARY_IS_BEEM':{
            'Definition':"If the member has their primary banking at Beem, then it demonstrates the minmimum required for Loyalty",
            'Weight':.20,
            'CALCULATION':['PRIMARY_IS_BEEM','EQ',1]},
        'DURATION':{
            'Definition':"The longer a member has been with Beem, the more loyal they are. Not that this metric should be combined with Primary Attribute, as simply having an account does not demonstrate loyalty",
            'Weight':.20,
            'CALCULATION':['DURATION','GT',7]},
        'MORTGAGE_BALANCE':{
            'Definition':"Scored related to relevant balance, with consideration to relative amount known to be held. Before Including Absolute value, need to imperically test. Notion is High Value, Low Perc are rate sensitive",
            'Weight':.15,
            'CALCULATION':['PERC_MORTGAGE','GT',.8]},
        'DEPOSIT_BALANCE':{
            'Definition':"Scored related to relevant balance, with consideration to relative amount known to be held. Before Including Absolute value, need to imperically test. Notion is High Value, Low Perc are rate sensitive",
            'Weight':.15,
            'CALCULATION':['PERC_MORTGAGE','GT',.8]},
        
        ################################
        # THIS IS WRONG - PROXY
        ################################
        
        
        'TXN_VALUE':{
            'Definition':"Scored related to relevant balance, with consideration to relative amount known to be held. Before Including Absolute value, need to imperically test. Notion is High Value, Low Perc are rate sensitive",
            'Weight':.15,
            'CALCULATION':['PERC_SPENDING_KNOWN','GT',.7]},
        'TXN_VOLUME':{
            'Definition':"Scored related to relevant balance, with consideration to relative amount known to be held. Before Including Absolute value, need to imperically test. Notion is High Value, Low Perc are rate sensitive",
            'Weight':.15,
            'CALCULATION':['PERC_SPENDING_KNOWN','GT',.7]},
    } 
    
    df[new_column_name] = 0
    
    for metric in list(engagement_dict.keys()):
        column = engagement_dict[metric]['CALCULATION'][0]
        calculation = engagement_dict[metric]['CALCULATION'][1]
        threshold = engagement_dict[metric]['CALCULATION'][2]
        score = engagement_dict[metric]['Weight']
 
    
        if calculation=='GT':           
            df[new_column_name] = np.where(df[column]>threshold,score,0) + df[new_column_name]
        elif calculation=='EQ':
            df[new_column_name] = np.where(df[column]==threshold,score,0) + df[new_column_name]
        elif calculation=='LT':
            df[new_column_name] = np.where(df[column]<threshold,score,0) + df[new_column_name]
    
    
    
def create_column_from_dict_distribution(df,
                                         column_name,
                                         new_column_name,
                                         cdf_dict,
                                         calculation='lt'):
    '''
    
    
    '''
    
    # Dict will not always hold Numerical Order.
    
    
    items = sorted(cdf_dict.items(), key=lambda kv: kv[1])

    conditions = []
    values = []
    
    if calculation=='lt':
        for value in items:
            conditions.append(df[column_name]<=value[1])
            values.append(value[0])
            
    elif calculation=='eq':
        for value in items:
            conditions.append(df[column_name]>=value[1])
            values.append(value[0])
            
            
    df[new_column_name] = np.select(conditions,values,default=np.nan)


def calculate_distribution_from_dictlist(dict_,decimals=4):
    
    value_dict = {}
    
    for key,value in dict_.items():
        value_dict[key] = np.round(np.random.uniform(value[0],value[1]),decimals)
    
    return value_dict
        



def create_random_value_from_dict(df,dict_,dict_lvl=2,decimals=4):
    
    list_ = list(dict_.keys())
    
    df = df.copy()
    
    if dict_lvl==2:
        new_df = flatten_clean_dict({'test':dict_},high_low_list_fix=True)
        new_df.drop('INDEX',axis=1,inplace=True)
        
    new_df = replicate_df_row(new_df,len(df))
            
    for column in list_:
        calculate_rng_from_df_low_high(new_df,column.upper(),f'{column}_low',f'{column}_high',decimals)
    
    return new_df[list_]
    
def decouple_txn(df, 
                 reference_value,
                 txn_dict,
                 primary_key='MEMBERNBR',
                 exclude_non_ho=[]):
    '''
    
    '''

    temp_df = create_random_value_from_dict(df,txn_dict)
    
    for value in exclude_non_ho:
        temp_df[value] = temp_df[value]*df['HOME_OWNER']
          
    temp_df['OTHER'] = (1 - temp_df.sum(axis=1)).clip(lower=0)
    
    temp_df = pd.concat([df[[primary_key,reference_value]],temp_df],axis=1)
 
    for value in temp_df.drop([primary_key,reference_value],axis=1).columns:
        temp_df[value] = temp_df[value]*temp_df[reference_value]

    temp_df = temp_df[temp_df[reference_value]!=0].drop(reference_value,axis=1)
    
    temp_df = transpose_df(temp_df,primary_key)
    
    return temp_df[temp_df['value']>0]


def update_growth_rates(df,cols=['DEPOSIT_GROWTH','LOAN_GROWTH','TXN_GROWTH']):
    '''
    Simple function to update the growth rates for members who are flagged as no change to be whatever the random assignment
    Provided
    
    '''    
    for col in cols:
        df[col] = np.where(df['MBR_GROWTH_PROFILE']=='NO_CHANGE',df['EXP_GROWTH'],df[col])

def create_mbr_information(branch_df,
                           branch_profile_dict,
                           mbr_profile_df,
                           start_date = datetime.date(2024,1,30),
                           mbr_nbr_start=0,
                           columns_not_weights=['weight']):
    
    # Generate DF for Selecting Member Growth Profile at conclusion

    growth_profile_df = flatten_clean_dict(mbr_growth_attrition_profiles,index_name='MBR_GROWTH_PROFILE',high_low_list_fix=True)
    #growth_profile_df.rename(columns={x:x.replace('[0]','_low').replace('[1]','_high') for x in growth_profile_df.columns},inplace=True)
    mbr_growth_dist_perc = list(growth_profile_df['distribution_high'])
    mbr_growth_profile_list = list(growth_profile_df['MBR_GROWTH_PROFILE'])

            
    def generate_single_branch():
        
        # Iterate through Each Branch
        for branch_record in range(len(branch_df)):
            # Generate Branch Info
            branch_info = branch_df.iloc[branch_record]
            # Create A Temporary DataFrame for the Branch to Populate Members
            temp_mbr_df = pd.DataFrame()

            # Branch Name 
            total = branch_info['MEMBERS']
            temp_mbr_df['MEMBERNBR'] = [x+mbr_nbr_start+1 for x in range(total)]
            temp_mbr_df['STATUS'] = "Active"
            temp_mbr_df['BRANCHNAME'] = [branch_info['BRANCHNAME'] for x in range(total)]
            temp_mbr_df['CITY'] = branch_info['CITY']
            
            branch_mbr_composition = branch_profile_dict[branch_info['BRANCH_MBR_COMPOSITION_CLASS']]['profile']
            branch_mbr_composition_list = list(branch_mbr_composition.keys())
            branch_mbr_composition_perc = list(branch_mbr_composition.values())

            temp_mbr_df['CLASSIFICATION'] = [random.choices(branch_mbr_composition_list,weights=branch_mbr_composition_perc)[0] for x in range(total)]

            temp_mbr_df =  temp_mbr_df.merge(mbr_profile_df,on='CLASSIFICATION',how='left')
            
           # Iterate through Member Profile Dict To Create Final Values 
        
            columns_to_iterate_through = list(mbr_profile_dict[list(mbr_profile_dict.keys())[0]].keys())
                          
            # Based on Distribution of Profile, create Explicit Random Values for a particular Member
            for column in [x for x in columns_to_iterate_through if x not in columns_not_weights]:
                calculate_rng_from_df_low_high(temp_mbr_df,column.upper(),f'{column}_low',f'{column}_high',2)
                    
            # Convert Is Primary Into Binary
            temp_mbr_df['PRIMARY_IS_BEEM'] = np.where(temp_mbr_df['PRIMARY_IS_BEEM']>[np.random.rand() for x in range(len(temp_mbr_df))],1,0)
            temp_mbr_df['HOME_OWNER'] = np.where(temp_mbr_df['HOME_OWNER']>[np.random.rand() for x in range(len(temp_mbr_df))],1,0)
            temp_mbr_df['MORTGAGE_DEBT'] = temp_mbr_df['MORTGAGE_DEBT']* temp_mbr_df['HOME_OWNER']
                       
            # Include Number of Kinds
            create_column_from_dict_distribution(temp_mbr_df,'KIDS','NUMBER_KIDS',general_assumptions['NUMBER_OF_KIDS']['value'])
            
            # Generate Percentage of Known Spending
            temp_mbr_df['PERC_SPENDING_KNOWN'] = temp_mbr_df['BILL_PAY_DEBIT'] +temp_mbr_df['POS_TXN_DEBIT'] + temp_mbr_df['INVESTMENT_DEBIT']
            temp_mbr_df['PERC_INCOME_KNOWN'] = temp_mbr_df['OTHER_DEPOSIT'] +temp_mbr_df['PAYROLL_DEPOSIT']
    
            # Calculate Transaction values based on Income
            txn_values = ['BILL_PAY_DEBIT','POS_TXN_DEBIT','INVESTMENT_DEBIT','PAYROLL_DEPOSIT','OTHER_DEPOSIT']
            for txn_val in txn_values:
                temp_mbr_df[txn_val]  = np.round(temp_mbr_df[txn_val]* temp_mbr_df['ANNUAL_INCOME']*(1/12),2)
                # Apply discount factor Based on Not Being a Primary Member, want to include some Randomness to see if I can find the Pattern.
                flag_ = rng.random(len(temp_mbr_df)) <.98 
                scale_non = rng.uniform(0.25, .75, len(temp_mbr_df))
                scale_beem = rng.uniform(0.85, 1, len(temp_mbr_df))
                # For 98% of Non Primary Members, Clean Out Transaction Activity on Monthly Basis.
                temp_mbr_df[txn_val]  = np.where((flag_)&(temp_mbr_df['PRIMARY_IS_BEEM']==0),0,temp_mbr_df[txn_val])
                # Provide a randomized slightly Decreasing Factor, which differs for Non and Beem Members
                temp_mbr_df[txn_val]  = np.round(np.where(temp_mbr_df['PRIMARY_IS_BEEM']==0,temp_mbr_df[txn_val]*scale_non,temp_mbr_df[txn_val]*scale_beem),2)

            # Calculate Deposit and Loan Values Held with Beem
            # For Loans. If Primary Account is here, 95% that we have Loan Balance and 10% for non members
            # For Deposits If Primary, 80% we have deposit, and 15% we have Loan. 
                        
            temp_mbr_df['MORTGAGE_BALANCE'] = np.where((rng.random(len(temp_mbr_df)) <.95)&(temp_mbr_df['PRIMARY_IS_BEEM']==1),temp_mbr_df['MORTGAGE_DEBT'],(rng.random(len(temp_mbr_df)) <.15)*temp_mbr_df['MORTGAGE_DEBT'])
            temp_mbr_df['DEPOSIT_BALANCE'] = np.where((rng.random(len(temp_mbr_df)) <.8)&(temp_mbr_df['PRIMARY_IS_BEEM']==1),temp_mbr_df['LIQUID_ASSETS'],(rng.random(len(temp_mbr_df)) <.10)*temp_mbr_df['LIQUID_ASSETS'])
            temp_mbr_df['PERC_MORTGAGE'] =  temp_mbr_df['MORTGAGE_BALANCE']/temp_mbr_df['MORTGAGE_DEBT']
            temp_mbr_df['PERC_DEPOSIT'] =   temp_mbr_df['DEPOSIT_BALANCE']/temp_mbr_df['LIQUID_ASSETS']
   
            #temp_mbr_df['CNS_BALANCE'] = temp_mbr_df['CONSUMER_DEBT']
            
            simplistic_engagement_calculation(temp_mbr_df)
            
            temp_mbr_df =  temp_mbr_df.drop(mbr_profile_df.drop('CLASSIFICATION',axis=1).columns.tolist(),axis=1)
            temp_mbr_df['MONTH'] = start_date
            
            return temp_mbr_df
        
    if len(branch_df)==1:
        # Generate ORG Dataset
        return generate_single_branch().merge(branch_df[['BRANCHNAME','EXP_GROWTH']],on='BRANCHNAME',how='left')

    else:
        # Iterate through Branch DF
        final_mbr_df = pd.DataFrame()
        mbr_nbr_start = 0
        
        # Iterate through Branch Profile and Create Members for Each Branch
        for branch in branch_df['BRANCHNAME'].unique():
            temp_df = branch_df[branch_df['BRANCHNAME']==branch]            
            final_mbr_df = pd.concat([final_mbr_df,create_mbr_information(temp_df,branch_profile_dict,mbr_profile_df,mbr_nbr_start=mbr_nbr_start)])
            mbr_nbr_start += temp_df['MEMBERS'].item()

        # Add MBR Growth Profile
        final_mbr_df['MBR_GROWTH_PROFILE'] = rng.choice(mbr_growth_profile_list,p=mbr_growth_dist_perc,size=len(final_mbr_df))
        final_mbr_df = final_mbr_df.merge(growth_profile_df.drop(['Description','distribution_low','distribution_high'],axis=1),on='MBR_GROWTH_PROFILE',how='left')

        # Calculate Member Level Growth and Impact Rates
        drop_cols = []
        for column in [x for x in mbr_growth_attrition_profiles[list(mbr_growth_attrition_profiles.keys())[0]].keys() if (x.find('_growth')!=-1)|(x.find('_impact')!=-1)]:
            calculate_rng_from_df_low_high(final_mbr_df,column.upper(),f'{column}_low',f'{column}_high')
            # Drop Now Unnecessary Values in DataFrame as They have been used to calculate Mbr Level Percentage. Store and then apply once.
            drop_cols.extend([f'{column}_low',f'{column}_high'])
        final_mbr_df.drop(drop_cols,axis=1,inplace=True)
        update_growth_rates(final_mbr_df)


        return final_mbr_df.reset_index(drop=True)


def create_branch(unique_records=40,legacy_distribution=[.5,.15,.3,.05]):
    
    # Create Branch DataFrame
    branch_df = random_uniform_normalized_df(unique_records=unique_records,
                                             name='BRANCHNAME',
                                             LEGACY=legacy_distribution)

    # Add Random City from Selection List such that they are explicitly defined based on Legacy
    branch_df["CITY"] = branch_df["LEGACY"].apply(lambda legacy: pick_from_dict(legacy_city_dict, legacy))

    # Add a Mortgage Multipler, again defined based on Legacy 
    branch_df['MTG_MULTIPLIER'] = branch_df['LEGACY'].map(general_assumptions['MTG_Multiplier']['value'])

    # Branch Efficiency Factor
    branch_df['BEF'] = [np.random.uniform(.98,1.05) for x in range(len(branch_df))]

    # Legacy Efficiency Factor
    branch_df['LEF'] = branch_df['LEGACY'].map(general_assumptions['Legacy_efficieny_factor_dict']['value'])

    # Expected Growth Rate (I wanted to Keep this Small, Not Large)
    branch_df['EXP_GROWTH'] = branch_df['LEF']*branch_df['BEF'] 

    # Select the Default Member Composition Profile to which Classification members will be utilized
    branch_df['BRANCH_MBR_COMPOSITION_CLASS'] = [random.choices(branch_mbr_composition_templates,weights=branch_mbr_composition_templates_perc)[0] for x in range(0,len(branch_df))]

    # Total Number of Members based on Perc allocated and total number as defined in General Assumptions
    branch_df['MEMBERS'] = branch_df['PERC_'].apply(lambda x:int(x*total_mbrs))
    
    return branch_df

    
def create_txn_df(df,
                    exclude = ['INSURANCE_HOUSE','PROPERTY_TAX_PROV','PROPERTY_TAX_MUNI','UTILITIES_GAS','UTILITIES_HYDRO']):
    
    monthly_bp_df = decouple_txn(df[['BILL_PAY_DEBIT','MEMBERNBR','HOME_OWNER']],
                             reference_value='BILL_PAY_DEBIT',
                             txn_dict=monthly_payment_dict,
                             exclude_non_ho=exclude)

    monthly_pos_df = decouple_txn(df[['POS_TXN_DEBIT','MEMBERNBR','HOME_OWNER']],
                                  reference_value='POS_TXN_DEBIT',
                                  txn_dict=pos_txn_dict)
    
    final_df = pd.concat([monthly_bp_df,monthly_pos_df])
    
    final_df['MONTH'] = df['MONTH'].iloc[0]    

    return  final_df


def generate_historical_data(mbr_df,
                             total_months=5,
                             start_date=datetime.date(2024,1,30)):
    
    
    
    # Create a Final DataFrame for storing all data.
    
    # Copy Current Member
    final_mbr = mbr_df.copy()
    
    # Generate a Month 1 Transaction DataFrame
    final_txn = create_txn_df(mbr_df)

    # Initite a Values which will be overridden
    temp_mbr = mbr_df.copy()
    date = start_date
    
    for count in range(total_months):
        temp_mbr,temp_txn = progress_df_one_month(temp_mbr,start_date=date)
        date += relativedelta(months=1)
        final_mbr = pd.concat([final_mbr,temp_mbr])
        final_txn = pd.concat([final_txn,temp_txn])
        
    return final_mbr,final_txn   


def progress_df_one_month(df,
                          start_date=datetime.date(2024,1,30)):
    
    '''
    
    
    # Do I want to Update Liquid Assets and Such based on PRogression.
    # Update Classification of Definition.
    # Update Number of Kids
    # Amortize Debt.
    # Add Mortgage and TD Renewal
    
    '''
    
    def update_month(df,cols_to_clear=['BILL_PAY_DEBIT','POS_TXN_DEBIT','INVESTMENT_DEBIT','PAYROLL_DEPOSIT','OTHER_DEPOSIT','MORTGAGE_BALANCE','DEPOSIT_BALANCE']):
        
        '''
        
        '''
        
        df['MONTH'] = df['MONTH'] + relativedelta(months=1)
        df['AGE'] = np.round(df['AGE']+(30/365),2)
        df['DURATION'] = np.round(df['DURATION']+(30/365),2)

        # Impact Growth Rates
        df['MORTGAGE_BALANCE'] = df['MORTGAGE_BALANCE']*df['LOAN_GROWTH']
        df['DEPOSIT_BALANCE'] =  df['DEPOSIT_BALANCE']*df['DEPOSIT_GROWTH']
        df['ANNUAL_INCOME']    = df['ANNUAL_INCOME']*df['INCOME_GROWTH']
        df['PAYROLL_DEPOSIT'] =  df['PAYROLL_DEPOSIT']*df['INCOME_GROWTH']
        df['POS_TXN_DEBIT'] =   df['POS_TXN_DEBIT']*df['TXN_GROWTH']
        df['BILL_PAY_DEBIT'] =  df['BILL_PAY_DEBIT']*df['TXN_GROWTH']

        # Impact Raw totals

        df['MORTGAGE_BALANCE'] = np.round(df['MORTGAGE_BALANCE'] + df['LOAN_IMPACT'],2)
        df['DEPOSIT_BALANCE'] =  np.round(df['DEPOSIT_BALANCE']  + df['DEPOSIT_IMPACT'],2)
        df['ANNUAL_INCOME']    = np.round(df['ANNUAL_INCOME']    + df['INCOME_IMPACT'],2)
        df['PAYROLL_DEPOSIT'] =  np.round(df['PAYROLL_DEPOSIT'] +  df['INCOME_IMPACT']*(1/12),2)
        df['POS_TXN_DEBIT'] =   np.round(df['POS_TXN_DEBIT']    +  df['TXN_IMPACT']*(1/2),2)
        df['BILL_PAY_DEBIT'] =  np.round(df['BILL_PAY_DEBIT']   +  df['TXN_IMPACT']*(1/2),2)


        for col in cols_to_clear:
            df[col] = np.where(df['STATUS']=='Closed',0,df[col])
    
    
    
    # Need to make a Copy, because I want to retain Original Dataset.
    df = df.copy()
    
    # Randomly Close Some Members Operational Closing
    close_low,close_high = general_assumptions['OPERATIONAL_ATTRITION']['value']
    df['RANDOM_CLOSE'] = rng.binomial(n=1, p=rng.uniform(close_low,close_high,size=len(df)))

    # Randomly Close Members. Health Related.
    health_low, health_high = general_assumptions['MEMBER_HEALTH']['value']
    health_flag = df['AGE'] > 70
    df['HEALTH_CLOSE'] = 0
    df.loc[health_flag, 'HEALTH_CLOSE'] = rng.binomial(n=1, p=rng.uniform(health_low,health_high,size=health_flag.sum()), size=health_flag.sum())
    
    # Update Status of Closed Accounts
    df['STATUS'] = np.where((df['RANDOM_CLOSE']==1)|(df['HEALTH_CLOSE']==1),'Closed',df['STATUS'])
    
    # Update Values based on Update Function
    update_month(df)
    
    # Generate a Transaction DataFrame for the month
    txn_df = create_txn_df(df)   

    return df,txn_df   

def generate_synthetic_dataset(number_branches=40,
                                total_months=18,
                                legacy_distribution=[.5,.15,.3,.05],
                                start_date = datetime.date(2024,1,30)):

    ###########################################################################################################################
    # Create Lists, Dictionaries and Macro Values
    ###########################################################################################################################
    
    ## Generation of Input Data, including Lists, Non Static Dictionaries, etc

    # DataFrame of General Assumption Attributes
    general_assumptions_df = flatten_clean_dict(general_assumptions,high_low_list_fix=True,index_name='CLASSIFICATION')[['CLASSIFICATION','Description','value_low','value_high','model_status']].fillna(0)

    # DataFrame on Member Profile Attributes
    mbr_profile_df = flatten_clean_dict(mbr_profile_dict,index_name='CLASSIFICATION',high_low_list_fix=True)
    
    ###########################################################################################################################
    ## Generation of Data
    ###########################################################################################################################
    
    # Create Branch Netword
    branch_df = create_branch(unique_records=number_branches,
                              legacy_distribution=legacy_distribution)
    
    # Create Member Dataframe
    mbr_df = create_mbr_information(branch_df,
                                    branch_mbr_composition_dict,
                                    mbr_profile_df,
                                    start_date=start_date)
    ###########################################################################################################################


    # Create Historical Information
    final_mbr_df,final_txn_df = generate_historical_data(mbr_df,
                                                         total_months=total_months,
                                                         start_date=start_date)
    
    return final_mbr_df,final_txn_df


################################################################################################################################################################



def transpose_df(df, index, columns=None):
    '''

    Transposes a non-time-series DataFrame from wide to long format by melting specified columns.

    This is especially useful for flattening columns into a single column to support tools 
    like Power BI, where long format enables dynamic pivoting and aggregation.


    Parameters:
        df (DataFrame): The input pandas DataFrame.
        index (list): Columns to retain as identifiers (will remain unchanged).
        columns (list): Columns to unpivot into key-value pairs.

    Returns:
        DataFrame: A long-format DataFrame with 'variable' and 'value' columns.

    date_created:1-Jan-24
    date_last_modified: 30-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        Example Function Call

    '''
    if not columns:
        columns = [col for col in df.columns if col not in index]
    return df.melt(id_vars=index, value_vars=columns)   

def random_uniform_normalized_list(n, skew=1):

    """
    Function to create a list of RNG numbers for the purposes of creating a distribution.
    Values equal 1.

    Parameters:
        n(int): Number of Values to Return in list.
        skew(int): Skew to include in data, Values Greater than 0 will create 

    Returns:
        list

    date_created:29-Dec-25
    date_last_modified: 29-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        create_distribution_weight(5)


    """  
    # Generate random positive numbers
    raw = rng.random(n) ** skew  # apply skew
    weights = raw / raw.sum()    # normalize to sum to 1
    return [float(w) for w in weights]

def random_choice_from_uniform_list(total_records,
                                    name="Example",
                                    distinct_entities=0,
                                    list_distribution=[],
                                    return_value=None,
                                    skew=1):
    '''
    Create a random generate list from provided inputs. List is of length as defined in total records, the name of the records is defined in name. 
    The distribution of values is conditionally determined by either distinct entities, or the distribution as provided in list_distribution.

    Parameters:
        total_records(int): Number of records to be returned in list.
        name(str): Name of Random Records.
        distinct_entities(int): If populated, it will be used to generate a random distribution of defined values, also used as the number of reocrds
        list_distribution(list): Distribution to be used for random sampling.
        return_value(str): Default to None, and will return a list. Can input 'df' to return a dataframe
        skew(float): Skew to include in random distribution.
        
    Returns:
        list
        if return_value is 'df' then DataFrame

    date_created:29-Dec-25
    date_last_modified: 29-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        random_choice_from_uniform_list(unique_records=40,name='BRANCHNAME',LEGACY=[.5,.15,.3,.05])
    '''
    
    if (distinct_entities==0)&(list_distribution==[]):
        raise TypeError('User must select either Number of Distinct Entries or Provide a Distribution')
    
    if distinct_entities==0:
        distinct_entities = len(list_distribution)
        
    if len(list_distribution)==0:
        list_distribution = random_uniform_normalized_list(distinct_entities,skew=skew)
        
    name_list = [f"{name} {x+1}" for x in range(0,distinct_entities)]
    
    final_list = [random.choices(name_list,weights=list_distribution)[0] for x in range(0,total_records)]
    
    if return_value=='df':
        return pd.DataFrame(final_list,columns=[name])
    else:
        return final_list
    


def replicate_df_row(df,records=5):
    
    '''
    Function which Replicates a single row DataFrame for the purposes of Multiplying it against a larger row.
    Function written using tile, which is a C based language, and considerably faster than straight using nunpy vectorized Calculations.

    Parameters:
        df(dataframe): DataFrame which you wish to extend, should be a Single Row, but techincally it will duplicate any size
        records(int): Number of times you wish DF to be duplicated, ideally it should be len(other_df) to which you want to multiply

    Returns:
        df

    date_created:30-Dec-25
    date_last_modified: 30-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        df = pd.DataFrame([[1,2,3]],columns=['A','B','C'])
        replicate_df_row(df)
    
    '''
    
    row = df.to_numpy()
    columns = df.columns.tolist()

    # Repeat row N times using NumPy
    data = np.tile(row, (records, 1))  # shape (N, len(row))
    return pd.DataFrame(data, columns=columns)

def random_uniform_normalized_df(unique_records,
                                 name='Example',
                                 skew=1.25,
                                 **kwargs):
    '''
    Create a Dataframe (which is a series of n * 1) of Random Values for purposes of creating a Random Distribution DataFrame.
    Kwargs can be used to create New Columns. Kwargs should be Lists of distribution Frequencies, to create new random Columns (Not cdf).

    Parameters:
        unique records(int): Number, representing the number of random columns to be included in the output DF.
        name(str): Name of Column to Included (values will be numbered).
        skew(float): If Data is to have a skewed distribution, 1 will be normal uniform (mean=1,std_dev=0).
        **kwargs: Should be List of values equalling 1, to create a new random value.

    Returns:
        Object Type

    date_created:29-Dec-25
    date_last_modified: 29-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        random_uniform_normalized_df(unique_records=40,name='BRANCHNAME',LEGACY=[.5,.15,.3,.05])
    
    '''
    obs_name_list = [f'{name} {x+1}' for x in range(0,unique_records)]
    dist_perc = random_uniform_normalized_list(unique_records,skew=skew)
    
    final_df = pd.DataFrame()
    
    for obs in range(0,unique_records):
        obs_name = obs_name_list[obs]
        perc_ = dist_perc[obs]
        temp_df = pd.DataFrame([[obs_name,perc_]],columns=[name,'PERC_'])
        final_df = pd.concat([final_df,temp_df])
            
    for kwarg_name, kwarg_value in kwargs.items():
        temp_df = random_choice_from_uniform_list(1000,name=kwarg_name,list_distribution=kwarg_value,return_value='df')
        final_df = final_df.reset_index(drop=True).merge(temp_df,left_index=True,right_index=True,how='left')
    
    return final_df
def time_series_statistics(df,calculuation_periods=[1,3,6,12],skipna=True):
    
    '''
    
    
    
    # Note How I calucated PERIOD CHANGE and CHANGE PERC. Made simpler to avoid Complications and error out.s

    '''
    
    temp_df = df.copy()

    # Generate Month List
    month_list = sorted(df.columns)

    # Generate a New Final Dataframe to Store Calculated Values.
    final_df = temp_df.copy()

    # Use of Slicing Only possible because New values are saved directly into final_df, while calculations are 
    # primary undertaken on temp_df. Note that referencing newly created columns must be via final_df.
      
    # Create Life Time Metrics
    
    final_df['CHG_DF']  = temp_df[month_list[-1]]-temp_df[month_list[0]]
    final_df['AVG_DF'] = temp_df[month_list].mean(axis=1, skipna=skipna)
    try:
        final_df[f'PERC_CHG'] = final_df[f'CHG_DF']/temp_df[month_list[-1]]
    except:
        final_df[f'PERC_CHG'] = 100
          
    final_df['STD'] = temp_df[month_list].std(axis=1, skipna=skipna)
    final_df['MAX'] = temp_df[month_list].max(axis=1, skipna=skipna)
    final_df['MIN'] = temp_df[month_list].min(axis=1, skipna=skipna)
    final_df['COUNT'] = temp_df[month_list].count(axis=1)
    
    # Period Change Calculation as Calculated be Calculation_Periods
    for period in calculuation_periods:
        if period==1: # Period Difference Hard Coded to 2 instead of 1.
            final_df[f'AVG_{period}M'] = np.round(temp_df[month_list[-period:]].mean(axis=1, skipna=skipna),2)
            final_df[f'CHG_{period}M'] = np.round(temp_df[month_list[-1]]-temp_df[month_list[-2]],2)
        else:
            final_df[f'AVG_{period}M'] = np.round(temp_df[month_list[-period:]].mean(axis=1, skipna=skipna),2)
            final_df[f'CHG_{period}M'] = np.round(temp_df[month_list[-1]]-temp_df[month_list[-period]],2)
        try:
            final_df[f'PERC_CHG_{period}M'] = final_df[f'CHG_{period}M']/temp_df[month_list[-1]]
            final_df[f'PERC_CHG_{period}M'] = final_df[f'PERC_CHG_{period}M'].fillna(0)
        except:
            final_df[f'PERC_CHG_{period}M'] = 0
    
    return final_df



def flatten_clean_dict(dict_,index_name='INDEX',clean=True,apply_new_lvl=False,high_low_list_fix=False):
    
    '''
    Function which takes a Nested Dictionary (Dictionary, which references dictionary, and converts it into a DataFrame, works best when
    Dictionary ultimate Values are List.

    Parameters:
        dict_(dict): Nested Dictionary
        index_name(str): Default Name of Column to be applied to First Level Dictionary in DataFrame.
        clean(bool): Used to convert a single flat DF to a Matrix, which each new column
        apply_new_level(bool): Optional Argument to support the application to a Single Dictionary
    Returns:
        DataFrame

    date_created:30-Dec-25
    date_last_modified: 30-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        from synthetic_member import mbr_profile_dict
        flatten_clean_dict (mbr_profile_dict)
    
    '''

    if apply_new_lvl:
        dict_ = {'ADDED_LEVEL':dict_}
    
    from collections.abc import Mapping, Sequence

    def flatten_dict(obj, parent_key="", sep="."):
        """
        Recursively flattens a nested dict/list into a flat dict of {column_name: value}.
        - Dict keys are joined with `sep`.
        - List elements are addressed with [index].
        - Scalars become leaf values.

        Examples:
          {"a": {"b": 1}} -> {"a.b": 1}
          {"a": [ {"x": 1}, {"x": 2} ]} -> {"a[0].x": 1, "a[1].x": 2}
        """
        items = {}

        # Dict-like
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                items.update(flatten_dict(v, new_key, sep=sep))

        # List/tuple-like (but not str/bytes)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                items.update(flatten_dict(v, new_key, sep=sep))

        # Scalar (leaf)
        else:
            items[parent_key] = obj

        return items


    def flatten_to_df(data, sep="."):
        """
        Accepts:
          - a single dict -> returns a 1-row DataFrame with one column per leaf.
          - a list/iterable of dicts -> returns a DataFrame with one row per item.
          - a mix (list containing dicts and scalars) -> flattens each element.

        Column names reflect the nested paths using `sep` and [index] for lists.
        """
        # Single dict -> one row
        if isinstance(data, Mapping):
            flat = flatten_dict(data, sep=sep)
            return pd.DataFrame([flat])

        # Iterable -> one row per element
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            rows = []
            for elem in data:
                if isinstance(elem, Mapping) or (isinstance(elem, Sequence) and not isinstance(elem, (str, bytes, bytearray))):
                    rows.append(flatten_dict(elem, sep=sep))
                else:
                    # scalar element becomes a single column named by its position
                    rows.append({ "value": elem })
            return pd.DataFrame(rows)

        # Scalar -> single column/value
        return pd.DataFrame([{"value": data}])
    
    flat_df = flatten_to_df(dict_)
    
    if not clean:
        return flat_df
    
    else:
        final_df = pd.DataFrame()
        for classification in dict_.keys():
            # Select Columns with Classification in it:
            name = f'{classification}.'
            cols = [c for c in flat_df.columns if c.startswith(name)]
            temp_df = flat_df[cols].copy()
            temp_df = temp_df.rename(columns={x:x.replace(name,"") for x in cols})
            temp_df.rename(index={0:classification},inplace=True)
            final_df = pd.concat([final_df,temp_df])
        final_df = final_df.reset_index().rename(columns={'index':index_name})
        
        # Update for Usage of Bracketing for High/ Low
        if high_low_list_fix:
             final_df.rename(columns={x:x.replace('[0]','_low').replace('[1]','_high') for x in final_df.columns},inplace=True)
        
        return final_df
