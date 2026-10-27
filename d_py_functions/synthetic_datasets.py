'''
module_word: synthetic_datasets
module_definition: Functions which are created for various testing and example purposes. Note, you can always expand them, try to preserve column Naming.

'''
import pandas as pd
import numpy as np

def synthetic_txn_dataset(months=6):
    '''
    Synthetic Dataframe for purposes of visualization


    
    '''
    rng = np.random.default_rng(42)
    
    transaction_types = [
        "ATM Withdrawal",
        "Bill Payment",
        "Debit Purchase",
        "E-Transfer",
        "Wire Transfer",
        "Cheque",
        "Cash Deposit",
        "Online Transfer"
    ]
    
    value_segments = [
        "$0-$100",
        "$100-$500",
        "$500-$1K",
        "$1K-$5K",
        "$5K-$10K",
        "$10K+"
    ]
    
    # Baseline transaction volumes by transaction type
    base_volumes = [12000, 18000, 30000, 22000, 1200, 4500, 6000, 15000]
    
    # Volume distribution across transaction value segments
    segment_weights = np.array([1.0, 0.75, 0.45, 0.20, 0.08, 0.025])
    
    # Baseline percentage by transaction type
    base_rates = [1.2, 0.8, 0.5, 2.5, 6.0, 1.8, 3.0, 1.5]
    
    records = []
    
    for i, transaction in enumerate(transaction_types):
        for j, segment in enumerate(value_segments):
    
            count = max(
                1,
                int(base_volumes[i] * segment_weights[j] * rng.uniform(0.7, 1.3))
            )
    
            percentage = np.clip(
                base_rates[i] + j * 0.9 + rng.normal(0, 1.2),
                0.1,
                35.0
            )
    
            records.append({
                "FINAL_TRANSACTION": transaction,
                "TRANSACTION_VAL_SEGMENT": segment,
                "FINAL_DECISION_COUNT": count,
                "PERCENTAGE": round(percentage, 2)
            })
    
    df2 = pd.DataFrame(records)
    df2['TOTAL'] = (df2['FINAL_DECISION_COUNT']/(df2['PERCENTAGE']/100)).astype(int)
    df2["TRANSACTION_VAL_SEGMENT"] = pd.Categorical(
        df2["TRANSACTION_VAL_SEGMENT"],
        categories=value_segments,
        ordered=True
    )

    if months>0:
        month_ends = pd.date_range(end=pd.Timestamp.today().date(),periods=months,freq='ME')

    for count,month in enumerate(month_ends):
        if count ==0:
            df2['SNAPSHOTDATE'] = month
        else:
            temp = df2[df2['SNAPSHOTDATE']==month_ends[count-1]].copy()
            temp['SNAPSHOTDATE'] = month

            for column in ['FINAL_DECISION_COUNT','TOTAL']:
                temp[column] = temp[column]*np.random.uniform(0.75, 1.25,size=len(temp))

            temp['PERCENTAGE'] = temp['FINAL_DECISION_COUNT']/temp['TOTAL']
            
            df2 = pd.concat([df2,temp])    
        

    return df2