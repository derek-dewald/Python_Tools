# File Description: Generation of Test Data Sets, Toy Data sets,  quick Access to Scikit Learn Files and other non file based Information.

from sklearn.datasets import fetch_openml
from sklearn import datasets
import pandas as pd
import numpy as np

from TextFunctions import extract_doc_sections_all
from MLPipeline import SKLearnModelList

def sklearn_dataset(df):
    # Load the diabetes dataset
    df1 = pd.DataFrame(df['data'],columns=df['feature_names'])
    df1 = pd.concat([df1,pd.DataFrame(df['target'],columns=['Target'])],axis=1)
    return df1

diabetes_df = sklearn_dataset(datasets.load_diabetes())
iris_df = sklearn_dataset(datasets.load_iris())

def FakeBaseballstats():
    Members = ['Roger Dorn','Rick Vaughn','Lou Brown','Pedro Cerrano','Rachel Phelps','Jake Taylor']
    Team = ['Montreal Expos','Boston Red Sox','New York Yankees','Toronto Blue Jays']
    years = [1990,1991,1992,1993,1994]
    games = 162

    final_df = pd.DataFrame()

    for year in years:
        for day in range(0,games):
            for player in Members:
                ab = np.random.randint(1,6)
                hits = np.random.randint(0,ab)
                temp_df = pd.DataFrame([[player,np.random.choice(Team),hits,ab,year]],columns=['Player','Opponent','Hits','Abs','Year'])
                final_df = pd.concat([final_df,temp_df])
        final_df = final_df.reset_index(drop=True)
    return final_df


member_dict = {'Baller':{'deposit':[100000,1000000],
                         'loan':[0,0,1000000],
                         'tran_count':[0,100],
                         'tran_value':[0,10000]},
               'Spender':{'deposit':[0,100000],
                         'loan':[40000,0,1000000],
                         'tran_count':[0,200],
                         'tran_value':[0,1000]},
               'Saver':{'deposit':[100000,500000],
                         'loan':[0,0,100000],
                         'tran_count':[0,50],
                         'tran_value':[0,100]},
               'Broke':{'deposit':[0,10000],
                         'loan':[20000,0,1000000],
                         'tran_count':[0,30],
                         'tran_value':[0,300]},
               'Bankrupt':{'deposit':[0,100],
                         'loan':[100000,0,1000000],
                         'tran_count':[0,10],
                         'tran_value':[0,100]},
               'Norm':{'deposit':[0,1000000],
                         'loan':[5000,0,1000000],
                         'tran_count':[0,20],
                         'tran_value':[0,1000]},
               'Inactive':{'deposit':[10000,100000],
                         'loan':[0,0,10000],
                         'tran_count':[0,5],
                         'tran_value':[0,375]},
               'Borrower':{'deposit':[0,50000],
                         'loan':[100000,0,1000000],
                         'tran_count':[0,10],
                         'tran_value':[0,1000]},
               'Full_Service':{'deposit':[100000,1000000],
                         'loan':[75000,0,1000000],
                         'tran_count':[0,100],
                         'tran_value':[0,3000]}}

def GrowthRate(value,outlook):
    '''
    Function to Generate a Synthic Growth Percentage, used to support GenerateFakeMember
    
    
    '''
    if outlook==-2:
        return max(0,value - np.random.uniform(.15,.3)*value)
    elif outlook==-1:
        return max(0,value - np.random.uniform(.05,.15)*value)
    elif outlook==0:
        return max(0,value + np.random.uniform(-.05,.05)*value)
    elif outlook==1:
        return max(0,value + np.random.uniform(0,.15)*value)
    elif outlook==2:
        return max(0,value + np.random.uniform(.15,.3)*value)

class Member:
    '''
    Class Variable, Used to Popualate GenerateFakeMember Dataset
    
    
    '''

    def __init__(self, mbr_nbr,status,month,member_dict=member_dict):
        self.id = mbr_nbr        
        self.classification = np.random.choice(list(member_dict.keys()))
        self.outlook = np.random.choice([-2, -1, 0, 1, 2])
        self.active=status
        self.month=month
        # Initial values
        self.deposit = np.random.randint(*member_dict[self.classification]['deposit'])
        self.loan = np.random.choice([
            member_dict[self.classification]['loan'][0],
            np.random.randint(*member_dict[self.classification]['loan'][1:])
        ])
        self.tran_count = np.random.randint(*member_dict[self.classification]['tran_count'])
        self.tran_value = self.tran_count * np.random.randint(*member_dict[self.classification]['tran_value'])
        
    def to_dict(self, month):
        return {
            'MEMBERNBR': self.id,
            'MONTH': month,
            'CLASSIFICATION': self.classification,
            'OUTLOOK': self.outlook,
            'DEPOSIT': int(self.deposit),
            'LENDING': int(self.loan),
            'TXN_COUNT': int(self.tran_count),
            'TXN_VALUE': int(self.tran_value),
            'ACTIVE':int(self.active)
        }
    
    def update_month(self, month,GrowthRateFunc,attrition_perc):
        if self.active == 0:
            self.month=month
            return self.to_dict(month)
        elif np.random.uniform(0,1)<=attrition_perc:
            self.month=month
            self.active = 0
            self.deposit = 0
            self.loan= 0
            self.tran_count = 0
            self.tran_value = 0
            return self.to_dict(month)
        else:
            self.month=month
            self.deposit = GrowthRateFunc(self.deposit, self.outlook)
            self.loan = GrowthRateFunc(self.loan, self.outlook)
            self.tran_count = max(0, int(GrowthRateFunc(self.tran_count, self.outlook)))
            self.tran_value = max(0, int(GrowthRateFunc(self.tran_value, self.outlook)))
            return self.to_dict(month)
        
def GenerateFakeMemberDF(mbrs,months,attrition_perc=.05,growth_max_perc=.1):
    '''
    Function to Create a Fake Customer Dataset for Analyzing Over Time Series.
    
    
    '''
    members = {}
    final_df = pd.DataFrame()
    current_id = 0

    for _ in range(mbrs):
        m = Member(current_id, 1,0)
        members[current_id] = m
        final_df = pd.concat([final_df, pd.DataFrame([m.to_dict(0)])])
        current_id += 1

    for month in range(1, months+1):  # months 1 to 18 inclusive
        monthly_records = []

        for m in members.values():
            updated = m.update_month(month=month, GrowthRateFunc=GrowthRate,attrition_perc=attrition_perc)
            monthly_records.append(updated)

        new_member_count = np.random.randint(0, mbrs*growth_max_perc)
        for _ in range(new_member_count):
            m = Member(current_id,1,month)
            members[current_id] = m
            monthly_records.append(m.to_dict(month))
            current_id += 1

        # 3. Append everything for this month to the final DataFrame
        final_df = pd.concat([final_df, pd.DataFrame(monthly_records)], ignore_index=True)  
    return final_df


def GenerateSKModelDoc():

    sklearn_model_df = SKLearnModelList()
       
    param_df = pd.DataFrame()
    desc_df = pd.DataFrame()

    for index,row in sklearn_model_df.iterrows():
        temp_para,temp_desc = extract_doc_sections_all(row['Estimator Class'],model_name=row['Model Name'])
        param_df = pd.concat([param_df,temp_para])
        desc_df = pd.concat([desc_df,temp_desc])

    sklearn_model_df = sklearn_model_df.merge(desc_df,on='Model Name',how='left') 

    sklearn_model_df.to_csv('SKLearnModels.csv',index=False)
    param_df.to_csv('SKlearnParameterList.csv',index=False)
    
    return sklearn_model_df,param_df


def MNIST_SKLEARN(normalize=True, flatten=False, random_state=42,return_value=None):
    """
    Loads the MNIST dataset from OpenML and returns train/test sets.

    Parameters:
        normalize (bool): If True, scale pixel values to [0, 1].
        flatten (bool): If True, keep images as (784,) instead of (28, 28).
        random_state (int): Random seed for reproducibility.

    Returns:
        
    """
    from tensorflow.keras.datasets import mnist
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape)
    
    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test  = X_test.reshape(len(X_test), -1)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    print(X_train.shape)
    
    return X_train,y_train,X_test,y_test