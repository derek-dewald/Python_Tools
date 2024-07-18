def TraderStatistics(df,
                     target,
                     merge_back=0,
                     trader='POOL1_TRADER',
                     trader1='POOL2_TRADER',
                     amount='p1.1.transaction_amt',
                     amount1='p1.0.transaction_amt',
                     buy_sell1='BUY_SELL_FLAG_POOL1',
                     buy=-1,
                     sell=1,
                     buy_sell2='BUY_SELL_FLAG_POOL2',
                     date='DATE',
                     time='time'):
    
    # Need to Decouple Pool 1 and Pool 2 into distinct Time Sorted Values
    temp_0 = df[[date,time,trader,amount,target,buy_sell1]].rename(columns={trader:'TRADER',amount:'VALUE',buy_sell1:'BUY_SELL'})
    temp_1 = df[[date,time,trader1,amount1,target,buy_sell2]].rename(columns={trader1:'TRADER',amount1:'VALUE',buy_sell2:'BUY_SELL'})
    tran_df = pd.concat([temp_0,temp_1])
    
    final_df = tran_df.pivot_table(index='TRADER',columns='BUY_SELL',values='VALUE',aggfunc=sum).reset_index().rename(columns={buy:'PURCHASE_VALUE',sell:"SELL_VALUE"})
    final_df = final_df.merge(tran_df.pivot_table(index='TRADER',columns='BUY_SELL',values='VALUE',aggfunc='count').reset_index().rename(columns={buy:'PURCHASE_VOLUME',sell:"SELL_VOLUME"}),on='TRADER',how='left')
    
    arb_tran_df = tran_df[tran_df[target]==1].copy()
    
    final_df = final_df.merge(arb_tran_df.pivot_table(index='TRADER',columns='BUY_SELL',values='VALUE',aggfunc=sum).reset_index().rename(columns={buy:'PURCHASE_VALUE_ARB',sell:"SELL_VALUE_ARB"}),on='TRADER',how='left')
    final_df = final_df.merge(arb_tran_df.pivot_table(index='TRADER',columns='BUY_SELL',values='VALUE',aggfunc='count').reset_index().rename(columns={buy:'PURCHASE_VOLUME_ARB',sell:"SELL_VOLUME_ARB"}),on='TRADER',how='left')
    
    
    final_df['ARB_PURCHASE_VAL'] = final_df['PURCHASE_VALUE_ARB']/final_df['PURCHASE_VALUE']
    final_df['ARB_PURCHASE_VOL'] = final_df['PURCHASE_VOLUME_ARB']/final_df['PURCHASE_VOLUME']
    final_df['ARB_SELL_VAL'] =     final_df['SELL_VALUE_ARB']/final_df['SELL_VALUE']
    final_df['ARB_SELL_VOL'] =     final_df['SELL_VOLUME_ARB']/final_df['SELL_VOLUME']
    
    final_df['TOTAL_VOLUME'] =  final_df['PURCHASE_VOLUME'] + final_df['SELL_VOLUME']
    final_df['TOTAL_VALUE']  =  final_df['PURCHASE_VALUE']  + final_df['SELL_VALUE']
    
    return tran_df,final_df

def RollingWindow(df,
                  new_column_name,
                  value_column,
                  net_volume_column=[],
                  grouping=[],
                  time_col="",
                  minutes=60,
                  calculation_type='val'):
    
    '''
    
    net_volume_column - For instances where direction of transaction matters, enables a user to calculate
    
    '''
    
    # Create Copy
    
    minutes = f"{minutes}min"
    new_col_vol = f"{new_column_name}_vol"
    new_col_val = f"{new_column_name}_val"
    new_col_net_vol = f"{new_column_name}_net_position"

    df = df.copy()
    if len(time_col)>0:
        df.set_index(time_col,inplace=True)
    
    if len(grouping)==0:
        df[new_col_val] = df[value_column].rolling(minutes).sum()
        df[new_col_v0l] = df[value_column].rolling(minutes).count()
        
        if len(net_volume_column)>0:
            df[new_col_net_vol] = df[net_volume_column].rolling(minutes).sum()
            
    else:
        df[new_col_val] = df.groupby(grouping)[value_column].rolling(minutes).sum().reset_index(level=0,drop=True)
        df[new_col_vol] = df.groupby(grouping)[value_column].rolling(minutes).count().reset_index(level=0,drop=True)
        
        if len(net_volume_column)>0:
            df[new_col_net_vol] = df.groupby(grouping)[net_volume_column].rolling(minutes).sum().reset_index(level=0,drop=True)
       
    if len(time_col)>0:
        return df.reset_index()
    else:
        return df
    
    
def AggregatePoolCalculations(df,
                              merge_back=1,
                              trader='POOL1_TRADER',
                              trader1='POOL2_TRADER',
                              amount='p1.1.transaction_amt',
                              amount1='p1.0.transaction_amt',
                              date='DATE',
                              time='time'):
    
    # Need to Decouple Pool 1 and Pool 2 into distinct Time Sorted Values
    temp_0 = df[[date,time,trader,amount]].rename(columns={trader:'TRADER',amount:'VALUE'})
    temp_1 = df[[date,time,trader1,amount1]].rename(columns={trader1:'TRADER',amount1:'VALUE'})
    temp_ = pd.concat([temp_0,temp_1])
    
    # Add Count as we will consolidate in the case of Concurrent Timing Transactions
    # Add Absolute Value so we can track Aggregate Activity and Also Net Position.
    temp_['TRADE_COUNT'] = 1
    temp_['ABS_VALUE'] = temp_['VALUE'].apply(lambda x:abs(x))
    
    # Aggregate to ensure that transactions occuring at the exact same time are considered.
    
    temp_ = temp_.groupby([date,time,'TRADER']).sum().reset_index()
    temp_['TRADER_COMBINED_POOL_DAILY_VAL'] = temp_.groupby(['DATE','TRADER'])['ABS_VALUE'].cumsum()
    temp_['TRADER_COMBINED_POOL_DAILY_VOL'] = temp_.groupby(['DATE','TRADER'])['TRADE_COUNT'].cumsum()
    
    temp_['TRADER_NET_DAILY_VAL_POSITION'] = temp_.groupby(['DATE','TRADER'])['ABS_VALUE'].cumsum()
    temp_['TRADER_NET_DAILY_BUY_SELL'] = temp_.groupby(['DATE','TRADER'])['TRADE_COUNT'].cumsum()
    
    if merge_back==1:
        trade1_block = temp_.rename(columns={x:x.replace('TRADER','POOL1_TRADER') for x in temp_.columns}).drop(['TRADE_COUNT','VALUE','ABS_VALUE'],axis=1)
        trade2_block = temp_.rename(columns={x:x.replace('TRADER','POOL2_TRADER') for x in temp_.columns}).drop(['TRADE_COUNT','VALUE','ABS_VALUE'],axis=1)

        return df.merge(trade1_block,on=["time",'POOL1_TRADER','DATE'],how='left').merge(trade2_block,on=["time",'POOL2_TRADER','DATE'],how='left')
    return temp_
import pandas as pd
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt

trader1 = ['DUDE1','DUDE2','DUDE3','DUDE4','DUDE5','DUDE6','DUDE7','DUDE8','DUDE9','DUDE10']
trader2 = ['DUDE2','DUDE6','DUDE8','DUDE9','DUDE0','DUDEZ','DUDES','DUDEQ','DUDEW','C']

records = 10000

tran_time = []
now = datetime.datetime.now()
dif = int(np.random.choice(range(0,100)))

for i in range(records):
    tran_time.append(now)
    now += datetime.timedelta(seconds=dif)
    
date_list = [x.strftime('%d-%b-%y') for x in tran_time]
    
target = [1 if np.random.rand()>.8 else 0 for x in range(records)]

trade_list1 = [np.random.choice(trader1) for x in range(records)]
trade_list2 = [np.random.choice(trader2) for x in range(records)]

amount_list1 = [random.randrange(-50000,50000) for x in range(records)]
amount_list2 = [random.randrange(-10000,50000) for x in range(records)]

trade_type1 = [1 if x>0 else -1 for x in amount_list1]
trade_type2 = [1 if x>0 else -1 for x in amount_list2]

data = {'time':tran_time,
        'DATE':date_list,
        'POOL1_TRADER':trade_list1,
        'POOL2_TRADER':trade_list2,
        'BUY_SELL_FLAG_POOL1':trade_type1,
        'BUY_SELL_FLAG_POOL2':trade_type2,
        'p1.1.transaction_amt':amount_list1,
        'p1.0.transaction_amt':amount_list2,
       'target':target}


df = pd.DataFrame(data)
df

# Add The Aggreate Trader Position Which Looks at the Net Position From Both Pools, by stacking unique Transactions from 
# Both Pools and then calculating by Time, then merging back in

df1 = AggregatePoolCalculations(df)

## NEED TO ADD UNIQUE TRANSACTIONS INTO DATAFRAME

# Calculates Transactions Based on a Rolling Window Approach. Use Visualization and ML to determine what an approrpiate Window looks like

df1 = RollingWindow(df1,
                    new_column_name='POOL1_TRADER_60M_WINDOW',
                    value_column='p1.1.transaction_amt',
                    net_volume_column=['BUY_SELL_FLAG_POOL1'],
                    grouping=['POOL1_TRADER'],
                    time_col='time')

df1['POOL1_DAILY_VAL'] = df1.groupby(['DATE'])['p1.1.transaction_amt'].cumsum()
df1['POOL1_DAILY_VOL'] = df1.groupby(['DATE'])['p1.1.transaction_amt'].cumcount()+1

df1['POOL1_TRADER_DAILY_VAL'] = df1.groupby(['DATE','POOL1_TRADER'])['p1.1.transaction_amt'].cumsum()
df1['POOL1_TRADER_DAILY_VOL'] = df1.groupby(['DATE','POOL1_TRADER'])['p1.1.transaction_amt'].cumcount() +1

df1

def SimpleBar(df,
               x_axis,
               y_axis,
               title="",
               binary_split_col="",
               figsize=(10,5),
               return_value='graph'):
    
    plt.figure(figsize=figsize)
      
    if binary_split_col!="":
        plt.plot(df[x_axis],df[y_axis],label='Population',alpha=.2,color='black')
        on = df[df[binary_split_col]==1]
        off = df[df[binary_split_col]==0]
        plt.plot(on[x_axis],on[y_axis],label='Target',alpha=.4,color='green')
        plt.plot(off[x_axis],off[y_axis],label='Not Target',alpha=.6,color='grey')
    
    else:
        plt.plot(df[x_axis],df[y_axis],label='Population')
    
    plt.legend()
    plt.title(title)
    
    if return_value == 'graph':
        plt.show()
    else:
        return fig
    
SimpleBar(a,
           x_axis='time',
           title='Daily Pool Aggregate Value over Time',
           y_axis='POOL1_DAILY_VAL',
           binary_split_col='target')  

def SimpleHist(df,
               x_axis,
               title="",
               binary_split_col="",
               figsize=(10,5),
               return_value='graph'):
    
    plt.figure(figsize=figsize)
      
    if binary_split_col!="":
        plt.hist(df[x_axis],label='Population',alpha=.2,color='grey')
        on = df[df[binary_split_col]==1]
        off = df[df[binary_split_col]==0]
        plt.hist(on[x_axis],label='Target',alpha=.5,color='blue')
        plt.hist(off[x_axis],label='Not Target',alpha=.2,color='green')
    
    else:
        plt.hist(df[x_axis],label='Population')
    
    plt.legend()
    plt.title(title)
    
    if return_value == 'graph':
        plt.show()
    else:
        return fig
    
SimpleHist(a,
           x_axis='POOL1_DAILY_VAL',
           title='Daily Pool Aggregate Histogram',
           binary_split_col='target')  

def IterateThroughColumnsToVisualize(df,exclude=['time','DATE','POOL1_TRADER','POOL2_TRADER','target']):
    for column in df.columns:
    
        if column in exclude:
            pass
        else:
            SimpleHist(df,
                       x_axis=column,
                       title=f'{column} Histogram',
                       binary_split_col='target')
            
IterateThroughColumnsToVisualize(df1)


def brackets(df,column_name,new_column_name,desired_list=[0,10,20,30,40,50,60,70,80,90,100],
                      less_text='Less than',other_text='Between',greater_text='Greater than'):
    
    '''
    Purpose: Simple pre-defined formula to create STR definiton of Value Bucket

    Input: DataFrame, Column Name to Evaluate (No Format), New Column Name (No Format)
    
    Constraint: If Last Number is Lower bound constraint, then constraint = "lower", this duplicates the last item in the
    list and ensures that Valuation is completed correctly. Else it is ignored.

    List of values to evaluate, Maximum 10 values, can do less
    
    Default: list evenly spaced between 0 - 100

    Notes: Should consider enhancing a output list of easy filtering.

    '''
 
    desired_list.append(desired_list[-1:][0])
    
    condition = []
    value = []

    for count,i in enumerate(desired_list):
        if count == 0:
            condition.append(df[column_name]<=i)
        elif count == len(desired_list)-1:
            condition.append(df[column_name]>=i)
        else:
            condition.append(df[column_name]<=i)
        
    for count,i in enumerate(desired_list):
        if count == 0:
            value.append(f"{less_text} {i:,}")
        elif count == len(desired_list)-1:
            value.append(f"{greater_text} {i:,}")
        else:
            value.append(f"{other_text} {desired_list[count-1]:,} and {desired_list[count]:,}")
              
    df[new_column_name]=np.select(condition,value)
    
    return df
