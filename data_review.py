def ReviewColumn(df,column_name):

    '''
    

    '''
    

    temp = pd.DataFrame(df[column_name].value_counts().head()).reset_index().rename(columns={column_name:'Value'})
    temp['Metric'] = i
    temp['Observation'] = 'Top 5 Value Count'

    non_zero = len(df[df[column_name].notnull()])
    zero = len(df[df[column_name]==""])
    blank = len(df[df[column_name].isnull()])

    temp_2 = pd.DataFrame([len(df),non_zero,zero,blank],index=['Dataframe Length','Not Null Record Count','Record is Blank Count','Null Record Count'],columns=['count']).reset_index().rename(columns={'index':'Value'})
    temp_2['Metric'] = column_name
    temp_2['Observation'] = 'Data Quality'
    
    try:
        mean = df[i].mean()
        median = df[i].median()
        std = df[i].std()
        min = df[i].min()
        max = df[i].max()

        temp_1 = pd.DataFrame([mean,median,min,max],index=['MEAN','MEDIAN','MIN','MAX'],columns=['count']).reset_index().rename(columns={'index':'Value'})
        temp_1['Metric'] = column_name
        temp_1['Observation'] = 'Statistical Review' 

        return pd.concat([temp_2,temp_1,temp])
    
    except:
        return pd.concat([temp_2,temp])
