import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def Heatmap(df,
            correlation=True,
            column_list=[],
            title='Heat Map of Correlation',
            cmap='coolwarm',
            annotate=True,
            x_rotate=0,
            y_rotate=0,
            cbar=True,
            set_center=0,
            figsize=(10,10)):
    
    '''
    Function Which Generates a Heatmap
    
    Parameters:
        Dataframe
        column_name (list): If included, will only show certain columns on the Horizontal Axis.
    
    Returns:
        matlplot plot.
    
    '''
    
    sns.set(style='white')
    
    # View column with Abbreviated title or full. Abbreviated displays nicer.
    if correlation:
        corr = df.corr()
    else:
        corr = df.copy()
    
    if len(column_list)!=0:
        corr = corr[column_list]
    
    mask= np.zeros_like(corr,dtype=bool)
    mask[np.triu_indices_from(mask)]=True
    f,ax = plt.subplots(figsize=figsize)
    
    if len(str(set_center))!=0:
        sns.heatmap(corr,mask=mask,cmap=cmap,center=set_center,square=True,linewidths=1,annot=annotate,cbar=cbar)
    else:
        sns.heatmap(corr,mask=mask,cmap=cmap,square=True,linewidths=1,annot=annotate,cbar=cbar)
    
    
    plt.title(title)
    if y_rotate !=0:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
            tick.set_horizontalalignment('right')
    if x_rotate !=0:
        plt.xticks(rotation=x_rotate,ha='center', va='top')

    plt.show()