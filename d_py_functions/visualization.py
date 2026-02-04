import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import sys
import os
if os.getcwd().find('Users/derekdewald/Doc')!=-1:
    sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

else:
    sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\BEEM_PY\\')
    sys.path.append('K:\\INFORMATION_SYSTEMS\\Reporting and Analytics\\Derek\\d_py_functions\\')

from data_d_lists import hex_color_list

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

def visualize_hex_color(hex_color_list=hex_color_list, columns=10, rows=20, title_fontsize=8):
    """
    Purpose: Visualize a list of hex colors (or any matplotlib-compatible colors)
    as a grid of swatches with labels.

    Parameters:
        hex_color_list (list[str]): List of colors (e.g., ["#FF0000", "#00FF00"]).
        columns (int): Number of columns in the grid.
        rows (int): Number of rows in the grid.
        title_fontsize (int): Font size for hex labels.

    Returns:
        None (shows a matplotlib figure)
    """
    if not hex_color_list:
        print("No colors provided.")
        return

    total = rows * columns
    colors = list(hex_color_list)[:total]

    fig, axs = plt.subplots(rows, columns, figsize=(columns * 1.3, rows * 1.3))
    # axs is always a 2D array when rows, columns > 1; if either is 1, we normalize:
    if rows == 1 and columns == 1:
        axs = [[axs]]
    elif rows == 1:
        axs = [axs]
    elif columns == 1:
        axs = [[ax] for ax in axs]

    # Fill swatches
    for i in range(rows * columns):
        r = i // columns
        c = i % columns
        ax = axs[r][c]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        if i < len(colors):
            color = colors[i]
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
            ax.set_title(str(color), fontsize=title_fontsize)
        else:
            # Empty cells (when fewer colors than grid cells)
            ax.axis("off")

    plt.tight_layout()
    plt.show()
