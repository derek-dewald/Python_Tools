## File Description: This file should include functions related to visualization tools. Creating, manipulating, saving graphs, Images, etc. Please include all formating functions within this workbook.

from IPython.display import display, HTML
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def SimpleBar(df,
              x_axis,
              y_axis,
              title="",
              binary_split_col="",
              figsize=(10,5),
              legend=0,
              auto_scale_y='',
              export_location="",
              return_value='graph'):
    '''
    Function to simplify the creation of a Bar Chart.
    
    Note, currently the binary_split_col has not been tested.
    
    
    Parameters:
        
        
        
    Returns:
        
    

    
    '''
    
    plt.figure(figsize=figsize)
      
    if binary_split_col!="":
        plt.plot(df[x_axis],df[y_axis],label='Population',alpha=.2)
        on = df[df[binary_split_col]==1]
        off = df[df[binary_split_col]==0]
        plt.plot(on[x_axis],on[y_axis],label='Target',alpha=.8)
        plt.plot(off[x_axis],off[y_axis],label='Not Target',alpha=.5)
    
    elif isinstance(y_axis,list):
        x_pos = np.arange(len(df[x_axis]))
        bar_width = 1 / (len(y_axis) + 1)
        
        for i, axis in enumerate(y_axis):
            plt.bar(x_pos + (i - len(y_axis) // 2) * bar_width, df[axis], bar_width, label=axis)
        
        plt.xticks(x_pos, df[x_axis], rotation=45)
            
    else:
        plt.bar(df[x_axis],df[y_axis],label='Population')

    if legend==1:
        plt.legend()
    
    plt.title(title, fontsize=16, ha='center', fontweight='bold', color='black')
    
    if auto_scale_y != "":
        try:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: ConvertAxisValue(x, auto_scale_y)))
        except Exception as e:
            print(f"Error in formatting Y-axis: {e}")
    
    if len(export_location)!=0:
        plt.savefig(export_location,bbox_inches='tight',transparent=False, facecolor='white')
    
    if return_value == 'graph':
        plt.show()
    else:
        return fig
    
def Heatmap(df,
            column_list=[],
            title='Heat Map of Correlation',
            cmap='coolwarm',
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
    corr = df.corr()
    
    if len(column_list)!=0:
        corr = corr[column_list]
    
    mask= np.zeros_like(corr,dtype=bool)
    mask[np.triu_indices_from(mask)]=True
    f,ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr,mask=mask,cmap=cmap,center=0,square=True,linewidths=1,annot=True)
    
    plt.title(title)
    plt.show()

def plot_histograms(df, bins=30):
    """
    Plots histograms for each column in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    bins (int): Number of bins for the histograms (default=30).
    """
    num_columns = len(df.columns)
    num_rows = math.ceil(num_columns / 4)  # Determine rows for 4 columns
    
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows))
    axes = axes.flatten()  # Flatten in case of fewer than 4 columns

    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=bins, color="skyblue", edgecolor="black")
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_scatter_matrix(df, target_col='Target'):
    """
    Plots scatter plots of each feature against the target variable.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The name of the target variable.
    """
    num_features = df.drop(columns=[target_col]).shape[1]
    num_rows = (num_features // 4) + 1
    
    fig, axes = plt.subplots(num_rows, 4, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        if col != target_col:
            sns.scatterplot(x=df[col], y=df[target_col], ax=axes[i])
            axes[i].set_title(f"{col} vs. {target_col}")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




def ConvertAxisValue(x, scale):
    '''
    Function to Simply support Formating on Matplotlib Object. To be applied as lambda function.
    Function written by Chatgpt. 
    
    '''
    
    
    if scale == 'M':
        return f'{x * 1e-6:.1f}M'
    elif scale == 'K':
        return f'{x * 1e-3:.1f}K'  # Convert to thousands
    else:
        return f'{x:.1f}'  # Format as default with one decimal place
    







def JupyterNotebookMarkdown(df,return_value=""):
    
    '''
    Function to Create a Markdown file from Process DF, which is a data frame of the structure, 
    Title,Header,Description
    
    Args:
        Dataframe( Must be of format, Title, Header, Description)
        return_value (str: "" or text):
            If Blank, will render text in HTML Format. 
            If text, then will return text for rendering in HTML Markdown
    
    Returns:
        Conditional on Return Value. Please read Args.
    
    
    '''
    
    try:
        df1 = df[['Title','Header','Description']]
    
    except:
        
        print('DataFrame does not meet structure requirement, which must include 3 Column: Title, Header, Description')
        return ''
    
    title= ""
    step_number = 1
    text = ""

    l2_bullet = '-'  # Level 2 Bullet
    l3_bullet = '*'  # Level 3 Bullet

    for index, row in df1.iterrows():
        # Ensure previous list is closed before starting a new title
        if title and title != row.iloc[0]:  
            text += "</ul>\n"  # Close the last unordered list before switching to a new title

        # If it's a new title, start a new section
        if title == "" or title != row.iloc[0]:
            text += f"<h4>{step_number}. {row.iloc[0]}</h4>\n<ul>\n"  # Reset indentation
            step_number += 1
            title = row.iloc[0]  # Store the new title

        # Add Level 2 content (Column 2)
        if isinstance(row.iloc[1], str) and row.iloc[1].strip():
            text += f"  <li>{row.iloc[1]}</li>\n"  # L2 starts here

            # Add Level 3 content (Column 3) only if it exists
            if isinstance(row.iloc[2], str) and row.iloc[2].strip():
                text += f"    <ul><li>{row.iloc[2]}</li></ul>\n"  # L3 indented under L2

    text += "</ul>\n"  # Close any remaining lists

    if return_value =="":# Display the formatted HTML output in Jupyter Notebook
        display(HTML(text))
        
    else:
        return text



























## Always Leave Pixelate Image as Last!

def pixelate_image(image_path, PixelSizeList=[10,50,75,100,125]):
    '''
    Function to take a .jpg file and Pixelate it, Developed for Playing a "Fun" Game.
    
    Parameters:
        image_path: Location of JPG File
        PixelSizeList: List of Pixelation Levels to Apply, 1 being a near replica of the image, 200 being a near unrecognizable generalist Blob.

    Returns:
        pix
    
    '''

    import cv2

    # Load the image
    image = cv2.imread(image_path)
    
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    for pixel_size in PixelSizeList:

        # Resize to a smaller version
        small_image = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)

        # Resize back to original size (creates pixelation effect)
        pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Save or display the pixelated image
        pixelated_path = f"pixelated_image_{pixel_size}.jpg"
        cv2.imwrite(pixelated_path, pixelated_image)
    
    print(f"Pixelated image saved at: {image_path}")
    
    return pixelated_path