## File Description: This file should include functions related to visualization tools. Creating, manipulating, saving graphs, Images, etc. Please include all formating functions within this workbook.
emoji_dict = {
    "Status / Progress / Outcome": [
        '✅', '✔️', '⏸️', '🟢', '🟡', '🔴', '⏳', '🕐', '📤', '📥', '🗂️'
    ],
    "Data / Analytics / Reports": [
        '📈', '📉', '📊', '📋', '📝', '📌', '📍', '📚', '🔢', '📦', '🧾', '🔍', '🔎', '🧠'
    ],
    "Risk / Alerts / Flags": [
        '⚠️', '❌', '🚨', '🚩', '🛑', '❗', '❓', '⛔', '💣', '🧨'
    ],
    "Search / Insight / Discovery": [
        '🔍', '🔍', '🔍', '🔍', '🔎', '💡', '💭', '🤔', '🧐', '🔒', '🔓'
    ],
    "Tools / Engineering / Action": [
        '🛠️', '🔧', '🧰', '⚙️', '🗜️'
    ],
    "Motivation / Energy / Wins": [
        '🚀', '💥', '💥', '🔥', '💫', '💰', '💎', '🌟', '🏆', '🎯', '🏁', '🧗', '🌱'
    ],
    "Collaboration / Communication": [
        '📣', '🗣️', '👥', '🤝', '📨'
    ],
    "Experimentation / Exploration": [
        '🧪', '🧭', '🪄'
    ]
}


hex_color_list = ['#808080','#efc050','#0000cd','#060','#ff4040','#FFC0CB','#EEDFCC','#0ff','#8a3324','#6495ed','#ff7f00','#8a2be2',
                  '#050505','#841b2d','#a4c639','#cd9575','#ffbf00','#c46210','#E3CF57','#f4c2c2','#fae7b5','#ffe4c4',
                  '#fe6f5e','#bf4f51','#a57164','#ace5ee','#5d8aa8','#00308f','#72a0c1','#a32638','#f0f8ff','#e32636','#efdecd',
                  '#e52b50','#ff7e00','#ff033e','#96c','#f2f3f4','#915c83','#faebd7','#008000','#8db600','#fbceb1','#7fffd4',
                  '#4b5320','#3b444b','#e9d66b','#b2beb5','#87a96b','#f96','#a52a2a','#fdee00','#6e7f80','#568203','#007fff',
                  '#f0ffff','#89cff0','#a1caf1','#21abcd','#ffe135','#7c0a02','#848482','#98777b','#bcd4e6','#9f8170','#f5f5dc',
                  '#9c2542','#3d2b1f','#000','#3d0c02','#253529','#3b3c36','#ffebcd','#318ce7','#faf0be','#00f','#a2a2d0',
                  '#1f75fe','#69c','#0d98ba','#0093af','#0087bd','#339','#0247fe','#126180','#de5d83','#79443b',
                  '#0095b6','#e3dac9','#c00','#006a4e','#873260','#0070ff','#b5a642','#cb4154','#1dacd6','#6f0','#bf94e4',
                  '#c32148','#ff007f','#08e8de','#d19fe8','#f4bbff','#ff55a3','#fb607f','#004225','#cd7f32','#964b00','#a52a2a',
                  '#ffc1cc','#e7feff','#f0dc82','#480607','#800020','#deb887','#c50','#e97451','#808080','#bd33a4','#702963',
                  '#536872','#5f9ea0','#91a3b0','#006b3c','#ed872d','#e30022','#fff600','#a67b5b','#4b3621','#1e4d2b','#a3c1ad',
                  '#c19a6b','#efbbcc','#78866b','#ffef00','#ff0800','#e4717a','#00bfff','#592720','#c41e3a','#0c9','#960018',
                  '#d70040','#eb4c42','#ff0038','#ffa6c9','#b31b1b','#99badd','#ed9121','#062a78','#92a1cf','#ace1af',
                  '#007ba7','#2f847c','#b2ffff','#4997d0','#de3163','#ec3b83','#007ba7','#2a52be','#6d9bc3','#007aa5',
                  '#e03c31','#a0785a','#fad6a5','#36454f','#e68fac','#dfff00','#7fff00','#de3163','#ffb7c5','#cd5c5c','#de6fa1',
                  '#a8516e','#aa381e','#7b3f00','#d2691e','#ffa700','#98817b','#e34234','#d2691e','#e4d00a','#fbcce7',
                  '#0047ab','#d2691e','#6f4e37','#9bddff','#f88379','#002e63','#8c92ac','#b87333','#da8a67','#ad6f69',
                  '#cb6d51','#966','#ff3800','#ff7f50','#f88379','#808080','#893f45','#fbec5d','#b31b1b','#fff8dc',
                  '#fff8e7','#ffbcd9','#fffdd0', '#dc143c','#be0032', '#0ff','#00b7eb', '#ffff31', '#f0e130', '#00008b',
                  '#654321', '#5d3954', '#a40000', '#08457e','#986960', '#cd5b45', '#008b8b', '#536878', '#b8860b', 
                  '#a9a9a9', '#013220', '#00416a','#1a2421', '#bdb76b', '#483c32', '#734f96', '#8b008b', '#036', '#556b2f',
                  '#ff8c00','#9932cc', '#779ecb', '#03c03c', '#966fd6', '#c23b22', '#e75480','#039', '#872657', '#8b0000',
                  '#e9967a', '#560319', '#8fbc8f', '#3c1414', '#483d8b', '#2f4f4f','#177245', '#918151', '#ffa812',
                  '#483c32', '#cc4e5c', '#00ced1', '#9400d3','#9b870c', '#00703c', '#555', '#d70a53', '#a9203e', '#ef3038',
                  '#e9692c', '#da3287', '#fad6a5','#b94e48', '#704241', '#c154c1', '#004b49', '#95b', '#c0c', '#ffcba4', '#ff1493',
                  '#843f5b', '#f93', '#00bfff', '#66424d', '#1560bd', '#c19a6b', '#edc9af', '#696969','#1e90ff', '#d71868',
                  '#85bb65', '#967117', '#00009c', '#e1a95f', '#555d50', '#c2b280', '#614051', '#f0ead6', '#1cac78', 
                  '#1034a6', '#7df9ff', '#ff003f', '#0ff', '#0f0', '#6f00ff', '#f4bbff', '#cf0', '#bf00ff', '#3f00ff',
                 '#8f00ff', '#ff0', '#50c878', '#b48395', '#96c8a2', '#c19a6b', '#801818', '#b53389','#2c1608','#00FF00',
                 '#f400a1', '#e5aa70', '#4d5d53', '#4f7942', '#ff2800', '#6c541e', '#ce2029', '#b22222','#ff0', '#9acd32',
                 '#e25822', '#fc8eac', '#f7e98e', '#eedc82', '#fffaf0', '#ffbf00', '#ff1493', '#cf0','#efcc00', '#ffd300',
                 '#ff004f', '#014421', '#228b22', '#a67b5b', '#0072bb', '#86608e', '#cf0', '#c72c48','#009f6b', '#00a550', 
                 '#f64a8a', '#f0f', '#c154c1', '#f7f', '#c74375', '#e48400', '#c66','#fefe33','#0014a8','#66b032', '#adff2f',
                 '#dcdcdc', '#e49b0f', '#f8f8ff', '#b06500', '#6082b6', '#e6e8fa', '#d4af37', '#ffd700','#bebebe', '#0f0',        
                 '#a99a86', '#00ff7f', '#663854', '#446ccf', '#5218fa', '#e9d66b', '#3fff00', '#c90016','#008000', '#00a877', 
                 '#da9100', '#808000', '#df73ff', '#f400a1', '#f0fff0', '#007fbf', '#49796b', '#ff1dce',
                 '#ff69b4', '#355e3b','#71a6d2', '#fcf75e', '#002395', '#b2ec5d', '#138808','#cd5c5c', '#e3a857',
                 '#6f00ff', '#00416a', '#4b0082', '#002fa7', '#ff4f00', '#ba160c', '#c0362c', '#5a4fcf','#a8e4a0', '#465945', 
                 '#f4f0ec', '#009000', '#fffff0', '#00a86b', '#f8de7e', '#d73b3e', '#a50b5e', '#343434','#ffdf00', '#daa520', 
                 '#fada5e', '#bdda57', '#29ab87', '#4cbb17', '#7c1c05', '#c3b091', '#f0e68c', '#e8000d','#996515', '#fcc200', 
                 '#b57edc', '#c4c3d0', '#9457eb', '#ee82ee', '#e6e6fa', '#fbaed2', '#967bb6', '#fba0e3','#ccf', '#fff0f5',
                 '#e6e6fa', '#7cfc00', '#fff700', '#fffacd', '#e3ff00', '#1a1110', '#fdd5b1', '#add8e6','#a9ba9d', '#cf1020', 
                 '#b5651d', '#e66771', '#f08080', '#93ccea', '#f56991', '#e0ffff', '#f984ef', '#fafad2','#26619c', '#fefe22', 
                 '#d3d3d3', '#90ee90', '#f0e68c', '#b19cd9', '#ffb6c1', '#e97451','#ffa07a', '#f99','#087830', '#d6cadd', 
                 '#20b2aa', '#87cefa', '#789', '#b38b6d', '#e68fac', '#ffffe0', '#c8a2c8', '#bfff00','#BFD0CA',
                 '#32cd32', '#0f0', '#9dc209', '#195905', '#faf0e6', '#c19a6b', '#6ca0dc', '#534b4f',
                 '#e62020', '#f0f', '#ca1f7b', '#ff0090', '#aaf0d1', '#f8f4ff', '#c04000', '#fbec5d',
                 '#6050dc', '#0bda51', '#979aaa', '#ff8243', '#74c365', '#880085', '#c32148', '#800000',
                 '#b03060', '#e0b0ff', '#915f6d', '#ef98aa', '#73c2fb', '#e5b73b', '#6da', '#e2062c',
                 '#af4035', '#f3e5ab', '#035096', '#1c352d', '#dda0dd', '#ba55d3', '#0067a5', '#9370db',
                 '#bb3385', '#aa4069', '#3cb371', '#7b68ee', '#c9dc87', '#00fa9a', '#674c47', '#48d1cc',
                 '#79443b', '#d9603b', '#c71585', '#f8b878', '#f8de7e', '#fdbcb4', '#191970', '#004953',
                 '#ffc40c', '#3eb489', '#f5fffa', '#98ff98', '#ffe4e1', '#faebd7', '#967117', '#73a9c2', '#ae0c00', '#addfad',
                 '#30ba8f', '#997a8d', '#18453b', '#c54b8c', '#ffdb58', '#21421e', '#f6adc6', '#2a8000',
                 '#fada5e', '#ffdead', '#000080', '#ffa343', '#fe4164', '#39ff14', '#d7837f', '#a4dded',
                 '#059033', '#0077be', '#c72', '#008000', '#cfb53b', '#fdf5e6', '#796878', '#673147',
                 '#c08081', '#808000', '#3c341f', '#6b8e23', '#9ab973', '#353839', '#b784a7', '#A5B2B5',
                 '#ff9f00', '#ff4500', '#fb9902', '#ffa500', '#da70d6', '#654321', '#900', '#414a4c','#ffae42','#ffef00',
                 '#ff6e4a', '#002147', '#273be2', '#682860', '#bcd4e6', '#afeeee', '#987654', '#af4035',
                 '#9bc4e2', '#ddadaf', '#da8a67', '#abcdef', '#e6be8a', '#eee8aa', '#98fb98', '#dcd0ff',
                 '#f984e5', '#fadadd', '#dda0dd', '#db7093','#96ded1', '#c9c0bb', '#ecebbd', '#bc987e','#db7093',
                  '#78184a', '#ffefd5', '#50c878','#aec6cf', '#836953', '#cfcfc4', '#7d7','#f49ac2', '#ffb347',
                  '#dea5a4', '#b39eb5','#ff6961', '#cb99c9', '#fdfd96', '#800080','#536878', '#ffe5b4', '#ffcba4', '#fc9',
                  '#ffdab9', '#fadfad', '#d1e231', '#eae0c8','#88d8c0', '#b768a2', '#e6e200', '#ccf','#1c39bb', '#00a693',
                  '#32127a', '#d99058','#f77fbe', '#701c1c','#c33', '#fe28a2','#ec5800', '#cd853f', '#df00ff', '#000f89',
                  '#123524', '#fddde6', '#01796f', '#ffddf4', '#f96','#e7accf', '#f78fa7', '#93c572',
                  '#e5e4e2', '#8e4585', '#dda0dd', '#ff5a36','#b0e0e6', '#ff8f00','#701c1c', '#003153','#df00ff', '#c89',
                  '#ff7518', '#69359c','#800080', '#9678b6','#9f00c5', '#fe4eda','#50404d', '#a020f0','#51484f', '#5d8aa8',
                  '#ff355e', '#fbab60','#e30b5d', '#915f6d', '#e25098', '#b3446c','#826644', '#f3c', '#e3256b', '#f00',
                  '#a52a2a', '#860111', '#f2003c', '#c40233','#ff5349', '#ed1c24', '#fe2712', '#c71585','#ab4e52', '#522d80',
                  '#002387', '#004040','#f1a7fe', '#d70040', '#0892d0', '#a76bcf', '#b666d2', '#b03060', '#414833', '#0cc',
                  '#ff007f','#f9429e', '#674846', '#b76e79', '#e32636', '#f6c', '#aa98a9', '#905d5d', '#ab4e52',
                  '#65000b', '#d40000', '#bc8f8f', '#0038a8', '#002366', '#4169e1', '#ca2c92', '#7851a9',
                  '#fada5e', '#d10056', '#e0115f', '#9b111e', '#ff0028', '#bb6528', '#e18e96', '#a81c07',
                  '#80461b', '#b7410e', '#da2c43', '#00563f', '#8b4513', '#ff6700', '#f4c430','#F535AA','#808000',
                  '#ff8c69', '#ff91a4', '#c2b280', '#967117', '#ecd540', '#f4a460', '#967117', '#92000a','#507d2a',
                  '#0f52ba', '#0067a5', '#cba135', '#ff2400', '#fd0e35', '#ffd800', '#76ff7a','#9D00FF','#8E7618',
                  '#006994', '#2e8b57', '#321414', '#fff5ee', '#ffba00', '#704214', '#8a795d', '#009e60','#fc0fc0',
                  '#ff6fff', '#882d17', '#c0c0c0', '#cb410b','#007474', '#87ceeb', '#cf71af', '#6a5acd', '#708090', '#039',
                  '#933d41', '#100c08','#fffafa', '#0fc0fc', '#a7fc00', '#00ff7f', '#23297a', '#4682b4', '#fada5e', '#900',
                  '#4f666a', '#e4d96f', '#fc3', '#fad6a5', '#d2b48c', '#f94d00', '#f28500', '#fc0',
                  '#e4717a', '#483c32', '#8b8589', '#d0f0c0', '#f88379', '#f4c2c2', '#008080', '#367588',
                  '#00827f', '#cf3476', '#cd5700', '#e2725b', '#d8bfd8', '#de6fa1', '#fc89ac', '#0abab5',
                  '#e08d3c', '#dbd7d2', '#eee600', '#ff6347', '#746cc0', '#ffc87c', '#fd0e35', '#00755e', '#0073cf',
                  '#417dc1', '#deaa88', '#b57281', '#30d5c8', '#00ffef', '#a0d6b4', '#7c4848', '#8a496b',
                  '#66023c', '#03a', '#d9004c', '#8878c3', '#536895', '#ffb300', '#3cd070', '#ff6fff',
                  '#120a8f', '#4166f5', '#635147', '#ffddca', '#5b92e5', '#b78727', '#ff6', '#014421',
                  '#7b1113', '#ae2029', '#e1ad21', '#004f98', '#900', '#fc0', '#d3003f', '#f3e5ab',
                  '#c5b358', '#c80815', '#43b3ae', '#e34234', '#d9603b', '#a020f0', '#8f00ff', '#324ab2',
                  '#7f00ff', '#8601af', '#ee82ee', '#40826d', '#922724', '#9f1d35', '#da1d81', '#ffa089',
                  '#9f00ff', '#004242', '#a4f4f9', '#645452', '#f5deb3', '#fff', '#f5f5f5', '#a2add0','#9BCD9B',
                  '#ff43a4', '#fc6c85', '#722f37', '#673147', '#c9a0dc', '#c19a6b', '#738678', '#0f4d92']


from IPython.display import display, HTML
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

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
    
def JupyterNotebookMarkdown(df, return_value=""):
    '''
    Function to Create a Markdown file from Process DF, which is a data frame of the structure, 
    Title, Header, Description

    Args:
        df (DataFrame): Must include columns Title, Header, Description
        return_value (str): 
            If "", renders HTML in notebook.
            If text, returns HTML Markdown string.
    
    Returns:
        str or display: Based on return_value
    '''
    try:
        df1 = df[['Title', 'Header', 'Description']]
    except:
        print('DataFrame must include columns: Title, Header, Description')
        return ''

    text = ""
    step_number = 1
    last_title = None
    last_header = None
    open_l2 = False  # Track if L2 <ul> is open
    open_l3 = False  # Track if L3 <ul> is open

    for _, row in df1.iterrows():
        curr_title = row['Title']
        curr_header = row['Header']
        curr_description = row['Description']

        # If new Title
        if curr_title != last_title:
            if open_l3:
                text += "</ul>\n"
                open_l3 = False
            if open_l2:
                text += "</ul>\n"
                open_l2 = False
            if last_title is not None:
                text += "</ul>\n"  # Close previous title's outer <ul>

            text += f"<h4>{step_number}. {curr_title}</h4>\n<ul>\n"
            step_number += 1
            last_title = curr_title
            last_header = None  # Reset header context

        # If new Header
        if curr_header != last_header and isinstance(curr_header, str) and curr_header.strip():
            if open_l3:
                text += "</ul>\n"
                open_l3 = False
            if open_l2:
                text += "</ul>\n"
                open_l2 = False

            text += f"  <ul><li>{curr_header}</li>\n"
            open_l2 = True
            last_header = curr_header

        # If Description exists
        if isinstance(curr_description, str) and curr_description.strip():
            if not open_l3:
                text += "    <ul>\n"
                open_l3 = True
            text += f"      <li>{curr_description}</li>\n"

    # Close any open lists at the end
    if open_l3:
        text += "    </ul>\n"
    if open_l2:
        text += "  </ul>\n"
    text += "</ul>\n"

    if return_value == "":
        display(HTML(text))
    else:
        return text
def Scatter(df,
            X,
            y,
            title='Scatter Plot',
            x_axis='Feature',
            y_axis='Target'):
    
    plt.scatter(df[X], df[y], color="blue", alpha=0.5)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    plt.show()

def SimpleOverTimeGraph(df, x, y, z=None, title='', cols=4):

    '''
    
    
    '''
    

    df = df.copy()
    df[x] = pd.to_datetime(df[x])
    df = df.sort_values(by=x)

    if z:
        categories = df[z].unique()
        total = len(categories)
        rows = math.ceil(total / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()

        for idx, cat in enumerate(categories):
            ax = axes[idx]
            temp_df = df[df[z] == cat].sort_values(x)
            ax.plot(temp_df[x], temp_df[y], marker='o', linestyle='-')
            ax.set_title(f'{cat}')
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)
            
        # Hide unused subplots
        for idx in range(total, len(axes)):
            fig.delaxes(axes[idx])
        
            

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    else:
        plt.figure(figsize=(12, 6))
        plt.plot(df[x], df[y], marker='o', linestyle='-')
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(rotation=45)
        
        plt.show()

def TrainingTestPerformanceChart(train_errors,test_errors,LossType='Mean Squared Error'):
    '''
    Function to Graphically Depict Testing and Training of ML Function.

    Parameters
        training_errors (list): List of Training Errors
        test_errors (list): List of Test Errors
        LossType(str): Type of Loss which was calculated (Default Mean Squre Error)

    Returns
        Matplotlib Visualization
        
    '''

    records = len(train_errors)
    
    plt.plot(records,train_errors,label='Training Errors')
    plt.plot(records,test_errors,label='Test Errors')
    plt.xlabel('Observations')
    plt.ylabel(LossType)
    plt.legend()


def visualize_hex_color(hex_color_list,columns=15,rows=10):
    
    '''
    Purpose: Function to Review a distribution of colors from a particular list. 
    Primarily Utilizing a predetermined ordered list.
        
    Parameters:
    
    Returns:
    
    visualize_hex_color(hex_color_list[:150],15)
  
    '''
    
    total_iterations = rows*columns
    
    if len(hex_color_list) == 0:
        
        hex_color_list = hex_color_list[:total_iterations]
            
    num_rows = (total_iterations+(columns-1))//columns
    num_cols = min(columns,total_iterations)
    
    fig, axs = plt.subplots(num_rows,num_cols,figsize=(20,20))
    
    for i,hex_color in enumerate(hex_color_list):
        row = i//num_cols
        col = i% num_cols
        ax = axs[row,col] if total_iterations>1 else axs
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=hex_color))
        ax.set_title(hex_color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.show()

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



