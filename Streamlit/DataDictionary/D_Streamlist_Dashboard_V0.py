import streamlit as st
import requests


# This Function is manually produced here as by Default Git Doesnt know where to Look, and this helps it. Also Saved in connections.

def read_git_folder(owner='derek-dewald',repo='Python_Tools',branch='main',folder='d_py_functions'):

    '''

    Program to Extract .py files from a Git Directory.
    Parameters borrowed from Git Mapping Structure, https://github.com/owner/repo/tree/branch/folder

    Parameters:
        owner(str):  As defined in Git Mapping Structure 
        repo(str): As defined in Git Mapping Structure 
        branch(str): As defined in Git Mapping Structure 
        folder(str): As defined in Git Mapping Structure 

    Returns:
        list

    date_created:15-Dec-25
    date_last_modified: 15-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        py_files_in_git_folder = read_git_folder()

    '''
    
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder}?ref={branch}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        items = resp.json()
    except Exception as e:
        print(f'Failed to read folder {url}')
        return []
        
    py_files = [
        {
            "name": item["name"],
            "download_url": item["download_url"]
        }
        for item in items
        if item.get("type") == "file" and item["name"].endswith(".py")
    ]
    return py_files

git_files = read_git_folder()

# Create Validation to See if read_git_folder() is working.
st.set_page_config(page_title="GitHub .py Browser", layout="wide")
st.title("GitHub Folder: .py File List")
#st.write(git_files)

import os
st.write(os.getcwd())

