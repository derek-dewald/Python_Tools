import datetime
today = datetime.datetime.now().strftime('%d-%b-%y')
# Example of Text String Required to Populate A D Function.

def gpt_question(list_,word_list=None):

    for word in list_:
        if not word_list:
            word_list = word
        else:
            word_list += f', {word}'
    text = f"""For the following words, can you please provide me for each word a definition which is 4-5 sentences highlighting the definition,\nits origin, its importance and application. If the word is a ML Model, can you also please define the following in point form,\ntype of ML learning, type of ML problem, size of data set it works best on, and anything else which might be appropriate.\nUnique Words: {word_list}
    """

    print(text)
template_doc_string = f'''

    Definition of Function

    Parameters:
        List of Parameters

    Returns:
        Object Type

    date_created:{today}
    date_last_modified: {today}
    classification:TBD
    sub_classification:TBD
    usage:
        Example Function Call
'''

def template_doc_string_print(text_=template_doc_string):
    '''

    Function to Return the Printed Value of the Doc String template_doc_string. Easier to Copy and Paste in this format opposed ot raw text.

    Parameters:
        text(str): String Value to be Printed, defaulted to template_doc_string
        
    Returns:
        string

    date_created:17-Dec-25
    date_last_modified: 17-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        template_doc_string_print()
'''
    print(text_)


def notebook_break():
    print('#############################################################################################################################################')