import datetime
today = datetime.datetime.now().strftime('%d-%b-%y')
# Example of Text String Required to Populate A D Function.

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


google_mapping_sheet_csv = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSwDznLz-GKgWFT1uN0XZYm3bsos899I9MS-pSvEoDC-Cjqo9CWeEuSdjitxjqzF3O39LmjJB0_Fg-B/pub?output=csv'