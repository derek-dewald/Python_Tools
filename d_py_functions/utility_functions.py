'''
module_word: utility_functions
module_definition: Location for functions which do not fall within the scope of any other classification and are Simple, High Generalized, General Purpose functions.

'''
import pandas as pd
import numpy as np
import inspect

from pathlib import Path
from PIL import Image
import math
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def InspectFunction(function_name):
    print(inspect.getsource(function_name))


def extract_tables_from_db(
    db_name='Analytics.INFORMATION_SCHEMA.COLUMNS',
    table_dictionary_source= {'location':'C:\\Users\\ddewald\\desktop\\D.xlsx',"sheet_name":'Tables'},
    include_columns=True,
    exclude_schemas=['Cyder','Datamart','dbo','TEST']
    
):
      
    if include_columns:
        sql = f'''  
SELECT
    TABLE_SCHEMA,
    TABLE_NAME,
    COLUMN_NAME,
    DATA_TYPE,
    CHARACTER_MAXIMUM_LENGTH
FROM {db_name}
ORDER BY
    TABLE_SCHEMA,
    TABLE_NAME,
    ORDINAL_POSITION'''
        
    else:
        sql = '''
SELECT TABLE_SCHEMA, TABLE_NAME
FROM Analytics.INFORMATION_SCHEMA.TABLES
ORDER BY TABLE_SCHEMA, TABLE_NAME'''
        
    df = TIME_SQL(sql)
    
    try:
        dd = pd.read_excel(table_dictionary_source['location'],sheet_name=table_dictionary_source['sheet_name'])
        dd = dd.rename(columns={'Table':"TABLE_NAME",'Schema':'TABLE_SCHEMA'})
        
        hist_df = dd[dd['Historical']==1].copy()
        hist_df['TABLE_SCHEMA'] = 'HISTORICAL'
        dd = pd.concat([dd,hist_df])
        df = df.merge(dd,on=['TABLE_NAME','TABLE_SCHEMA'],how='left')
        
    except:
        pass
        
    df = df[~df['TABLE_SCHEMA'].isin(exclude_schemas)]
        
    return df


def gpt_question(list_):
    text = f"Can you please provide a definition for the following word(s) in my list: {list_}. Can each definition be a single paragraph of 4-5 sentences. In the sentence please include a technical definition, when it is relevant, why is it important and where possible information about the origin. My definitions lean toward a Data Science application, when in doubt please assume it is related to Data Science."
    print(text)



def net_present_value_calculator(
    annual_interest=None,
    payment=None,
    total_amount=None,
    payments=None,
    annual_payments=12
):
    """
    Calculate the missing variable in a fixed-payment annuity.

    Exactly one parameter must be None.

    Supports scalars, NumPy arrays, and pandas Series.

    annual_interest : Nominal annual interest rate (0.05 = 5%)
    payment         : Payment amount per period
    total_amount    : Present value / loan principal
    payments        : Total number of payments
    annual_payments : Number of payments per year

    Assumes payments occur at the end of each period.
    """

    params = {
        'annual_interest': annual_interest,
        'payment': payment,
        'total_amount': total_amount,
        'payments': payments,
        'annual_payments': annual_payments
    }

    missing = [k for k, v in params.items() if v is None]

    if len(missing) != 1:
        raise ValueError("Exactly one parameter must be None.")

    missing = missing[0]

    # Preserve pandas index when Series are provided
    series_inputs = [
        v for v in params.values()
        if isinstance(v, pd.Series)
    ]

    index = series_inputs[0].index if series_inputs else None

    if series_inputs:
        for s in series_inputs[1:]:
            if not s.index.equals(index):
                raise ValueError("Series inputs must have matching indexes.")

    # Convert inputs to NumPy arrays
    inputs = {
        k: np.asarray(v, dtype=float)
        for k, v in params.items()
        if v is not None
    }

    arrays = np.broadcast_arrays(*inputs.values())

    inputs = dict(zip(inputs.keys(), arrays))

    shape = arrays[0].shape

    def result_output(result):
        result = np.asarray(result)

        if result.ndim == 0:
            return result.item()

        if index is not None:
            if result.ndim != 1 or len(result) != len(index):
                raise ValueError("Result shape does not match Series index.")
            return pd.Series(result, index=index)

        return result

    def annuity_factor(rate, n):
        rate, n = np.broadcast_arrays(rate, n)

        if np.any(rate <= -1):
            raise ValueError("Periodic interest rate must exceed -100%.")

        result = np.empty(rate.shape, dtype=float)

        zero = np.abs(rate) < 1e-12

        result[zero] = n[zero]

        result[~zero] = (
            -np.expm1(
                -n[~zero] * np.log1p(rate[~zero])
            ) / rate[~zero]
        )

        return result

    # -----------------------------------------
    # Solve for annual interest
    # -----------------------------------------

    if missing == 'annual_interest':

        pmt = inputs['payment']
        principal = inputs['total_amount']
        n = inputs['payments']
        freq = inputs['annual_payments']

        if np.any(pmt <= 0) or np.any(principal <= 0) or np.any(n <= 0):
            raise ValueError(
                "Payment, total_amount and payments must be positive."
            )

        if np.any(freq <= 0):
            raise ValueError("annual_payments must be positive.")

        target = principal / pmt

        if np.any(target > n + 1e-10):
            raise ValueError(
                "These inputs require a negative interest rate."
            )

        result = np.zeros(shape, dtype=float)

        mask = ~np.isclose(target, n, rtol=1e-12, atol=1e-12)

        for idx in np.ndindex(shape):

            if not mask[idx]:
                continue

            periodic_rate = brentq(
                lambda r: (
                    -math.expm1(-n[idx] * math.log1p(r)) / r
                    - target[idx]
                ),
                1e-12,
                max(1.0, pmt[idx] / principal[idx])
            )

            result[idx] = periodic_rate * freq[idx]

        return result_output(result)

    # -----------------------------------------
    # Validate interest rate
    # -----------------------------------------

    interest = inputs['annual_interest']

    if np.any(interest < 0):
        raise ValueError("Annual interest must be nonnegative.")

    # -----------------------------------------
    # Solve for annual payment frequency
    # -----------------------------------------

    if missing == 'annual_payments':

        pmt = inputs['payment']
        principal = inputs['total_amount']
        n = inputs['payments']

        if np.any(interest == 0):
            raise ValueError(
                "Cannot uniquely determine payment frequency at zero interest."
            )

        if np.any(pmt <= 0) or np.any(principal <= 0) or np.any(n <= 0):
            raise ValueError(
                "Payment, total_amount and payments must be positive."
            )

        target = principal / pmt

        if np.any(target <= 0) or np.any(target >= n):
            raise ValueError(
                "No unique positive payment frequency satisfies these inputs."
            )

        result = np.zeros(shape, dtype=float)

        for idx in np.ndindex(shape):

            periodic_rate = brentq(
                lambda r: (
                    -math.expm1(-n[idx] * math.log1p(r)) / r
                    - target[idx]
                ),
                1e-12,
                max(1.0, pmt[idx] / principal[idx])
            )

            result[idx] = interest[idx] / periodic_rate

        return result_output(result)

    freq = inputs['annual_payments']

    if np.any(freq <= 0):
        raise ValueError("annual_payments must be positive.")

    rate = interest / freq

    # -----------------------------------------
    # Solve for total amount
    # -----------------------------------------

    if missing == 'total_amount':

        pmt = inputs['payment']
        n = inputs['payments']

        if np.any(n <= 0):
            raise ValueError("payments must be positive.")

        result = pmt * annuity_factor(rate, n)

        return result_output(result)

    # -----------------------------------------
    # Solve for payment
    # -----------------------------------------

    if missing == 'payment':

        principal = inputs['total_amount']
        n = inputs['payments']

        if np.any(n <= 0):
            raise ValueError("payments must be positive.")

        factor = annuity_factor(rate, n)

        result = principal / factor

        return result_output(result)

    # -----------------------------------------
    # Solve for number of payments
    # -----------------------------------------

    if missing == 'payments':

        pmt = inputs['payment']
        principal = inputs['total_amount']

        if np.any(pmt <= 0) or np.any(principal <= 0):
            raise ValueError(
                "Payment and total_amount must be positive."
            )

        if np.any(pmt <= principal * rate):
            raise ValueError(
                "Payment is insufficient to repay the principal."
            )

        result = np.divide(
            principal,
            pmt,
            out=np.zeros(shape, dtype=float),
            where=pmt != 0
        )

        nonzero = rate != 0

        result[nonzero] = (
            -np.log1p(
                -principal[nonzero]
                * rate[nonzero]
                / pmt[nonzero]
            )
            / np.log1p(rate[nonzero])
        )

        return result_output(result)


def preview_images_in_n_by_n_matrix(
    folder_location,
    columns=5,
    image_size=3,
    show_filename=True
):
    """

    Definition:
        Display all PNG images in a directory as a gallery.
    Parameters:
        folder_location(str): Directory containing PNG images.
        columns(int): Number of columns in the gallery.
        image_size(int): Size of each gallery cell in inches.
        show_filename(bool): Whether to display filenames beneath images.

    Returns:
        Rendered
    Date Created:
        04-Nov-26
    Date Last Modified:
        04-Nov-26
    Process:
        TBD
    Categorization:
        TBD
    Usage:
        TBD
    Notes:
        None
    Required Functions:
        None
    
    
    """

    folder = Path(folder_location)

    image_paths = sorted(
        folder.glob("*.png"),
        key=lambda p: p.name.lower()
    )

    if not image_paths:
        print("No PNG images found.")
        return []

    n_images = len(image_paths)
    rows = math.ceil(n_images / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * image_size, rows * image_size),
        squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for i, image_path in enumerate(image_paths):

        ax = axes.flat[i]

        with Image.open(image_path) as img:
            ax.imshow(img.convert("RGBA"))

        if show_filename:
            ax.set_title(
                f"{i + 1}. {image_path.stem}",
                fontsize=9,
                pad=8
            )

    plt.tight_layout()
    plt.show()