
default_code = {
    'Jupter' = {'Fixed Decimals':"'pd.options.display.float_format = '{:.2f}'.format	Fixed Decimal Points'}"}
    }

duration_bins = {
    30: '1) Less than 30D',
    365: '2) Greater than 30D less Than 1Y',
    1095: '3) Greater than 1Y less than 3Y',
    1825: '4) Greater than 3Y less than 5Y',
    3650: '5) Greater than 5Y less than 10Y',
    7300: '6) Greater than 10Y less than 20Y',
    float('inf'): '7) Greater than 20Y'
}

python_exceptions = {
    "Exception": "Base class for all built-in exceptions.",
    "ValueError": "Raised when a function receives an argument of the correct type but inappropriate value.",
    "TypeError": "Raised when an operation is applied to an object of inappropriate type.",
    "KeyError": "Raised when a dictionary key is not found.",
    "IndexError": "Raised when a sequence index is out of range.",
    "AttributeError": "Raised when an attribute reference or assignment fails.",
    "NameError": "Raised when a variable name is not found.",
    "ImportError": "Raised when an import fails.",
    "ModuleNotFoundError": "Raised when a module cannot be found.",
    "RuntimeError": "Raised when an error is detected that doesn't fall into other categories.",
    "ZeroDivisionError": "Raised when dividing by zero.",
    "OverflowError": "Raised when a calculation exceeds limits.",
    "FloatingPointError": "Raised when a floating point operation fails.",
    "FileNotFoundError": "Raised when a file or directory is requested but doesn’t exist.",
    "PermissionError": "Raised when trying to open a file without the right permissions.",
    "IOError": "Raised when an I/O operation fails (mostly replaced by OSError).",
    "AssertionError": "Raised when an assert statement fails.",
    "NotImplementedError": "Raised when an abstract method is not implemented.",
    "StopIteration": "Raised to signal the end of an iterator.",
    "StopAsyncIteration": "Raised to signal the end of an asynchronous iterator.",
    "SystemExit": "Raised by sys.exit().",
    "KeyboardInterrupt": "Raised when the user hits the interrupt key (Ctrl+C).",
    "MemoryError": "Raised when an operation runs out of memory.",
}

four_horsemen = {'MEMBER_DURATION':'ALL',
            'LOB':"ALL",
            'ENTITY':"ALL",
            'BRANCHNAME':'ALL'}