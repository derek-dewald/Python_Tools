def ConvertListtoSQLText(list_, 
                         return_value=None, 
                         column_name=None,
                         sql_query=None):
    """
    Converts a Python list into SQL-compatible text:
    - return_value='INT': returns a comma-separated list of quoted ints
    - return_value='CTE': returns a SQL CTE string using VALUES clause
    """
    if return_value == 'INT':
        return ', '.join(f"'{int(x)}'" for x in list_)

    elif return_value == 'CTE':
        
        if not sql_query:
            sql_query = 'select * from CTE_TABLE'
        column_data  = " UNION ALL\n".join(f"SELECT {int(x)}" for x in list_)
        sql1 = f""" WITH CTE_TABLE ({column_name}) AS ( {column_data} ) {sql_query}"""
        
        return TIME_SQL(sql1)
        
    else:
        # String version
        return ', '.join(f"'{str(x)}'" for x in list_)


def get_varchar_bucket(length):
    '''
    Function Used for generate_create_table_sql to round VARCHAR VALUES
    '''
    thresholds = [8, 16, 32, 64, 128, 255]
    for t in thresholds:
        if length <= t:
            return t
    return 255 


def generate_create_table_sql(df, table_name, schema, db='ANALYTICS'):
    """
    Function to create a SQL Statement to Create a New Table.
    Function Utilizes a review of the DataFrame to recommend Column Names, Formating and appropriate Column Sizing.
    
    Function is NOT BEYOND REPROACH, requires some manual review and should not be automated.
    

    Parameters:
    df(df): Any DataFrame
    table_name(str): desired name for the SQL table
    schema(str): Target Schema Name
    db(str): Database Name (function designed for Analytics, but can be applied to any MS SQL)

    Returns:
    - str: SQL CREATE TABLE statement
    
    """

    type_mapping = {
        "int64": "BIGINT",
        "int32": "INT",
        "float64": "DECIMAL(16,2)",
        "float32": "DECIMAL(16,2)",
        "bool": "VARCHAR(5)",
        "datetime64[ns]": "DATE",
        "object": "VARCHAR"
    }

    max_len = max(len(col) for col in df.columns)

    column_defs = []
    for col in df.columns:
        dtype = str(df[col].dtype)

        if col == "MEMBERNBR":
            sql_type = 'BIGINT'
        elif col == "ACCTNBR":
            sql_type = 'BIGINT'
        elif col.lower().find('date')!=-1:
            sql_type = 'DATE'
        elif col.lower().find('flag')!=-1:
            if df[col].max()>1:
                sql_type = 'SMALLINT'
            else:
                sql_type = 'BIT'
        
        elif dtype == 'object':
            max_str_len = df[col].astype(str).map(len).max()
            varchar_len = get_varchar_bucket(min(max_str_len + 10, 255))
            sql_type = f"VARCHAR({varchar_len})"
        else:
            sql_type = type_mapping.get(dtype, "VARCHAR(100)")  # fallback for unexpected types

        # Use [col] instead of 'col' for SQL Server compatibility
        column_defs.append(f"    [{col}] {sql_type}")

    column_sql = ",\n".join(column_defs)
    full_table_name = f"[{db}].[{schema}].[{table_name}]"
    create_stmt = (
        f"CREATE TABLE {full_table_name} (\n"
        f"{column_sql}\n);\n"
    )

    print(create_stmt)
    