def ConvertListtoSQLText(list_, 
                         return_value=None, 
                         column_name=None,
                         sql_query=None):
    """
    Function to convert a python list into SQL code, of various formatinng. 

    If return_value is CTE, then Python List will be turning into a SQL statement, with Column Name of COLUMN_NAME, and then a SQL query to merge various 
    tables from whatever the statement is.


    Parameters:
            List_(list): Python List
            return_vaule(str): Indicator to control If statements in Function, default to return a raw list.
            column_name (str): In combination with Return Value, name of column in SQL Table Creation.
            sql_query: Statement after the CTE table statement.
    

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
        "float64": "FLOAT",
        "float32": "FLOAT",
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
        f"GO\n"
        f"GRANT SELECT, UPDATE, DELETE ON {full_table_name} TO [Python_User];"
    )

    print(create_stmt)

def TableRecordCountByDate(table_dict,
                           end_date=None,
                           total_days=15):
    
    """
    Generate a SQL Server query to count records by day for each table and pivot them.
    
    Parameters:
        dates (List[datetime.datetime]): List of dates to pivot on.
        table_dict (Dict[str, str]): Mapping of table name -> date column name.
        
    Returns:
        str: A full SQL query string.
    """
    
    if not end_date:
        end_date = datetime.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    
    start_date = end_date - datetime.timedelta(days=total_days)
    
    dates = generate_day_list(start_date=start_date,end_date=end_date)
    
    min_date = dates[0].strftime('%Y-%m-%d')
    max_date = dates[-1].strftime('%Y-%m-%d')

    # Generate the UNION ALL query block
    union_parts = []
    for table, date_column in table_dict.items():
        block = f"""
        SELECT 
            '{table}' AS TableName,
            CAST({date_column} AS DATE) AS CreatedDate,
            COUNT(*) AS Cnt
        FROM {table}
        WHERE {date_column} BETWEEN '{min_date}' AND '{max_date}'
        GROUP BY CAST({date_column} AS DATE)
        """
        union_parts.append(block.strip())

    union_query = "\n    UNION ALL\n    ".join(union_parts)

    # Format dates as [YYYY-MM-DD] for pivot columns
    pivot_columns = ", ".join(f"[{d.strftime('%Y-%m-%d')}]" for d in dates)

    # Combine full query
    full_query = f"""
    SELECT TableName, {pivot_columns}
    FROM (
        {union_query}
    ) AS SourceData
    PIVOT (
        SUM(Cnt)
        FOR CreatedDate IN ({pivot_columns})
    ) AS PivotResult
    ORDER BY TableName;
    """.strip()

    return full_query