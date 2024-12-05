import pandas as pd
import pyodbc as odbc

# Uses the connection string to connect to the database and return the data
def create_df_from_db(connection_string):
    try:
        with odbc.connect(connection_string) as conn:
            sql = '''SELECT * FROM BankChurners'''
            # cursor = conn.cursor()
            # cursor.execute(sql)
            # dataset = cursor.fetchall()
            df = pd.read_sql(sql, conn)
            return df
    except Exception as e:
        raise RuntimeError(f"Error accessing database: {e}")

# Turns a df into a csv
def create_csv(df, csv_file_path):
    try:
        df.to_csv(csv_file_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Error writing CSV: {e}")

# Filters the DataFrame by the given client number
def filter_by_clientnum(df, clientnum):
    if clientnum:
        clientnum = clientnum.strip().lower()
        return df[df["CLIENTNUM"].astype(str).str.lower().str.contains(clientnum)]
    return df

# Sorts the DataFrame by the specified column name if it exists
def sort_by_column(df, column_name):
    if column_name in df.columns:
        return df.sort_values(by=column_name)
    return df