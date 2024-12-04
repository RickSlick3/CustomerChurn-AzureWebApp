import pandas as pd
import pyodbc as odbc

def create_df_from_db(connection_string):
    try:
        with odbc.connect(connection_string) as conn:
            sql = '''SELECT * FROM BankChurners'''
            # cursor = conn.cursor()
            # cursor.execute(sql)
            # dataset = cursor.fetchall()
            df = pd.read_sql("SELECT * FROM BankChurners", conn)
            return df
    except Exception as e:
        raise RuntimeError(f"Error accessing database: {e}")

def create_csv(df, csv_file_path):
    try:
        df.to_csv(csv_file_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Error writing CSV: {e}")