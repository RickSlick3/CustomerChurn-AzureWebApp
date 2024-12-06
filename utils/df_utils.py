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

def set_up(df):
    # encode labels
    mappings = {'Attrition_Flag' : {'Existing Customer': 0, 'Attrited Customer': 1},
                'Gender' : {'M': 0, 'F': 1},
                'Education_Level': {'Uneducated': 0, 'High School': 1, 'College': 2, 'Graduate': 3, 'Post-Graduate': 4, 'Doctorate': 5, 'Unknown': 6},
                'Marital_Status': {'Single': 0, 'Married': 1, 'Divorced': 2, 'Unknown': 3},
                'Income_Category' : {'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2, '$80K - $120K': 3, '$120K +': 4, 'Unknown': 5},
                'Card_Category' : {'Blue': 0, 'Silver': 1,'Gold': 2, 'Platinum': 3}}
    
    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)

    return df