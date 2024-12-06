import pandas as pd
import pyodbc as odbc
import seaborn as sns
import numpy as np
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# correltion heatmap
def corr_heatmap(df):
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={"shrink": 0.8}, linewidths=0.5)
    plt.title('Correlation Matrix')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return img

# Churn by income
def churn_by_income(df):
    income_attrition = df.groupby('Income_Category')['Attrition_Flag'].mean()
    plt.figure(figsize=(7, 3))
    plt.bar(['<40K', '40K-60K', '60K-80K', '80K-120K', '>120K', 'Unknown'], income_attrition, color='green')
    plt.title('Churn by Income Category')
    plt.xlabel('Income Category')
    plt.ylabel('Attrition Rate (Proportion)')
    plt.xticks(rotation=45)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return img

# dictionary for attrition correlation
def corr_dict(df):
    target_variable = 'Attrition_Flag'
    correlation_dict = {}

    #Fill the correlation dict
    for column in df.columns:
        if column != target_variable:  # Exclude the target variable itself
            correlation = df[column].corr(df[target_variable])
            correlation_dict[column] = correlation

    keys = list(correlation_dict.keys())
    values = list(correlation_dict.values())

    plt.figure(figsize=(15, 4))
    plt.bar(keys, values, color='skyblue')
    plt.title('Attrition Correlation')
    plt.xlabel('Columns')
    plt.ylabel('Correlation')
    plt.xticks(rotation=90)
    plt.axhline(0, color='black', linewidth=1, linestyle='-')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return img

# Attrition and gender
def gender_attrition(df):
    gender_attrition = df.groupby(['Gender', 'Attrition_Flag']).size().unstack()
    labels = ['Stayed', 'Left']

    # Plot 1: Proportion of Male customers who stayed vs. left
    plt.figure(figsize=(11, 3))

    plt.subplot(1, 3, 1)
    male_proportions = gender_attrition.loc[0] / gender_attrition.loc[0].sum()
    plt.pie(male_proportions, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange'])
    plt.title('Attrition of male customers')

    # Plot 2: Proportion of Female customers who stayed vs. left
    plt.subplot(1, 3, 2)
    female_proportions = gender_attrition.loc[1] / gender_attrition.loc[1].sum()
    plt.pie(female_proportions, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightpink', 'teal'])
    plt.title('Attrition of female customers')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return img

# marital status and attrition
def rel_status_attrition(df):
    # Group by Marital_Status and Attrition_Flag and count occurrences
    marital_attrition = df.groupby(['Marital_Status', 'Attrition_Flag']).size().unstack()

    labels = ['Stayed', 'Left']

    # Plot setup
    plt.figure(figsize=(11, 3))

    # Plot 1: Proportion of Single customers who stayed vs. left
    plt.subplot(1, 3, 1)
    single_proportions = marital_attrition.loc[0] / marital_attrition.loc[0].sum()  # Single status: 0
    plt.pie(single_proportions, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange'])
    plt.title('Attrition of Single customers')

    # Plot 2: Proportion of Married customers who stayed vs. left
    plt.subplot(1, 3, 2)
    married_proportions = marital_attrition.loc[1] / marital_attrition.loc[1].sum()  # Married status: 1
    plt.pie(married_proportions, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'red'])
    plt.title('Attrition of Married customers')

    # Plot 3: Proportion of Divorced customers who stayed vs. left
    plt.subplot(1, 3, 3)
    divorced_proportions = marital_attrition.loc[2] / marital_attrition.loc[2].sum()  # Divorced status: 2
    plt.pie(divorced_proportions, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'purple'])
    plt.title('Attrition of Divorced customers')

    # Adjust layout for better display
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return img