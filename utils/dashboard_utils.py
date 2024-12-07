import pandas as pd
import seaborn as sns
import numpy as np
# import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def remove_file(save_path):
    if os.path.exists(save_path):
        os.remove(save_path)

def make_corr_dict(df):
    target_variable = 'Attrition_Flag'
    correlation_dict = {}

    #Fill the correlation dict
    for column in df.columns:
        if column != target_variable:  # Exclude the target variable itself
            correlation = df[column].corr(df[target_variable])
            correlation_dict[column] = correlation

    return correlation_dict

# correltion heatmap
def corr_heatmap(df):
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={"shrink": 0.8}, linewidths=0.5)
    plt.title('Correlation Matrix')

    save_path = os.path.join("static", "images", "corr_heatmap.png")
    # remove_file(save_path)
    if not os.path.exists(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Churn by income
def churn_by_income(df):
    income_attrition = df.groupby('Income_Category')['Attrition_Flag'].mean()
    plt.figure(figsize=(6, 3))
    plt.bar(['<40K', '40K-60K', '60K-80K', '80K-120K', '>120K', 'Unknown'], income_attrition, color='green')
    plt.title('Churn by Income Category')
    plt.xlabel('Income Category')
    plt.ylabel('Attrition Rate (Proportion)')
    plt.xticks(rotation=45)

    save_path = os.path.join("static", "images", "income.png")
    # remove_file(save_path)
    if not os.path.exists(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# dictionary for attrition correlation
def corr_dict(df):
    correlation_dict = make_corr_dict(df)

    correlation_df = pd.DataFrame(list(correlation_dict.items()), columns=['Feature', 'Correlation'])

    # Sort by absolute correlation value in descending order
    correlation_df['Abs_Correlation'] = correlation_df['Correlation'].abs()
    correlation_df = correlation_df.sort_values('Abs_Correlation', ascending=False)

    # Create a new column for color based on the sign of the correlation
    correlation_df['Color'] = correlation_df['Correlation'].apply(lambda x: 'blue' if x > 0 else 'red')
    plt.figure(figsize=(11, 5))

    # Create horizontal bars for absolute correlations
    bars = plt.barh(correlation_df['Feature'], correlation_df['Abs_Correlation'], color=correlation_df['Color'])
    plt.xlabel('Correlation with Attrition_Flag (Magnitude)')
    plt.title('Feature Correlation with Attrition_Flag')

    # Add values on the bars
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.2f}',
                va='center', ha='left', color='green')

    # Add a legend to explain the colors
    import matplotlib.lines as mlines
    blue_patch = mlines.Line2D([], [], color='blue', label='Positive Correlation')
    red_patch = mlines.Line2D([], [], color='red', label='Negative Correlation')
    plt.legend(handles=[blue_patch, red_patch])
    plt.tight_layout()

    save_path = os.path.join("static", "images", "corr_dict.png")
    # remove_file(save_path)
    if not os.path.exists(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Attrition and gender
def gender_attrition(df):
    gender_attrition = df.groupby(['Gender', 'Attrition_Flag']).size().unstack()
    labels = ['Stayed', 'Left']

    # Plot 1: Proportion of Male customers who stayed vs. left
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    male_proportions = gender_attrition.loc[0] / gender_attrition.loc[0].sum()
    plt.pie(male_proportions, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange'])
    plt.title('Attrition of male customers')

    # Plot 2: Proportion of Female customers who stayed vs. left
    plt.subplot(1, 3, 2)
    female_proportions = gender_attrition.loc[1] / gender_attrition.loc[1].sum()
    plt.pie(female_proportions, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightpink', 'teal'])
    plt.title('Attrition of female customers')

    save_path = os.path.join("static", "images", "gender_attrition.png")
    # remove_file(save_path)
    if not os.path.exists(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# marital status and attrition
def rel_status_attrition(df):
    # Group by Marital_Status and Attrition_Flag and count occurrences
    marital_attrition = df.groupby(['Marital_Status', 'Attrition_Flag']).size().unstack()

    labels = ['Stayed', 'Left']

    # Plot setup
    plt.figure(figsize=(13, 3))

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
    plt.tight_layout()

    save_path = os.path.join("static", "images", "rel_status_attrition.png")
    # remove_file(save_path)
    if not os.path.exists(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def edu_attrition(df):
    # Group by Education_Level and count total customers
    education_totals = df.groupby('Education_Level').size()

    # Group by Education_Level and Attrition_Flag, filter for customers who left (Attrition_Flag = 1)
    education_left = df[df['Attrition_Flag'] == 1].groupby('Education_Level').size()

    # Calculate the percentage of customers who left for each education level
    education_left_percentage = (education_left / education_totals) * 100

    # Ensure alignment with all education levels (fill missing values with 0)
    education_left_percentage = education_left_percentage.reindex(education_totals.index, fill_value=0)

    # Set labels for the education levels
    labels = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', 'Unknown']

    # Plot setup
    plt.figure(figsize=(11, 4))

    # Plot 1: Percentage of total customers by education level
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(labels, (education_totals / education_totals.sum()) * 100, color='skyblue')
    plt.title('Percentage of Total Customers by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Percentage of Total Customers')
    plt.xticks(rotation=45)

    # Annotate bars with the number of total customers (rotated text within the bar)
    for bar, total in zip(bars1, education_totals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 1, f'{total}',
                ha='center', va='top', color='black', fontsize=9)

    # Plot 2: Percentage of customers who left by education level
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(labels, education_left_percentage, color='orange')
    plt.title('Percentage of Customers Who Left by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Percentage of Customers Who Left')
    plt.xticks(rotation=45)

    # Annotate bars with the number of customers who left (rotated text within the bar)
    for bar, left in zip(bars2, education_left.reindex(education_totals.index, fill_value=0)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 1, f'{int(left)}',
                ha='center', va='top', color='black', fontsize=9)

    plt.figtext(0.5, -0.02, "Note: Raw numbers are displayed inside the bars.",
                ha='center', fontsize=10, color='gray')
    plt.tight_layout()

    save_path = os.path.join("static", "images", "edu_attrition.png")
    # remove_file(save_path)
    if not os.path.exists(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Define a function for moving average
def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1, center=True).mean()

def age_plot(df):
    age_totals = df.groupby('Customer_Age').size()
    age_left_totals = df[df['Attrition_Flag'] == 1].groupby('Customer_Age').size()
    age_left_percentage = (age_left_totals / age_totals * 100).fillna(0)

    # Apply moving average to smooth the data
    age_totals_smoothed = moving_average(age_totals, window_size=3)
    age_left_percentage_smoothed = moving_average(age_left_percentage, window_size=3)

    plt.figure(figsize=(11, 5))

    # Plot 1: Smoothed total number of customers by age
    plt.subplot(1, 2, 1)
    plt.plot(age_totals_smoothed.index, age_totals_smoothed.values, marker='o', linestyle='-', color='skyblue')
    plt.title('Total Customers by Age (Smoothed)')
    plt.xlabel('Age')
    plt.ylabel('Number of Customers')
    plt.grid(True)

    # Plot 2: Smoothed percentage of customers who have left by age
    plt.subplot(1, 2, 2)
    plt.plot(age_left_percentage_smoothed.index, age_left_percentage_smoothed.values, marker='o', linestyle='-', color='orange')
    plt.title('Percentage of Customers Who Left by Age (Smoothed)')
    plt.xlabel('Age')
    plt.ylabel('Percentage of Customers Who Left')
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join("static", "images", "age_plot.png")
    # remove_file(save_path)
    if not os.path.exists(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def transactions_plot(df):
    save_path = os.path.join("static", "images", "transactions_plot.png")
    # remove_file(save_path)
    if not os.path.exists(save_path):
        # Filter the data based on Attrition_Flag
        existing_customers = df[df['Attrition_Flag'] == 0]
        churned_customers = df[df['Attrition_Flag'] == 1]

        # Create the scatter plot
        plt.figure(figsize=(11, 5))

        # Scatter plot for existing customers (Attrition_Flag == 0)
        plt.scatter(existing_customers['Total_Trans_Ct'], existing_customers['Total_Trans_Amt'],
                    color='blue', label='Existing Customers', alpha=0.6)

        # Scatter plot for churned customers (Attrition_Flag == 1)
        plt.scatter(churned_customers['Total_Trans_Ct'], churned_customers['Total_Trans_Amt'],
                    color='red', label='Churned Customers', alpha=0.6)

        plt.xlabel('Total Transactions Count')
        plt.ylabel('Total Transaction Amount')
        plt.title('Number of transactions vs Amount Transacted')
        plt.legend()
        plt.tight_layout()

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

# def credit_plot(df):
#     save_path = os.path.join("static", "images", "credit_plot.png")
#     # remove_file(save_path)
#     if not os.path.exists(save_path):
#         # Define bins for Total Revolving Balance
#         bins = np.arange(0, df['Total_Revolving_Bal'].max() + 100, 100)
#         bin_labels = [f'{i}-{i + 99}' for i in range(0, df['Total_Revolving_Bal'].max(), 100)]

#         # Create a new column for Total Revolving Balance bins
#         df['Revolving_Balance_Group'] = pd.cut(df['Total_Revolving_Bal'], bins=bins, labels=bin_labels, right=False)

#         # Group by the Revolving Balance bins and calculate the percentage of churned customers in each bin
#         churned_percentage = df.groupby('Revolving_Balance_Group').apply(
#             lambda x: (x['Attrition_Flag'].sum() / len(x)) * 100
#         ).reset_index(name='Churn_Percentage')

#         # Plotting
#         plt.figure(figsize=(11, 4))
#         plt.bar(churned_percentage['Revolving_Balance_Group'], churned_percentage['Churn_Percentage'], color='skyblue')
#         plt.xlabel('Total Revolving Balance')
#         plt.ylabel('Percentage of Churned Customers')
#         plt.title('Percentage of Churned Customers vs Unpayed Credit')
#         plt.figtext(0.5, -0.02, "Note: Total Revolving Balance is the amount of credit has not yet been payed.",
#                     ha='center', fontsize=10, color='gray')
#         plt.xticks(rotation=45, fontsize=8)
#         plt.tight_layout()

#         plt.savefig(save_path, bbox_inches='tight')
#         plt.close()