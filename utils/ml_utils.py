from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNC
import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.df_utils import set_up
from utils.dashboard_utils import remove_file, make_corr_dict
# from df_utils import set_up
# from dashboard_utils import remove_file, make_corr_dict

def make_df():
    df = pd.read_csv("utils/bank_churners_data.csv")
    df = df.drop(columns='Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1')
    df = df.drop(columns='Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2')
    df = set_up(df)
    return df

def logistic_regression():
    df = make_df()
    correlation_dict = make_corr_dict(df)

    scaler = StandardScaler()
    x = scaler.fit_transform(df[list(correlation_dict.keys())])
    y = df['Attrition_Flag']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LogisticRegression(C=1, max_iter=100, solver='lbfgs')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Results:')
    print(f'Accuracy: {accuracy:.2f}')
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Transaction history graph
    plt.scatter(df['Total_Trans_Ct'][y_test.index][y_test==0],df['Total_Ct_Chng_Q4_Q1'][y_test.index][y_test==0], label = 'Existing Customers', color = 'blue')
    plt.scatter(df['Total_Trans_Ct'][y_test.index][y_test==1],df['Total_Ct_Chng_Q4_Q1'][y_test.index][y_test==1], label = 'Churned Customers', color = 'red')
    plt.scatter(df['Total_Trans_Ct'][y_test.index][y_test!=y_pred],df['Total_Ct_Chng_Q4_Q1'][y_test.index][y_test!=y_pred],  label = 'Wrong Classification', color = 'yellow')
    plt.xlabel("Transactions in the last 12 months")
    plt.ylabel("Change in transaction count (Q4 over Q1)")
    plt.title('Customer Churn (Logistic Regression model)')
    plt.legend()
    # plt.scatter(x_test[y_test == 0][:, 0], x_test[y_test == 0][:, 1], label='Existing Customers', color='blue')
    # plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], label='Churned Customers', color='red')
    # plt.scatter(x_test[y_test != y_pred][:, 0], x_test[y_test != y_pred][:, 1], label='Wrong Classification', color='yellow')
    # plt.xlabel("Transactions in the last 12 months")
    # plt.ylabel("Change in transaction count (Q4 over Q1)")
    # plt.title('Customer Churn (Logistic Regression model)')
    # plt.legend()

    save_path = os.path.join("static", "images", "logistic_regression.png")
    remove_file(save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

def random_forest():
    df = make_df()
    correlation_dict = make_corr_dict(df)
    # Scale the features
    scaler = StandardScaler()
    x = scaler.fit_transform(df[list(correlation_dict.keys())])
    y = df['Attrition_Flag']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Results:')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')

    plt.scatter(df['Total_Trans_Ct'][y_test.index][y_test==0],df['Total_Ct_Chng_Q4_Q1'][y_test.index][y_test==0], label = 'Existing Customers', color = 'blue')
    plt.scatter(df['Total_Trans_Ct'][y_test.index][y_test==1],df['Total_Ct_Chng_Q4_Q1'][y_test.index][y_test==1], label = 'Churned Customers', color = 'red')
    plt.scatter(df['Total_Trans_Ct'][y_test.index][y_test!=y_pred],df['Total_Ct_Chng_Q4_Q1'][y_test.index][y_test!=y_pred],  label = 'Wrong Classification', color = 'yellow')
    plt.xlabel("Transactions in the last 12 months")
    plt.ylabel("Change in transaction count (Q4 over Q1)")
    plt.title('Customer Churn (Random Forest model)')
    plt.legend()

    save_path = os.path.join("static", "images", "random_forest.png")
    remove_file(save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

def knn_classifier():
    df = make_df()
    # df['Total_Trans_Ct'] = df['Total_Trans_Ct'].astype('float64')
    # print(df[['Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']].dtypes)
    scaler = StandardScaler()
    
    x = scaler.fit_transform(df[['Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']])
    y = df['Attrition_Flag']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    modelKNN = KNC(n_neighbors = 5)
    modelKNN.fit(x_train,y_train)
    y_pred = modelKNN.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Results:')
    print(f'Accuracy: {accuracy:.2f}')
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    plt.scatter(df['Total_Trans_Ct'][y_test.index][y_test==0],df['Total_Ct_Chng_Q4_Q1'][y_test.index][y_test==0], label = 'Existing Customers', color = 'blue')
    plt.scatter(df['Total_Trans_Ct'][y_test.index][y_test==1],df['Total_Ct_Chng_Q4_Q1'][y_test.index][y_test==1], label = 'Churned Customers', color = 'red')
    plt.scatter(df['Total_Trans_Ct'][y_test.index][y_test!=y_pred],df['Total_Ct_Chng_Q4_Q1'][y_test.index][y_test!=y_pred], label = 'Wrong Classification', color = 'yellow')
    plt.xlabel("Transactions in the last 12 months")
    plt.ylabel("Change in transaction count (Q4 over Q1)")
    plt.title('Customer Churn (KNN Model)')
    plt.legend()

    save_path = os.path.join("static", "images", "knn_classifier.png")
    remove_file(save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

if __name__ == '__main__':
    logistic_regression()
    random_forest()
    knn_classifier()