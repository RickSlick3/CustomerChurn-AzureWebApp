import pyodbc as odbc
import pandas as pd
from flask import Flask, render_template

# hide connection string

conn = odbc.connect(connection_string)

sql = '''SELECT * FROM BankChurners'''
cursor = conn.cursor()
cursor.execute(sql)
dataset = cursor.fetchall()

df = pd.read_sql("SELECT * FROM BankChurners", conn)

# columns = [col[0] for col in cursor.description]
# df = pd.DataFrame(dataset, columns=columns)
# print(df)

csv_file_path = 'bank_churners_data.csv'  # Specify your file path here
df.to_csv(csv_file_path, index=False)

app = Flask(__name__)

@app.route('/')
def display_csv():
    table_html = df.to_html(classes="table table-bordered", index=False)  # 'table table-bordered' adds Bootstrap styling
    
    # Render the HTML template and pass the table_html
    return render_template('index.html', table_html=table_html)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
