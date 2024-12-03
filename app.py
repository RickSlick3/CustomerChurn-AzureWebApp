from flask import Flask, render_template
from dotenv import load_dotenv

import pyodbc as odbc
import pandas as pd
import os

# protect connection string in .env
load_dotenv()

app = Flask(__name__)

app.config['CONNECTION_STRING'] = os.getenv('CONNECTION_STRING')

conn = odbc.connect(app.config['CONNECTION_STRING'])

sql = '''SELECT * FROM BankChurners'''
cursor = conn.cursor()
cursor.execute(sql)
dataset = cursor.fetchall()

df = pd.read_sql("SELECT * FROM BankChurners", conn)
csv_file_path = 'bank_churners_data.csv'
df.to_csv(csv_file_path, index=False)


# Routes
@app.route('/')
def display_csv():
    table_html = df.to_html(classes="table table-bordered", index=False)  # 'table table-bordered' adds Bootstrap styling
    
    # Render the HTML template and pass the table_html
    return render_template('index.html', table_html=table_html)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=os.getenv('FLASK_ENV') == 'development')