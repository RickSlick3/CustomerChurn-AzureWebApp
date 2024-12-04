from flask import Flask, render_template
from dotenv import load_dotenv
import os
from utils.df_utils import *

# get .env
load_dotenv()

app = Flask(__name__)

# get connectin string from .env
app.config['CONNECTION_STRING'] = os.getenv('CONNECTION_STRING')
if not app.config['CONNECTION_STRING']:
    raise ValueError("CONNECTION_STRING is not set in the .env file.")

df = create_df_from_db(app.config['CONNECTION_STRING'])

csv_file_path = 'bank_churners_data.csv'
create_csv(df, csv_file_path)


# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sample')
def display_csv():
    sample_df = df.head(100)
    table_html = sample_df.to_html(classes="table table-bordered", index=False)  # 'table table-bordered' adds Bootstrap styling
    
    # Render the HTML template and pass the table_html
    return render_template('sample.html', table_html=table_html)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=os.getenv('FLASK_ENV') == 'development')