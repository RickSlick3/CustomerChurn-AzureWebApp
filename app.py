from flask import Flask, render_template, request, redirect, url_for
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
@app.route("/", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        # Redirect to welcome page with username as a query parameter
        return redirect(url_for("welcome", username=username))
    return render_template("form.html")

@app.route("/welcome")
def welcome():
    username = request.args.get("username", "Guest")
    return render_template("welcome.html", username=username)

@app.route('/sample')
def display_csv():
    sample_df = df.head(100) # make 400
    table_html = sample_df.to_html(classes="table table-bordered")  # 'table table-bordered' adds Bootstrap styling
    
    # Render the HTML template and pass the table_html
    return render_template('sample.html', table_html=table_html)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=os.getenv('FLASK_ENV') == 'development')