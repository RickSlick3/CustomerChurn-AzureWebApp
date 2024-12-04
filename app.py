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


@app.route("/data", methods=["GET", "POST"])
def display_data():
    filtered_df = df.head(400)

    if request.method == "POST":
        # Get the clientnum from the form
        clientnum = request.form.get("clientnum", "")
        filtered_df = df[df["CLIENTNUM"].astype(str).str.lower().str.contains(clientnum)]

    # Check if a sort request was made
    sort_column = request.args.get('sort_column')
    if sort_column and sort_column in df.columns:
        filtered_df = filtered_df.sort_values(by=sort_column)
    
    df_html = filtered_df.to_html(index=True)
    return render_template("data.html", table=df_html)


@app.route("/alldata")
def display_all_data():
    df_html = df.to_html(index=True)
    return render_template("alldata.html", table=df_html)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=os.getenv('FLASK_ENV') == 'development')