from flask import Flask, render_template, request, redirect, url_for, Response
from dotenv import load_dotenv
import os
from utils.df_utils import *

app = Flask(__name__)

DEFAULT_ROWS = 400

# get .env
load_dotenv()

# get connectin string from .env
app.config['CONNECTION_STRING'] = os.getenv('CONNECTION_STRING')
if not app.config['CONNECTION_STRING']:
    raise ValueError("CONNECTION_STRING is not set in the .env file.")

df = create_df_from_db(app.config['CONNECTION_STRING'])

mapped_df = set_up(df)

csv_file_path = 'utils/bank_churners_data.csv'
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
    filtered_df = df.head(DEFAULT_ROWS)

    # Handle POST request for filtering
    if request.method == "POST":
        # clientnum = request.form.get("clientnum", "")
        filtered_df = filter_by_clientnum(df, request.form.get("clientnum", ""))

    # Handle GET request for sorting
    if request.method == "GET":
        # sort_column = request.args.get('sort_column')
        filtered_df = sort_by_column(filtered_df, request.args.get('sort_column'))

    df_html = filtered_df.to_html(index=True)
    return render_template("data.html", table=df_html)

@app.route("/alldata")
def display_all_data():
    df_html = df.to_html(index=True)
    return render_template("alldata.html", table=df_html)

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

@app.route("/corrheatmap")
def corrheatmap():
    img = corr_heatmap(mapped_df)
    return Response(img, mimetype='image/png')

@app.route("/income")
def income():
    img = churn_by_income(mapped_df)
    return Response(img, mimetype='image/png')

@app.route("/corrdict")
def corrdict():
    img = corr_dict(mapped_df)
    return Response(img, mimetype='image/png')

@app.route("/genderattrition")
def genderattrition():
    img = gender_attrition(mapped_df)
    return Response(img, mimetype='image/png')

@app.route("/relstatusattrition")
def relstatusattrition():
    img = rel_status_attrition(mapped_df)
    return Response(img, mimetype='image/png')

@app.route("/churn")
def churn():
    return

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=os.getenv('FLASK_ENV') == 'development')