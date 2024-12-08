from flask import Flask, render_template, request, redirect, url_for, Response
from dotenv import load_dotenv
import os
from utils.df_utils import *
from utils.dashboard_utils import *
from utils.ml_utils import *

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
    filtered_df = df.head(DEFAULT_ROWS)

    # Handle POST request for filtering
    if request.method == "POST":
        clientnum = request.form.get("clientnum", "")
        filtered_df = filter_by_clientnum(df, clientnum)

    # Handle GET request for sorting
    if request.method == "GET":
        sort_column = request.args.get('sort_column')
        filtered_df = sort_by_column(filtered_df, sort_column)

    df_html = filtered_df.to_html(index=True)
    return render_template("data.html", table=df_html)

@app.route("/alldata")
def display_all_data():
    df_html = df.to_html(index=True)
    return render_template("alldata.html", table=df_html)

@app.route("/dashboard")
def dashboard():
    churn_by_income(mapped_df)
    corr_dict(mapped_df)
    gender_attrition(mapped_df)
    rel_status_attrition(mapped_df)
    edu_attrition(mapped_df)
    age_plot(mapped_df)
    transactions_plot(mapped_df)
    # credit_plot(mapped_df)
    return render_template('dashboard.html')

@app.route("/churn")
def churn():
    log = logistic_regression()
    forest = random_forest()
    knn = knn_classifier()
    return render_template('churn.html', log=log, forest=forest, knn=knn)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=os.getenv('FLASK_ENV') == 'development')