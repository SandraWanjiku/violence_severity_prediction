from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and preprocessing files
model = joblib.load("models/womens_violence_model.pkl")
model_features = joblib.load("models/model_features.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    # Get user inputs
    year = int(request.form["year"])
    violence_type = request.form["violence_type"]
    victim_age = request.form["victim_age"]
    literate_scale = request.form["literate_scale"]
    death_count = int(request.form["death_count"])
    incident_count = int(request.form["incident_count"])

    # Feature engineering
    if incident_count == 0:
        fatality_rate = 0
    else:
        fatality_rate = death_count / incident_count

    if year < 2015:
        year_category = "Early"
    else:
        year_category = "Recent"

    decade = (year // 10) * 10

    age_map = {
        "0-18": "Child",
        "19-24": "Young Adult",
        "25-30": "Young Adult",
        "31+": "Adult"
    }

    age_risk_group = age_map[victim_age]

    input_data = pd.DataFrame([{
        "Year": year,
        "Violence Type": violence_type,
        "Victim Age": victim_age,
        "Literate Scale": literate_scale,
        "Death Count": death_count,
        "Fatality Rate": fatality_rate,
        "Year Category": year_category,
        "Age Risk Group": age_risk_group,
        "Decade": decade
    }])

    # Match training encoding
    encoded_input = pd.DataFrame(
        0,
        index=[0],
        columns=model_features
    )

    numerical_features = [
        "Year",
        "Death Count",
        "Fatality Rate",
        "Decade"
    ]

    for feature in numerical_features:
        encoded_input[feature] = input_data[feature].iloc[0]

    categorical_features = [
        "Violence Type",
        "Victim Age",
        "Literate Scale",
        "Year Category",
        "Age Risk Group"
    ]

    for feature in categorical_features:
        value = input_data[feature].iloc[0]
        dummy_column = f"{feature}_{value}"

        if dummy_column in model_features:
            encoded_input[dummy_column] = 1

    # Make prediction
    prediction = model.predict(encoded_input)
    severity = label_encoder.inverse_transform(prediction)[0]

    return render_template(
        "result.html",
        prediction=severity
    )


if __name__ == "__main__":
    app.run(debug=True)