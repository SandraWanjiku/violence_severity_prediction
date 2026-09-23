# Women's Violence Analysis and Prediction System

## Project Overview

This project analyzes historical data on violence against women and applies machine learning to classify violence severity.

The project combines:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Statistical analysis
* Exploratory Data Analysis (EDA)
* Machine Learning
* Flask

The final system provides a web interface where users can enter selected characteristics of a violence case and receive a predicted severity category.

## Problem Statement

Violence against women is a significant social issue that affects safety, health, and social and economic wellbeing.

Data analysis can help identify patterns in reported violence, examine relationships between demographic and social factors, and provide evidence that can support further research and intervention planning.

This project therefore analyzes historical women's violence data, investigates important patterns, develops classification models for violence severity, and deploys the resulting model through a Flask web application.

## Objectives

The project aims to:

1. Inspect and clean the women's violence dataset.
2. Perform exploratory data analysis.
3. Identify patterns and trends in violence incidents and deaths.
4. Examine relationships between violence, age, literacy, and violence type.
5. Conduct statistical tests on selected variables.
6. Develop machine learning classification models.
7. Evaluate the performance of the models.
8. Save the selected machine learning model.
9. Deploy the model using Flask.
10. Provide an interactive prediction interface.

## Exploratory Data Analysis

The analysis examines:

* Violence incidents over time
* Deaths over time
* Violence types
* Victim age groups
* Literacy levels
* Fatality rates
* Relationships between categorical and numerical variables

The project also includes univariate, bivariate, and multivariate analysis supported by visualizations.

## Statistical Analysis

Three statistical techniques were applied:

### T-Test

A t-test was used to examine whether incident counts differed between literacy groups.

### ANOVA

One-way ANOVA was used to examine differences in incident counts across violence types.

### Chi-Square Test

A chi-square test was used to examine the association between victim age and violence type.

## Machine Learning

Three classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

The Random Forest model achieved an accuracy of approximately **92.31%** on the test set.

Model evaluation included:

* Accuracy
* Classification report
* Confusion matrix
* ROC curve
* ROC-AUC
* Feature importance

## Feature Engineering

Additional features were created to support the analysis and machine learning process, including:

* Decade
* Year Category
* Fatality Rate
* Age Risk Group
* Violence Severity

## Flask Web Application

The trained model was deployed through Flask.

The application contains:

### Home Page

Provides:

* Project introduction
* Problem statement
* Project overview

### Prediction Page

Allows users to enter:

* Year
* Violence Type
* Victim Age
* Literacy Scale
* Incident Count
* Death Count

### Result Page

Displays the predicted violence severity:

* Low
* Medium
* High

## Project Structure

```text
womensviolenceseverity/
│
├── app.py
├── clean_womens_violence.csv
├── womensviolenceanalysis_capstone.ipynb
├── .gitignore
│
├── models/
│   ├── womens_violence_model.pkl
│   ├── model_features.pkl
│   └── label_encoder.pkl
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── result.html
│
└── static/
    └── style.css
```

## How to Run the Application

### 1. Clone the repository

```bash
git clone https://github.com/SandraWanjiku/violence_severity_prediction.git
```

### 2. Navigate into the project

```bash
cd violence_severity_prediction
```

### 3. Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install the required packages

```bash
pip install flask pandas joblib scikit-learn
```

### 5. Run the Flask application

```bash
python app.py
```

### 6. Open the application

Open the local Flask address displayed in the terminal, normally:

```text
http://127.0.0.1:5000
```

## Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| Pandas       | Data manipulation and analysis |
| NumPy        | Numerical computing            |
| Matplotlib   | Data visualization             |
| Seaborn      | Statistical visualization      |
| Scikit-learn | Machine learning               |
| Joblib       | Model serialization            |
| Flask        | Web application deployment     |
| HTML/CSS     | Web interface                  |

## Key Findings

The exploratory analysis identified several notable patterns, including:

* 2011 recorded the highest number of violence incidents in the analyzed dataset.
* 2024 recorded the lowest number of violence incidents.
* Overall incident levels showed a declining trend between 2010 and 2024, although some intermediate years showed increases.
* Young adults aged 19–30 represented the most affected age group.
* Statistical testing identified significant relationships within selected literacy, violence-type, and age variables.

## Limitations

The results should be interpreted within the limitations of the available dataset.

In particular, the violence severity target was derived from incident-count thresholds, meaning that incident count is closely related to the definition of the target variable. This can introduce target leakage when incident count is also used as a predictor.

Therefore, the reported model performance should not be interpreted as evidence of equivalent real-world predictive performance.

The system is intended primarily as an academic demonstration of data analysis, statistical analysis, machine learning, and Flask deployment.

## Future Improvements

Future versions could:

* Use a larger and more representative dataset.
* Improve the definition of violence severity.
* Address target leakage through improved feature selection.
* Add additional machine learning models.
* Introduce model explainability techniques.
* Add interactive data visualizations.
* Improve the prediction interface.
* Develop the application into a broader data-driven decision-support platform.

## Author

**Sandra Wanjiku Ataro**

Data Science / Machine Learning Capstone Project
