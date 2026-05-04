# WorkShield Employee Attrition Predictor

WorkShield is a Flask-based employee attrition prediction app. Users upload employee records as a CSV file, the saved SVM model predicts high-risk employees, and the app generates a risk CSV.

The project also includes a second-stage analysis flow that adds rule-based `Reason` and `Action` columns for high-risk employees.

## Features

- CSV upload for employee attrition prediction
- SVM model prediction using saved `svm_model.pkl` and `scaler.pkl`
- Automatic high-risk employee CSV generation
- Post-prediction risk analysis
- Downloadable final analysis CSV
- Sample CSV files for testing

## Project Structure

```text
app.py
requirements.txt
svm_model.pkl
scaler.pkl
sample_5_rows.csv
sample_10_rows.csv
sample_50_rows.csv
static/
  1stmodel.css
templates/
  first_model.html
  result.html
  risk_analysis.html
```

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open the app at:

```text
http://127.0.0.1:5000/
```

## Workflow

1. Upload an employee CSV file.
2. Download or preview the predicted high-risk employees.
3. Analyze high-risk employees to generate reason and action recommendations.
