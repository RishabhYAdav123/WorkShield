from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
from pathlib import Path
import traceback

app = Flask(__name__)
OUTPUT_DIR = Path("static/output")
RISK_OUTPUT_PATH = OUTPUT_DIR / "predicted_attrition_yes.csv"
FINAL_ANALYSIS_PATH = OUTPUT_DIR / "final_analysis.csv"

# Load SVM model and scaler.
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = list(scaler.feature_names_in_)


def get_reason(row):
    if row.get("JobSatisfaction", 0) < 2:
        return "Low Job Satisfaction"
    if row.get("MonthlyIncome", 0) < 3000:
        return "Low Salary"
    if str(row.get("OverTime", "")).strip().lower() == "yes":
        return "Work Overload"
    if row.get("DistanceFromHome", 0) > 20:
        return "Long Distance"
    return "General Attrition Risk"


def get_action(row):
    if row.get("PerformanceRating", 0) >= 3 and row.get("JobSatisfaction", 0) < 3:
        return "Retain (High Value Employee)"
    if row.get("PerformanceRating", 0) < 2:
        return "No Immediate Action"
    return "Monitor"


def validate_csv_upload(uploaded_file):
    if not uploaded_file or uploaded_file.filename == "":
        return "No file uploaded", "Choose a CSV file and try again."

    if not uploaded_file.filename.lower().endswith(".csv"):
        return "Unsupported file type", "Upload a .csv file so the system can read the employee data."

    return None, None


def prepare_analysis_data(df):
    required_columns = [
        "JobSatisfaction",
        "MonthlyIncome",
        "OverTime",
        "DistanceFromHome",
        "PerformanceRating",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing_columns)
        )

    numeric_columns = [
        "JobSatisfaction",
        "MonthlyIncome",
        "DistanceFromHome",
        "PerformanceRating",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df


def add_analysis_columns(df):
    df = prepare_analysis_data(df)
    df["Reason"] = df.apply(get_reason, axis=1)
    df["Action"] = df.apply(get_action, axis=1)
    return df


@app.route("/")
def index():
    return render_template("first_model.html")


@app.route("/sample/<sample_size>")
def sample(sample_size):
    sample_files = {
        "5": "sample_5_rows.csv",
        "10": "sample_10_rows.csv",
        "50": "sample_50_rows.csv",
    }
    filename = sample_files.get(sample_size)
    if not filename:
        return render_template(
            "result.html",
            status="error",
            title="Sample not found",
            message="Please choose one of the available sample CSV files.",
        ), 404

    return send_file(
        Path(filename),
        download_name=filename,
        as_attachment=True,
        mimetype="text/csv",
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        uploaded_file = request.files.get("csvFile")
        title, message = validate_csv_upload(uploaded_file)
        if title:
            return render_template(
                "result.html",
                status="error",
                title=title,
                message=message,
            ), 400

        df = pd.read_csv(uploaded_file)

        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0)
        df_encoded = df_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

        df_scaled = scaler.transform(df_encoded)

        raw_preds = model.predict(df_scaled)
        preds = [1 if pred == "Yes" or pred == 1 else 0 for pred in raw_preds]
        df["Attrition_Prediction"] = preds

        result_df = df[df["Attrition_Prediction"] == 1]

        if result_df.empty:
            return render_template(
                "result.html",
                status="success",
                title="No high-risk employees found",
                message="The model predicts no employees in this file are likely to leave.",
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(RISK_OUTPUT_PATH, index=False)

        preview_columns = [
            column for column in [
                "EmployeeNumber",
                "JobRole",
                "JobSatisfaction",
                "MonthlyIncome",
                "OverTime",
                "DistanceFromHome",
                "PerformanceRating",
                "Attrition_Prediction",
            ]
            if column in result_df.columns
        ]

        return render_template(
            "risk_analysis.html",
            total_rows=len(result_df),
            preview_rows=result_df[preview_columns].head(8).to_dict("records"),
            preview_columns=preview_columns,
        )

    except Exception as e:
        traceback.print_exc()
        return render_template(
            "result.html",
            status="error",
            title="Prediction failed",
            message=f"An error occurred: {str(e)}",
        ), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        uploaded_file = request.files.get("analysisFile")
        title, message = validate_csv_upload(uploaded_file)
        if title:
            return render_template(
                "result.html",
                status="error",
                title=title,
                message=message,
            ), 400

        df = pd.read_csv(uploaded_file)
        df = add_analysis_columns(df)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(FINAL_ANALYSIS_PATH, index=False)

        return send_file(
            FINAL_ANALYSIS_PATH,
            download_name="final_analysis.csv",
            as_attachment=True,
            mimetype="text/csv",
        )

    except ValueError as e:
        return render_template(
            "result.html",
            status="error",
            title="Analysis failed",
            message=str(e),
        ), 400

    except Exception as e:
        traceback.print_exc()
        return render_template(
            "result.html",
            status="error",
            title="Analysis failed",
            message=f"An error occurred: {str(e)}",
        ), 500


@app.route("/download-risk")
def download_risk():
    if not RISK_OUTPUT_PATH.exists():
        return render_template(
            "result.html",
            status="error",
            title="Risk CSV not found",
            message="Run the attrition prediction first, then download the risk CSV.",
        ), 404

    return send_file(
        RISK_OUTPUT_PATH,
        download_name="predicted_attrition_yes.csv",
        as_attachment=True,
        mimetype="text/csv",
    )


@app.route("/analyze-saved", methods=["POST"])
def analyze_saved():
    try:
        if not RISK_OUTPUT_PATH.exists():
            return render_template(
                "result.html",
                status="error",
                title="Risk CSV not found",
                message="Run the attrition prediction first before starting risk analysis.",
            ), 404

        df = pd.read_csv(RISK_OUTPUT_PATH)
        df = add_analysis_columns(df)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(FINAL_ANALYSIS_PATH, index=False)

        return send_file(
            FINAL_ANALYSIS_PATH,
            download_name="final_analysis.csv",
            as_attachment=True,
            mimetype="text/csv",
        )

    except ValueError as e:
        return render_template(
            "result.html",
            status="error",
            title="Analysis failed",
            message=str(e),
        ), 400

    except Exception as e:
        traceback.print_exc()
        return render_template(
            "result.html",
            status="error",
            title="Analysis failed",
            message=f"An error occurred: {str(e)}",
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
