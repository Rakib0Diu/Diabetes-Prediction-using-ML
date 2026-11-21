# ============================================================
# DiaHealth – Diabetes Prediction System
# Flask Backend (app.py)
# ============================================================
# This file loads the pre-trained model, scaler, and feature
# names from .pkl files and serves a web interface where users
# can enter health data and receive a diabetes prediction.
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# ------------------------------------------------------------
# 1. Create the Flask application
# ------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------
# 2. Load saved ML files at startup (loaded once, reused for
#    every prediction request)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained Balanced Logistic Regression model
model = joblib.load(os.path.join(BASE_DIR, "best_diabetes_model.pkl"))

# Load the already-fitted StandardScaler
scaler = joblib.load(os.path.join(BASE_DIR, "diabetes_scaler.pkl"))

# Load the list of feature names (column order used during training)
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

print("✅ Model, scaler, and feature names loaded successfully!")
print(f"   Features ({len(feature_names)}): {feature_names}")

# ------------------------------------------------------------
# 3. Home page route – serves the single-page website
# ------------------------------------------------------------
@app.route("/")
def home():
    """Render the main HTML page."""
    return render_template("index.html")

# ------------------------------------------------------------
# 4. Prediction route – processes the form submission
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive form data, process it, scale it with the saved
    scaler, run prediction with the saved model, and return
    the result.
    """
    try:
        # --- 4a. Read all 14 input values from the form ---
        age = float(request.form["age"])
        gender = request.form["gender"]            # "Female" or "Male"
        pulse_rate = float(request.form["pulse_rate"])
        systolic_bp = float(request.form["systolic_bp"])
        diastolic_bp = float(request.form["diastolic_bp"])
        glucose = float(request.form["glucose"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        bmi = float(request.form["bmi"])
        family_diabetes = int(request.form["family_diabetes"])
        hypertensive = int(request.form["hypertensive"])
        family_hypertension = int(request.form["family_hypertension"])
        cardiovascular_disease = int(request.form["cardiovascular_disease"])
        stroke = int(request.form["stroke"])

        # --- 4b. Encode gender (Female → 0, Male → 1) ---
        # This matches the original LabelEncoder mapping used
        # during model training.
        gender_encoded = 0 if gender == "Female" else 1

        # --- 4c. Build a dictionary of all input values ---
        input_data = {
            "age": age,
            "gender": gender_encoded,
            "pulse_rate": pulse_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "glucose": glucose,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "family_diabetes": family_diabetes,
            "hypertensive": hypertensive,
            "family_hypertension": family_hypertension,
            "cardiovascular_disease": cardiovascular_disease,
            "stroke": stroke,
        }

        # --- 4d. Create a DataFrame with the exact column order
        #         stored in feature_names.pkl ---
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # --- 4e. Apply the already-fitted scaler ---
        # IMPORTANT: We only call transform(), never fit().
        input_scaled = scaler.transform(input_df)

        # --- 4f. Make a prediction ---
        prediction = model.predict(input_scaled)[0]

        # --- 4g. Determine the result message ---
        if prediction == 1:
            result = "Diabetic"
            result_class = "danger"       # CSS class for styling
        else:
            result = "Non-Diabetic"
            result_class = "success"      # CSS class for styling

        # Return the result to the template
        return render_template(
            "index.html",
            result=result,
            result_class=result_class,
            form_data=request.form,       # Keep the form filled
        )

    except ValueError:
        # Handle cases where the user entered non-numeric text
        # in a numeric field.
        return render_template(
            "index.html",
            error="Invalid input! Please enter valid numeric values in all fields.",
            form_data=request.form,
        )
    except Exception as e:
        # Catch any other unexpected errors
        return render_template(
            "index.html",
            error=f"An error occurred: {str(e)}",
            form_data=request.form,
        )

# ------------------------------------------------------------
# 5. Run the Flask development server
# ------------------------------------------------------------
if __name__ == "__main__":
    # debug=True auto-reloads on code changes (handy during
    # development). Set to False before any public demo.
    app.run(debug=True)
