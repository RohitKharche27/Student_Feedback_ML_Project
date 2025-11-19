import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Student Prediction App", layout="centered")

st.title("Student Prediction (Logistic Regression) — Streamlit App")
st.write(
    "Load the trained model (`Student_model.pkl`) and dataset (`ML_RAW_DATASET_CLEANED.xlsx`) "
    "or upload them below. Enter features to get a prediction and probability."
)

@st.cache_data
def load_model_from_file(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_excel(path):
    return pd.read_excel(path)

def try_load_local_files():
    model = None
    df = None
    # Try local filenames (common when you deploy with files in repo)
    try:
        model = load_model_from_file("Student_model.pkl")
        st.success("Loaded model from Student_model.pkl")
    except Exception:
        model = None

    try:
        df = load_excel("ML_RAW_DATASET_CLEANED.xlsx")
        st.success("Loaded dataset from ML_RAW_DATASET_CLEANED.xlsx")
    except Exception:
        df = None

    return model, df

model, df = try_load_local_files()

st.sidebar.header("Files / Uploads")

if model is None:
    uploaded_model = st.sidebar.file_uploader(
        "Upload `Student_model.pkl` (Pickle file)", type=["pkl", "pickle"]
    )
    if uploaded_model is not None:
        try:
            model = pickle.load(uploaded_model)
            st.sidebar.success("Model uploaded and loaded.")
        except Exception as e:
            st.sidebar.error(f"Could not load model: {e}")

if df is None:
    uploaded_data = st.sidebar.file_uploader(
        "Upload dataset `ML_RAW_DATASET_CLEANED.xlsx` (optional, used for defaults and stats)",
        type=["xlsx", "xls", "csv"],
    )
    if uploaded_data is not None:
        try:
            # uploaded_data may be BytesIO
            if str(uploaded_data.type).startswith("application/vnd.ms-excel") or uploaded_data.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_data)
            else:
                df = pd.read_csv(uploaded_data)
            st.sidebar.success("Dataset uploaded and loaded.")
        except Exception as e:
            st.sidebar.error(f"Could not load dataset: {e}")

st.markdown("---")

# If dataset is available show a bit of it and compute defaults
if df is not None:
    st.subheader("Dataset preview & stats")
    # show first rows
    st.dataframe(df.head(8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.markdown("**Numeric summary statistics (from dataset):**")
        st.dataframe(df[numeric_cols].describe().T)
    # Try to infer feature ranges
    def get_default(col, fallback=0):
        try:
            return float(df[col].mean())
        except Exception:
            return float(fallback)
else:
    st.info("No dataset loaded. You can upload `ML_RAW_DATASET_CLEANED.xlsx` in the sidebar to get defaults and stats.")

st.markdown("## Input features")

# Assume features are: age, salary, experience (as you mentioned). Use dataset to suggest ranges if available.
if df is not None:
    # Try common column names
    possible_age_cols = [c for c in df.columns if "age" in c.lower()] or []
    possible_salary_cols = [c for c in df.columns if "sal" in c.lower() or "salary" in c.lower()] or []
    possible_exp_cols = [c for c in df.columns if "exp" in c.lower() or "experience" in c.lower()] or []
else:
    possible_age_cols = possible_salary_cols = possible_exp_cols = []

# Defaults and min/max
def col_stats(cols, default_min, default_max, default_mean):
    if not cols:
        return default_min, default_max, default_mean
    col = cols[0]
    try:
        s = df[col].dropna().astype(float)
        return float(s.min()), float(s.max()), float(s.mean())
    except Exception:
        return default_min, default_max, default_mean

age_min, age_max, age_mean = col_stats(possible_age_cols, 10, 80, 30)
sal_min, sal_max, sal_mean = col_stats(possible_salary_cols, 0, 200000, 30000)
exp_min, exp_max, exp_mean = col_stats(possible_exp_cols, 0, 50, 5)

age = st.number_input("Age", min_value=float(age_min), max_value=float(age_max), value=float(round(age_mean,0)))
salary = st.number_input("Salary", min_value=float(sal_min), max_value=float(sal_max), value=float(round(sal_mean,0)))
experience = st.number_input("Experience (years)", min_value=float(exp_min), max_value=float(exp_max), value=float(round(exp_mean,0)))

st.markdown("---")

# Button for single prediction
if st.button("Predict"):
    if model is None:
        st.error("No model loaded. Upload `Student_model.pkl` in the sidebar or place it in the app folder.")
    else:
        # Prepare input vector. We'll try to match the order expected by the model:
        # If the model was trained with features order [age, salary, experience]
        X = np.array([[age, salary, experience]], dtype=float)

        # If model expects scaled input, either the model pipeline includes the scaler
        # or we assume raw. We'll try to detect if model has 'predict_proba' and accepts shape.
        try:
            proba = model.predict_proba(X)
            class_idx = np.argmax(proba, axis=1)[0]
            pred_class = model.classes_[class_idx] if hasattr(model, "classes_") else int(class_idx)
            st.success(f"Predicted class: **{pred_class}**")
            st.write(f"Probabilities: {dict(zip(model.classes_, proba[0]))}" if hasattr(model, "classes_") else f"Probabilities: {proba[0]}")
        except Exception as e:
            # Sometimes model is a pipeline or expects different shape — try feeble fallback
            try:
                pred = model.predict(X)
                st.success(f"Predicted class: **{pred[0]}**")
                # if no predict_proba, show only prediction
            except Exception as e2:
                st.error(f"Model failed to predict. Error: {e} / {e2}")

# Batch predict: allow the user to upload a CSV of new rows
st.markdown("## Batch predictions (CSV)")
st.write("Upload a CSV with columns `age`, `salary`, `experience` (or similar). The app will try to map columns by name.")

batch_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="batch_csv")
if batch_file is not None:
    try:
        batch_df = pd.read_csv(batch_file)
        st.write("Uploaded preview:")
        st.dataframe(batch_df.head())
        # Try to find columns
        # find columns containing age/sal/exp
        def find_col(df, keywords):
            for c in df.columns:
                lc = c.lower()
                for kw in keywords:
                    if kw in lc:
                        return c
            return None

        age_col = find_col(batch_df, ["age"])
        sal_col = find_col(batch_df, ["sal", "salary", "income", "pay"])
        exp_col = find_col(batch_df, ["exp", "experience", "years"])

        missing = []
        if age_col is None:
            missing.append("age")
        if sal_col is None:
            missing.append("salary")
        if exp_col is None:
            missing.append("experience")

        if missing:
            st.warning(f"Could not auto-detect these columns: {missing}. You may need to rename columns before uploading.")
        else:
            X_batch = batch_df[[age_col, sal_col, exp_col]].astype(float).values
            if model is None:
                st.error("No model loaded.")
            else:
                try:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_batch)
                        preds = model.predict(X_batch)
                        out = batch_df.copy()
                        out["prediction"] = preds
                        # attach probability for predicted class
                        pred_probs = [probs[i, np.argmax(probs[i])] for i in range(len(probs))]
                        out["pred_prob"] = pred_probs
                        st.success("Batch prediction done.")
                        st.dataframe(out.head(20))
                        csv = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
                    else:
                        preds = model.predict(X_batch)
                        out = batch_df.copy()
                        out["prediction"] = preds
                        st.success("Batch prediction done (no probabilities available).")
                        st.dataframe(out.head(20))
                        csv = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

st.markdown("---")
st.caption("Tips: If your model was trained on scaled features, put the scaler and model into a single pipeline before pickling (e.g. sklearn.pipeline.Pipeline). That avoids having to scale at inference time.")
