import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from typing import List

st.set_page_config(page_title="Student Prediction (dynamic inputs)", layout="centered")
st.title("Student Prediction — Streamlit App (auto-detects model features)")

@st.cache_data
def load_model(path_or_file):
    try:
        if hasattr(path_or_file, "read"):
            # file-like (uploaded)
            return pickle.load(path_or_file)
        else:
            with open(path_or_file, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        raise

@st.cache_data
def load_excel(path_or_file):
    if hasattr(path_or_file, "read"):
        return pd.read_excel(path_or_file)
    else:
        return pd.read_excel(path_or_file)

# Try load local files
model = None
df = None
try:
    model = load_model("Student_model.pkl")
    st.sidebar.success("Loaded Student_model.pkl")
except Exception:
    model = None
try:
    df = load_excel("ML_RAW_DATASET_CLEANED.xlsx")
    st.sidebar.success("Loaded ML_RAW_DATASET_CLEANED.xlsx")
except Exception:
    df = None

st.sidebar.header("Uploads")
if model is None:
    uploaded_model = st.sidebar.file_uploader("Upload Student_model.pkl", type=["pkl", "pickle"])
    if uploaded_model is not None:
        try:
            model = load_model(uploaded_model)
            st.sidebar.success("Model uploaded")
        except Exception as e:
            st.sidebar.error(f"Could not load model: {e}")

if df is None:
    uploaded_data = st.sidebar.file_uploader("Upload dataset (optional)", type=["xlsx", "xls", "csv"])
    if uploaded_data is not None:
        try:
            if uploaded_data.name.lower().endswith((".xls", ".xlsx")):
                df = load_excel(uploaded_data)
            else:
                df = pd.read_csv(uploaded_data)
            st.sidebar.success("Dataset uploaded")
        except Exception as e:
            st.sidebar.error(f"Could not load dataset: {e}")

st.markdown("---")

if model is None:
    st.warning("No model loaded yet. Upload `Student_model.pkl` in the sidebar or place it in the app folder.")
    st.stop()

# Determine model input feature info
n_features_expected = getattr(model, "n_features_in_", None)
model_feature_names: List[str] = []
if hasattr(model, "feature_names_in_"):
    try:
        model_feature_names = list(model.feature_names_in_)
    except Exception:
        model_feature_names = []
elif hasattr(model, "coef_") and hasattr(model, "classes_"):
    # fallback guess: cannot reliably infer names
    model_feature_names = []

# If sklearn didn't store names, create generic names
if n_features_expected is None:
    # fallback - try to infer from model if possible, else assume 3 (age, salary, experience)
    n_features_expected = 3

if not model_feature_names:
    # create generic names
    model_feature_names = [f"feature_{i+1}" for i in range(n_features_expected)]
else:
    # if feature_names exist but length mismatch, adjust
    if len(model_feature_names) != n_features_expected:
        # ensure lengths match
        if len(model_feature_names) < n_features_expected:
            # pad generic names
            extra = [f"feature_{i+1}" for i in range(len(model_feature_names), n_features_expected)]
            model_feature_names = model_feature_names + extra
        elif len(model_feature_names) > n_features_expected:
            model_feature_names = model_feature_names[:n_features_expected]

st.info(f"Model expects **{n_features_expected}** features: `{', '.join(model_feature_names)}`")

# Helper: get sensible defaults from dataset if available
def default_for(col_name, fallback=0.0):
    if df is None:
        return float(fallback)
    # try exact col match or fuzzy
    if col_name in df.columns:
        try:
            return float(df[col_name].dropna().astype(float).mean())
        except Exception:
            return float(fallback)
    # fuzzy match by lower-case substring
    lc = col_name.lower()
    for c in df.columns:
        if lc in c.lower():
            try:
                return float(df[c].dropna().astype(float).mean())
            except Exception:
                continue
    return float(fallback)

st.header("Enter feature values")

# Create inputs dynamically
user_inputs = {}
for fname in model_feature_names:
    # choose numeric input by default
    default_val = default_for(fname, 0.0)
    # try to get a reasonable min/max from df if possible
    minv, maxv = None, None
    if df is not None:
        # fuzzy locate column
        found = None
        if fname in df.columns:
            found = fname
        else:
            for c in df.columns:
                if fname.lower() in c.lower() or c.lower() in fname.lower():
                    found = c
                    break
        if found is not None:
            try:
                s = df[found].dropna().astype(float)
                minv, maxv = float(s.min()), float(s.max())
            except Exception:
                minv, maxv = None, None
    # render numeric input; if ranges available use them
    label = f"{fname} (auto)"
    if minv is not None and maxv is not None:
        user_val = st.number_input(label, min_value=minv, max_value=maxv, value=float(round(default_val, 2)))
    else:
        user_val = st.number_input(label, value=float(round(default_val, 2)))
    user_inputs[fname] = float(user_val)

st.markdown("---")

# Single prediction
if st.button("Predict single row"):
    X = np.array([ [user_inputs[f] for f in model_feature_names] ], dtype=float)
    st.write("Input array shape:", X.shape)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            pred = model.predict(X)
            st.success(f"Prediction: **{pred[0]}**")
            # show probability of predicted class
            if hasattr(model, "classes_"):
                class_probs = dict(zip(map(str, model.classes_), proba[0].round(4)))
                st.write("Probabilities:", class_probs)
            else:
                st.write("Probabilities:", proba[0].round(4))
        else:
            pred = model.predict(X)
            st.success(f"Prediction: **{pred[0]}** (model has no predict_proba)")
    except Exception as e:
        st.error(f"Model failed to predict. Error: {e}")
        st.info("Common causes: wrong feature order, categorical features encoded differently, or model expects encoded/dummy features.")

# Batch prediction
st.markdown("## Batch predictions (CSV upload)")
st.write("Upload a CSV with columns corresponding to the model's feature names. If names differ, map columns in the next step.")
batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_csv")

if batch_file is not None:
    try:
        batch_df = pd.read_csv(batch_file)
    except Exception:
        batch_df = pd.read_csv(batch_file, encoding="latin1")
    st.write("Preview of uploaded CSV:")
    st.dataframe(batch_df.head())

    # Attempt auto-mapping
    mapping = {}
    detected = []
    for fname in model_feature_names:
        # find best-matching column in CSV
        match = None
        for c in batch_df.columns:
            if c == fname:
                match = c
                break
        if match is None:
            # substring match
            for c in batch_df.columns:
                if fname.lower() in c.lower() or c.lower() in fname.lower():
                    match = c
                    break
        if match:
            mapping[fname] = match
            detected.append(fname)

    st.write(f"Auto-detected mapping for: {detected}")

    # Show mapping UI for any unmapped features
    unmapped = [f for f in model_feature_names if f not in mapping]
    if unmapped:
        st.warning(f"Could not auto-detect columns for: {unmapped}. Please map them manually (or rename CSV columns).")
        st.write("Map CSV columns to model features:")
        for f in unmapped:
            choice = st.selectbox(f"Column for model feature `{f}`", options=["--none--"] + list(batch_df.columns), key=f"map_{f}")
            if choice != "--none--":
                mapping[f] = choice

    # confirm all mapped
    if set(mapping.keys()) == set(model_feature_names):
        # Build X in correct order
        X_batch = batch_df[[mapping[f] for f in model_feature_names]].astype(float).values
        st.write("Built input matrix for model with shape", X_batch.shape)
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
                st.download_button("Download predictions CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")
            else:
                preds = model.predict(X_batch)
                out = batch_df.copy()
                out["prediction"] = preds
                st.success("Batch prediction done (no probabilities).")
                st.dataframe(out.head(20))
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
    else:
        st.error("Not all model features are mapped. Map remaining features to columns and try again.")

st.markdown("---")
st.caption("If your model expects encoded categorical features (one-hot, label-encoded), make sure to provide the same encoding at inference time or pickle the whole preprocessing pipeline together with the model.")
