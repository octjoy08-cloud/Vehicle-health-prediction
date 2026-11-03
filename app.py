import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Vehicle Health Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: #f6f8fb; padding: 14px 16px; border-radius: 12px; border: 1px solid #e7ebf0;}
div[data-testid="stMetric"] label {font-size: .9rem;}
.small-note {color: #64748b; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Vehicle Health Intelligence Dashboard")
st.caption("Telematics and CAN-bus based monitoring • Vehicle Health Index • Fault-risk screening")

# ------------------------- Data loading -------------------------
def flatten_json_records(raw):
    """Flatten JSON/JSONL records used by the notebook (messages/messageData)."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return pd.DataFrame()

    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("messages"), list):
            rows.extend(x for x in item["messages"] if isinstance(x, dict))
        elif isinstance(item.get("messageData"), list):
            rows.extend(x for x in item["messageData"] if isinstance(x, dict))
        else:
            rows.append(item)
    return pd.DataFrame(rows)


def read_upload(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith((".jsonl", ".ndjson")):
            uploaded.seek(0)
            records = []
            for line in uploaded.getvalue().decode("utf-8-sig").splitlines():
                if line.strip():
                    records.append(json.loads(line))
            return flatten_json_records(records)
        if name.endswith(".json"):
            uploaded.seek(0)
            return flatten_json_records(json.loads(uploaded.getvalue().decode("utf-8-sig")))
        if name.endswith(".parquet"):
            return pd.read_parquet(uploaded)
    except Exception as exc:
        st.error(f"Could not read {uploaded.name}: {exc}")
    return None


def clean_telematics(frame):
    df = frame.copy()
    if "eventDateTime" in df.columns:
        df["eventDateTime"] = pd.to_datetime(df["eventDateTime"], errors="coerce", utc=True)
    if "vehicleBatteryVoltage" in df.columns:
        df["vehicleBatteryVoltage"] = pd.to_numeric(df["vehicleBatteryVoltage"], errors="coerce")
    for col in df.select_dtypes(include=["object"]).columns:
        # Convert object columns that are mostly numeric.
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() >= 0.8:
            df[col] = converted
    return df.drop_duplicates().reset_index(drop=True)


def vhi_from_voltage(v):
    # Same voltage-to-VHI mapping used in the supplied notebook.
    if pd.isna(v):
        return np.nan
    lo = 0.0
    return float(np.clip(np.interp(v, [23.0, 23.5, 24.0, 24.5, 25.0, 27.0],
                                  [0.0, 10.0, 40.0, 70.0, 100.0, 100.0]), lo, 100.0))


def health_label(v):
    if pd.isna(v):
        return "Unknown"
    if v >= 25.0:
        return "Healthy"
    if v >= 24.5:
        return "Normal degradation"
    if v >= 24.0:
        return "Attention required"
    return "Critical"


def numeric_features(df, target_cols=()):
    excluded = set(target_cols) | {
        "vehicleId", "vin", "eventDateTime", "timestamp",
        "vhi", "is_fault", "health_status", "can_health_status",
        "is_anomaly", "EngSpeed",
    }
    cols = []
    for c in df.select_dtypes(include=np.number).columns:
        if c not in excluded and df[c].nunique(dropna=True) > 1:
            cols.append(c)
    return cols


def fit_models(df):
    if "vehicleBatteryVoltage" not in df.columns:
        return None, None, [], "The telematics file must contain vehicleBatteryVoltage."

    data = df.copy()
    data["vehicleBatteryVoltage"] = pd.to_numeric(data["vehicleBatteryVoltage"], errors="coerce")
    data = data.dropna(subset=["vehicleBatteryVoltage"])
    if len(data) < 20:
        return None, None, [], "At least 20 valid telematics rows are recommended for model training."

    data["vhi"] = data["vehicleBatteryVoltage"].apply(vhi_from_voltage)
    data["is_fault"] = (data["vehicleBatteryVoltage"] < 24.0).astype(int)
    features = numeric_features(data, target_cols=["vehicleBatteryVoltage"])
    if not features:
        return None, None, [], "No usable numeric predictor columns found apart from battery voltage."

    X = data[features].replace([np.inf, -np.inf], np.nan)
    # Remove columns that are entirely missing.
    features = [c for c in features if X[c].notna().any()]
    X = X[features]
    y_reg = data["vhi"]
    y_cls = data["is_fault"]

    if len(features) == 0:
        return None, None, [], "No complete numeric predictor columns are available."

    X_train, X_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.25, random_state=42
    )
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1)),
    ])
    reg.fit(X_train, yr_train)
    yr_pred = reg.predict(X_test)
    reg_metrics = {
        "MAE": mean_absolute_error(yr_test, yr_pred),
        "RMSE": mean_squared_error(yr_test, yr_pred) ** 0.5,
        "R²": r2_score(yr_test, yr_pred) if len(yr_test) > 1 else np.nan,
    }

    cls = None
    cls_metrics = None
    if y_cls.nunique() > 1 and y_cls.value_counts().min() >= 2:
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, y_cls, test_size=0.25, random_state=42, stratify=y_cls
        )
        cls = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=250, random_state=42, class_weight="balanced",
                min_samples_leaf=2, n_jobs=-1
            )),
        ])
        cls.fit(Xc_train, yc_train)
        yc_pred = cls.predict(Xc_test)
        cls_metrics = {
            "Accuracy": accuracy_score(yc_test, yc_pred),
            "Precision": precision_score(yc_test, yc_pred, zero_division=0),
            "Recall": recall_score(yc_test, yc_pred, zero_division=0),
            "F1": f1_score(yc_test, yc_pred, zero_division=0),
        }
    return (reg, reg_metrics), (cls, cls_metrics), features, None


def metric_color(label):
    return {"Healthy": "🟢", "Normal degradation": "🟡",
            "Attention required": "🟠", "Critical": "🔴"}.get(label, "⚪")


# ------------------------- Sidebar -------------------------
with st.sidebar:
    st.header("Data input")
    tele_file = st.file_uploader(
        "Upload telematics data",
        type=["csv", "json", "jsonl", "ndjson", "parquet"],
        help="Supports CSV and JSON/JSONL. JSON records containing a 'messages' list are flattened automatically."
    )
    can_file = st.file_uploader(
        "Upload CAN-bus data (optional)",
        type=["csv", "json", "jsonl", "ndjson", "parquet"],
    )
    st.divider()
    st.markdown("**Expected telematics fields**")
    st.caption("vehicleBatteryVoltage, vehicleId, eventDateTime, speed, accelX/accelY/accelZ, odometer, signalStrength, gpsSignalQuality, backupBatteryVoltage")
    st.markdown("**Expected CAN fields**")
    st.caption("vin, timestamp, spnName, spnValue (or wide-format columns such as EngSpeed and RoadSpeed)")

tele_raw = read_upload(tele_file)
can_raw = read_upload(can_file)

if tele_raw is None:
    st.info("Upload your telematics CSV/JSON file in the sidebar to activate the dashboard.")
    st.markdown("### Dashboard preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicle Health Index", "—", "Waiting for data")
    c2.metric("Fault risk", "—", "Waiting for data")
    c3.metric("Telematics records", "0")
    c4.metric("CAN records", "0")
    st.markdown("""
    **Getting started**
    1. Export the telematics data used in your notebook as CSV or JSON.
    2. Upload it in the sidebar. For JSON from the notebook, records nested under `messages` are supported.
    3. Optionally upload CAN-bus data. The dashboard will identify engine-speed anomalies using the notebook's 500–2500 RPM thresholds.
    """)
    st.stop()

df = clean_telematics(tele_raw)
if df.empty:
    st.error("The uploaded telematics file contains no usable records.")
    st.stop()

# Build the notebook-aligned health labels.
if "vehicleBatteryVoltage" in df.columns:
    df["vehicleBatteryVoltage"] = pd.to_numeric(df["vehicleBatteryVoltage"], errors="coerce")
    df["vhi_rule"] = df["vehicleBatteryVoltage"].apply(vhi_from_voltage)
    df["health_status"] = df["vehicleBatteryVoltage"].apply(health_label)
    df["is_fault_rule"] = (df["vehicleBatteryVoltage"] < 24.0).astype("Int64")

# Fit models once per uploaded file, cache in session state based on data signature.
signature = (tele_file.name, len(df), tuple(df.columns))
if st.session_state.get("data_signature") != signature:
    with st.spinner("Training Random Forest models on the uploaded telematics data..."):
        reg_result, cls_result, feature_cols, train_error = fit_models(df)
    st.session_state["data_signature"] = signature
    st.session_state["reg_result"] = reg_result
    st.session_state["cls_result"] = cls_result
    st.session_state["feature_cols"] = feature_cols
    st.session_state["train_error"] = train_error

reg_result = st.session_state.get("reg_result")
cls_result = st.session_state.get("cls_result")
feature_cols = st.session_state.get("feature_cols", [])
train_error = st.session_state.get("train_error")

# Vehicle filter
vehicle_col = "vehicleId" if "vehicleId" in df.columns else ("vin" if "vin" in df.columns else None)
with st.sidebar:
    if vehicle_col and df[vehicle_col].nunique(dropna=True) > 1:
        vehicles = sorted(df[vehicle_col].dropna().astype(str).unique().tolist())
        selected_vehicle = st.selectbox("Select vehicle", ["All vehicles"] + vehicles)
    else:
        selected_vehicle = "All vehicles"

view = df.copy()
if vehicle_col and selected_vehicle != "All vehicles":
    view = view[view[vehicle_col].astype(str) == selected_vehicle].copy()

# ------------------------- Navigation -------------------------
page = st.sidebar.radio(
    "Dashboard sections",
    ["Overview", "Telematics analytics", "CAN diagnostics", "Predict a vehicle", "Data & model details"],
)

# Latest observation for selected vehicle
latest = view.sort_values("eventDateTime").iloc[-1] if "eventDateTime" in view.columns else view.iloc[-1]
latest_voltage = latest.get("vehicleBatteryVoltage", np.nan)
latest_health = health_label(latest_voltage) if pd.notna(latest_voltage) else "Unknown"
latest_vhi_rule = vhi_from_voltage(latest_voltage)

# ------------------------- Overview -------------------------
if page == "Overview":
    st.subheader("Fleet health overview")
    st.caption(f"Selected scope: {selected_vehicle} • {len(view):,} telematics records")

    healthy_count = int((view.get("health_status", pd.Series(index=view.index, dtype=object)) == "Healthy").sum())
    critical_count = int((view.get("health_status", pd.Series(index=view.index, dtype=object)) == "Critical").sum())
    fault_count = int((view.get("is_fault_rule", pd.Series(index=view.index, dtype=float)) == 1).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest rule-based VHI", f"{latest_vhi_rule:.1f}/100" if pd.notna(latest_vhi_rule) else "N/A")
    k2.metric("Latest health status", f"{metric_color(latest_health)} {latest_health}")
    k3.metric("Records below 24 V", f"{fault_count:,}", f"{(fault_count/max(len(view),1))*100:.1f}% of records")
    k4.metric("Vehicles in scope", f"{view[vehicle_col].nunique() if vehicle_col else 1:,}")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Battery voltage trend")
        if "eventDateTime" in view.columns and "vehicleBatteryVoltage" in view.columns:
            chart_df = view.dropna(subset=["eventDateTime", "vehicleBatteryVoltage"]).sort_values("eventDateTime")
            fig = px.line(chart_df, x="eventDateTime", y="vehicleBatteryVoltage",
                          color=vehicle_col if vehicle_col and selected_vehicle == "All vehicles" else None,
                          labels={"eventDateTime": "Timestamp", "vehicleBatteryVoltage": "Battery voltage (V)"},
                          title="Vehicle battery voltage over time")
            fig.add_hline(y=25.0, line_dash="dot", annotation_text="Healthy threshold (25 V)")
            fig.add_hline(y=24.0, line_dash="dash", annotation_text="Critical threshold (24 V)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("eventDateTime and vehicleBatteryVoltage are required for the trend chart.")
    with right:
        st.markdown("#### Health-status distribution")
        if "health_status" in view.columns:
            counts = view["health_status"].value_counts().rename_axis("Health status").reset_index(name="Records")
            fig = px.pie(counts, names="Health status", values="Records", hole=0.55)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Battery-voltage field is missing; status distribution is unavailable.")

    if reg_result and reg_result[0] is not None:
        st.markdown("#### Model-estimated vehicle health")
        X_latest = pd.DataFrame([{c: pd.to_numeric(latest.get(c), errors="coerce") for c in feature_cols}])
        predicted_vhi = float(np.clip(reg_result[0].predict(X_latest)[0], 0, 100))
        a, b = st.columns([1, 2])
        with a:
            st.metric("Random Forest predicted VHI", f"{predicted_vhi:.1f}/100")
        with b:
            st.progress(int(round(predicted_vhi)), text="Model-estimated VHI")
    if train_error:
        st.warning(f"Model training note: {train_error}")

    st.markdown("#### Recent records")
    display_cols = [c for c in [vehicle_col, "eventDateTime", "vehicleBatteryVoltage", "vhi_rule", "health_status", "speed", "odometer"] if c and c in view.columns]
    st.dataframe(view[display_cols].tail(15), use_container_width=True, hide_index=True)

# ------------------------- Telematics analytics -------------------------
elif page == "Telematics analytics":
    st.subheader("Telematics analytics")
    numeric_cols = view.select_dtypes(include=np.number).columns.tolist()
    preferred = [c for c in ["vehicleBatteryVoltage", "backupBatteryVoltage", "speed", "accelX", "accelY", "accelZ", "odometer", "signalStrength", "gpsSignalQuality"] if c in numeric_cols]
    if preferred:
        chosen = st.selectbox("Select signal to analyse", preferred)
        if "eventDateTime" in view.columns:
            plot_df = view.dropna(subset=["eventDateTime", chosen]).sort_values("eventDateTime")
            st.plotly_chart(px.line(plot_df, x="eventDateTime", y=chosen, title=f"{chosen} over time"), use_container_width=True)
        st.plotly_chart(px.histogram(view, x=chosen, nbins=40, title=f"Distribution of {chosen}"), use_container_width=True)
        st.metric(f"Average {chosen}", f"{view[chosen].mean():.3f}")
    else:
        st.warning("No numeric telematics signals found.")

    st.markdown("#### Correlation between numeric signals")
    corr_cols = [c for c in numeric_cols if view[c].nunique(dropna=True) > 1][:20]
    if len(corr_cols) >= 2:
        corr = view[corr_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                        title="Numeric feature correlation matrix")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------- CAN diagnostics -------------------------
elif page == "CAN diagnostics":
    st.subheader("CAN-bus diagnostics")
    if can_raw is None:
        st.info("Upload a CAN-bus CSV/JSON file in the sidebar to view CAN diagnostics.")
    else:
        can = can_raw.copy()
        if {"spnName", "spnValue"}.issubset(can.columns):
            can["spnValue"] = pd.to_numeric(can["spnValue"], errors="coerce")
            idx = [c for c in ["vin", "timestamp"] if c in can.columns]
            if idx:
                can = can.pivot_table(index=idx, columns="spnName", values="spnValue", aggfunc="mean").reset_index()
                can.columns.name = None
        if "EngSpeed" in can.columns:
            can["EngSpeed"] = pd.to_numeric(can["EngSpeed"], errors="coerce")
            can["CAN health status"] = np.select(
                [can["EngSpeed"] < 500, can["EngSpeed"] > 2500],
                ["Low RPM Anomaly", "High RPM Anomaly"],
                default="Normal Operation",
            )
            x1, x2, x3 = st.columns(3)
            x1.metric("CAN records", f"{len(can):,}")
            x2.metric("RPM anomalies", f"{(can['CAN health status'] != 'Normal Operation').sum():,}")
            x3.metric("Average engine speed", f"{can['EngSpeed'].mean():,.0f} RPM")
            st.plotly_chart(px.line(can.reset_index(), y="EngSpeed", title="Engine speed (RPM)"), use_container_width=True)
            st.plotly_chart(px.histogram(can, x="CAN health status", title="CAN health-status distribution"), use_container_width=True)
            st.dataframe(can.tail(20), use_container_width=True, hide_index=True)
            st.caption("Notebook rule: below 500 RPM = Low RPM Anomaly; above 2500 RPM = High RPM Anomaly. Validate thresholds against engine operating specifications.")
        else:
            st.warning("CAN file loaded, but EngSpeed could not be found. For long-format CAN data, provide spnName and spnValue columns with EngSpeed as an SPN name.")

# ------------------------- Prediction -------------------------
elif page == "Predict a vehicle":
    st.subheader("Vehicle health prediction")
    st.write("Enter the latest numeric sensor readings. The trained Random Forest models use the uploaded telematics dataset.")
    if not feature_cols or not reg_result or reg_result[0] is None:
        st.error(train_error or "Model is not ready. Upload a suitable telematics file with vehicleBatteryVoltage and additional numeric signals.")
    else:
        defaults = {}
        for col in feature_cols:
            val = pd.to_numeric(latest.get(col), errors="coerce")
            defaults[col] = float(val) if pd.notna(val) else float(view[col].median())
        with st.form("prediction_form"):
            st.markdown("##### Sensor inputs")
            cols = st.columns(3)
            values = {}
            for i, col in enumerate(feature_cols):
                with cols[i % 3]:
                    values[col] = st.number_input(col, value=defaults[col], format="%.4f", key=f"pred_{col}")
            submitted = st.form_submit_button("Predict vehicle health", type="primary")
        if submitted:
            input_df = pd.DataFrame([values], columns=feature_cols)
            pred_vhi = float(np.clip(reg_result[0].predict(input_df)[0], 0, 100))
            status = ("Healthy" if pred_vhi >= 85 else "Normal degradation" if pred_vhi >= 60 else "Attention required" if pred_vhi >= 30 else "Critical")
            p1, p2 = st.columns(2)
            p1.metric("Predicted Vehicle Health Index", f"{pred_vhi:.1f}/100")
            p2.metric("Predicted health band", f"{metric_color(status)} {status}")
            st.progress(int(round(pred_vhi)), text="Predicted VHI")
            if cls_result and cls_result[0] is not None:
                fault_probability = float(cls_result[0].predict_proba(input_df)[0][1])
                st.metric("Predicted fault probability", f"{fault_probability*100:.1f}%")
                st.caption("Fault label is defined in the notebook as battery voltage below 24.0 V; the model estimates this label from the other available numeric signals.")
            st.warning("This is a research prototype trained on the uploaded dataset, not a certified vehicle diagnostic system. Confirm alerts using engineering limits and physical inspection.")

# ------------------------- Data and model details -------------------------
elif page == "Data & model details":
    st.subheader("Data quality and model information")
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows", f"{len(df):,}")
    q2.metric("Columns", f"{df.shape[1]:,}")
    q3.metric("Duplicate rows after cleaning", f"{df.duplicated().sum():,}")

    st.markdown("#### Missing values")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        st.dataframe(missing.rename("Missing count").to_frame(), use_container_width=True)
    else:
        st.success("No missing values detected.")

    st.markdown("#### Regression model: Vehicle Health Index")
    st.caption("Random Forest Regressor; VHI target follows the notebook's piecewise mapping from vehicleBatteryVoltage. Battery voltage is excluded from predictors to reduce direct target leakage.")
    if reg_result and reg_result[1]:
        st.dataframe(pd.DataFrame([reg_result[1]]), use_container_width=True)
    st.markdown("#### Fault classification")
    st.caption("Random Forest Classifier; fault label = 1 when vehicleBatteryVoltage < 24.0 V. Battery voltage is excluded from model predictors.")
    if cls_result and cls_result[1]:
        st.dataframe(pd.DataFrame([cls_result[1]]), use_container_width=True)
    elif cls_result and cls_result[0] is None:
        st.info("Classification model not trained: the uploaded file may contain only one class under the 24.0 V rule, or too few examples.")
    st.markdown("#### Features used by the models")
    st.write(feature_cols if feature_cols else "No model features available.")
    st.download_button(
        "Download enriched telematics data",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="vehicle_health_enriched.csv",
        mime="text/csv",
    )
    st.warning("Interpret model scores cautiously. Labels are rule-derived from battery voltage and engine RPM in the supplied notebook; high scores do not independently prove real-world failure prediction.")
