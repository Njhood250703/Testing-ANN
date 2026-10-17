import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="ANN PV Power Prediction", layout="wide")

FEATURES = ["irradiance", "ambient_temp", "pv_rating", "pv_temp"]
TARGET = "power"
PV_RATING = 605.0  # W (fixed, single module)


# ---------------- Data ----------------
def demo_data(n=500):
    """Synthetic demo data. Replace with measured site data (upload CSV)."""
    rng = np.random.default_rng(42)
    irr = rng.uniform(100, 1000, n)
    amb = rng.uniform(25, 38, n)
    pvt = amb + irr * 0.03 + rng.normal(0, 1, n)
    power = 0.6 * irr * (1 - 0.004 * (pvt - 25)) + rng.normal(0, 8, n)
    return pd.DataFrame({"irradiance": irr, "ambient_temp": amb,
                         "pv_rating": PV_RATING, "pv_temp": pvt, "power": power})


@st.cache_resource(show_spinner="Training ANN...")
def train_model(df, n1, n2, test_size):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = MLPRegressor(hidden_layer_sizes=(n1, n2), activation="relu",
                         max_iter=2000, early_stopping=True, random_state=42)
    model.fit(X_tr_s, y_tr)
    pred_te = model.predict(X_te_s)
    pred_tr = model.predict(X_tr_s)
    metrics = {
        "r2_train": r2_score(y_tr, pred_tr),
        "r2_test": r2_score(y_te, pred_te),
        "rmse_test": float(np.sqrt(mean_squared_error(y_te, pred_te))),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    return model, scaler, metrics, y_te.values, pred_te


# ---------------- Sidebar: data ----------------
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload measured data (CSV)", type="csv")
st.sidebar.caption("Columns: " + ", ".join(FEATURES + [TARGET])
                   + ". If pv_rating is missing, 605 W is used.")
test_size = st.sidebar.slider("Test data (%)", 10, 40, 20) / 100

if up is not None:
    df = pd.read_csv(up)
    if "pv_rating" not in df.columns:
        df["pv_rating"] = PV_RATING
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        st.stop()
    df = df[FEATURES + [TARGET]].dropna()
    data_label = f"Measured data (uploaded, {len(df)} rows)"
else:
    df = demo_data()
    data_label = "DEMO data (synthetic)"

# ---------------- Main page (single section) ----------------
st.title("ANN PV Power Prediction")
st.caption(f"Bifacial PV module, {PV_RATING:.0f} W rating  |  Data: {data_label}")
if up is None:
    st.warning("Using synthetic DEMO data. Results are not real. "
               "Upload your measured site data (CSV) in the sidebar.")

# ===== 1. INPUT =====
st.subheader("1. Input")
c1, c2 = st.columns(2)
with c1:
    irr = st.number_input("Irradiance (W/m²)", 0.0, 1500.0, 800.0, 10.0)
    amb = st.number_input("Ambient temperature (°C)", 0.0, 60.0, 32.0, 0.5)
with c2:
    pvt = st.number_input("PV temperature (°C)", 0.0, 100.0, 55.0, 0.5)
    st.number_input("PV rating (W)", value=PV_RATING, disabled=True)

st.divider()

# ===== 2. HIDDEN LAYER =====
st.subheader("2. Hidden layer")
h1, h2 = st.columns(2)
n1 = h1.slider("Hidden layer 1 (neurons)", 2, 30, 10)
n2 = h2.slider("Hidden layer 2 (neurons)", 2, 30, 10)
st.caption(f"ANN structure: 4 inputs  →  {n1} neurons  →  {n2} neurons  →  1 output "
           "(activation: ReLU)")

model, scaler, m, y_te, pred_te = train_model(df, n1, n2, test_size)

st.divider()

# ===== 3. OUTPUT ANN =====
st.subheader("3. Output ANN")
x_new = pd.DataFrame([[irr, amb, PV_RATING, pvt]], columns=FEATURES)
p = max(float(model.predict(scaler.transform(x_new))[0]), 0.0)  # no negative power
o1, o2 = st.columns(2)
o1.metric("Predicted power", f"{p:.1f} W")
o2.metric("Percent of rating", f"{p / PV_RATING * 100:.1f} %")
lo_i, hi_i = df["irradiance"].min(), df["irradiance"].max()
if not (lo_i <= irr <= hi_i):
    st.info(f"Irradiance is outside the training range ({lo_i:.0f} to {hi_i:.0f} "
            "W/m²). Prediction may be unreliable.")

st.divider()

# ===== 4. TARGET OUTPUT (MEASURED VALUE) =====
st.subheader("4. Target output (measured value)")
meas = st.number_input("Measured power from PV panel (W)", 0.0, 1000.0, 0.0, 1.0,
                       help="Enter the measured value at the same time as the "
                            "inputs above. Leave 0 to skip.")
if meas > 0:
    err = meas - p
    t1, t2 = st.columns(2)
    t1.metric("Error (measured - ANN)", f"{err:.1f} W")
    t2.metric("Error (%)", f"{abs(err) / meas * 100:.1f} %")

st.markdown("**ANN performance on test data (measured vs ANN)**")
a, b, c = st.columns(3)
a.metric("R² (test)", f"{m['r2_test']:.4f}")
b.metric("RMSE (test)", f"{m['rmse_test']:.2f} W")
c.metric("R² (train)", f"{m['r2_train']:.4f}")
st.caption(f"Train samples: {m['n_train']}  |  Test samples: {m['n_test']}. "
           "R² train and test should be close. A large gap means overfitting.")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_te, pred_te, s=15)
lo, hi = float(y_te.min()), float(y_te.max())
ax.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
ax.set_xlabel("Measured power (W)")
ax.set_ylabel("ANN predicted power (W)")
ax.set_title("Actual vs Predicted (test data)")
ax.legend()
st.pyplot(fig)

with st.expander("Show data"):
    st.write(f"{len(df)} rows")
    st.dataframe(df.head(100))
