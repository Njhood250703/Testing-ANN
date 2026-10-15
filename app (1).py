"""
Solar PV Power Output Prediction using ANN (Artificial Neural Network)
Streamlit GUI - suitable for PV projects (e.g. 605 W module).

Run:  streamlit run app.py
"""

import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar PV ANN Predictor",
    page_icon="☀️",
    layout="wide",
)

DEMO_FEATURES = [
    "Solar Irradiance (W/m2)",
    "PV Temperature (C)",
    "Ambient Temperature (C)",
]
DEMO_TARGET = "P_output (W)"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def generate_demo_data(n_samples: int, p_rated: float, temp_coeff: float,
                       seed: int) -> pd.DataFrame:
    """Generate simulated PV data (for demo / system testing ONLY).

    Replace it with your real field data by uploading a CSV.
    """
    rng = np.random.default_rng(seed)

    hour = rng.uniform(7, 19, n_samples)
    sun = np.clip(np.sin(np.pi * (hour - 7) / 12), 0, None)
    cloud = rng.uniform(0.35, 1.0, n_samples)

    irradiance = np.clip(1050 * sun * cloud + rng.normal(0, 15, n_samples), 0, None)
    t_amb = 26 + 7 * sun + rng.normal(0, 1.2, n_samples)
    t_pv = t_amb + irradiance / 800 * 25 + rng.normal(0, 1.0, n_samples)

    power = p_rated * (irradiance / 1000) * (1 + temp_coeff * (t_pv - 25))
    power = power * (1 - 0.05 * np.exp(-irradiance / 150))  # low-irradiance loss (non-linear)
    power = np.clip(power + rng.normal(0, 4, n_samples), 0, None)

    return pd.DataFrame({
        "Solar Irradiance (W/m2)": irradiance,
        "PV Temperature (C)": t_pv,
        "Ambient Temperature (C)": t_amb,
        "P_output (W)": power,
    })


# ----------------------------------------------------------------------------
# Model ANN
# ----------------------------------------------------------------------------
def train_ann(X_train, y_train, hidden_layers, params):
    """Train an ANN (MLP) with input & output scaling."""
    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    model = MLPRegressor(hidden_layer_sizes=hidden_layers, **params)
    model.fit(
        scaler_x.transform(X_train),
        scaler_y.transform(y_train.reshape(-1, 1)).ravel(),
    )
    return model, scaler_x, scaler_y


def predict_ann(model, scaler_x, scaler_y, X):
    pred = model.predict(scaler_x.transform(X))
    return scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()


def compute_metrics(y_true, y_pred):
    mask = y_true > 10  # avoid division by near-zero values
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.any() else np.nan
    )
    return {
        "R²": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE (%)": mape,
    }


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def draw_architecture(n_inputs, hidden_layers, max_nodes=8):
    sizes = [n_inputs, *hidden_layers, 1]
    labels = ["Input"] + [f"Hidden {i + 1}" for i in range(len(hidden_layers))] + ["Output"]

    fig, ax = plt.subplots(figsize=(1.8 * len(sizes) + 2, 4))
    positions = []
    for x, size in enumerate(sizes):
        shown = min(size, max_nodes)
        ys = np.linspace(-(shown - 1) / 2, (shown - 1) / 2, shown)
        positions.append([(x, y) for y in ys])

    for a, b in zip(positions[:-1], positions[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color="#bbbbbb", lw=0.4, zorder=1)

    colors = ["#2a9d8f"] + ["#f4a261"] * len(hidden_layers) + ["#e63946"]
    for pos, c, size in zip(positions, colors, sizes):
        ax.scatter(*zip(*pos), s=180, color=c, zorder=2, edgecolor="white")
        if size > max_nodes:
            ax.text(pos[0][0], min(p[1] for p in pos) - 0.9, "⋮", ha="center", fontsize=14)

    for x, (label, size) in enumerate(zip(labels, sizes)):
        ax.text(x, max(p[1] for p in positions[x]) + 1.0, f"{label}\n({size})",
                ha="center", fontsize=9)

    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_results(y_true, y_pred, model):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].scatter(y_true, y_pred, s=10, alpha=0.5, color="#2a9d8f")
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set(title="Actual vs Predicted", xlabel="Actual (W)", ylabel="Predicted (W)")

    residual = y_true - y_pred
    axes[1].hist(residual, bins=30, color="#f4a261", edgecolor="white")
    axes[1].set(title="Error Distribution (Residual)", xlabel="Error (W)", ylabel="Count")

    if hasattr(model, "loss_curve_") and len(model.loss_curve_) > 0:
        axes[2].plot(model.loss_curve_, color="#e63946")
        axes[2].set(title="Training Loss Curve", xlabel="Epoch", ylabel="Loss")
    else:
        axes[2].text(0.5, 0.5, "No loss curve\n(lbfgs solver)", ha="center", va="center")
        axes[2].axis("off")

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("☀️ PV Power Output Prediction using ANN")
st.caption(
    "A GUI system to train an Artificial Neural Network (MLP) that predicts "
    "solar panel power output from environmental data."
)

# ----------------------------------------------------------------------------
# Sidebar: settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("1. Data")
    source = st.radio("Data source", ["Simulated data (demo)", "Upload CSV"])

    st.subheader("2. PV Module Specifications")
    p_rated = st.number_input("Rated power, Pmax (W)", 100.0, 1000.0, 605.0, 5.0)
    temp_coeff = st.number_input("Pmax temperature coefficient (/°C)", -0.01, 0.0, -0.0029,
                                 0.0001, format="%.4f")

    if source == "Simulated data (demo)":
        n_samples = st.slider("Number of samples", 200, 5000, 1500, 100)
        seed = st.number_input("Random seed (data)", 0, 9999, 42)

    st.subheader("3. ANN Architecture")
    n_layers = st.slider("Number of hidden layers", 1, 5, 2)
    hidden_layers = []
    for i in range(n_layers):
        default = max(8, 64 // (2 ** i))
        hidden_layers.append(
            st.slider(f"Neurons - Hidden layer {i + 1}", 2, 256, default, 2)
        )
    hidden_layers = tuple(hidden_layers)

    activation = st.selectbox("Activation function", ["relu", "tanh", "logistic"])
    solver = st.selectbox("Solver / optimizer", ["adam", "lbfgs", "sgd"])
    learning_rate = st.select_slider(
        "Learning rate", [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], value=0.001
    )
    max_iter = st.slider("Maximum epochs", 50, 3000, 500, 50)
    alpha = st.select_slider("L2 regularization (alpha)", [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                             value=1e-4)
    early_stop = st.checkbox("Early stopping", value=True)

    st.subheader("4. Training & Targets")
    test_size = st.slider("Test data ratio", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random seed (model)", 0, 9999, 42)
    target_r2 = st.slider("Minimum R² target", 0.80, 0.999, 0.95, 0.001)
    target_mape = st.slider("Maximum MAPE target (%)", 1.0, 20.0, 5.0, 0.5)

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
if source == "Simulated data (demo)":
    df = generate_demo_data(n_samples, p_rated, temp_coeff, int(seed))
    feature_cols = st.multiselect("Input variables (features)", list(df.columns[:-1]),
                                  default=DEMO_FEATURES)
    target_col = DEMO_TARGET
    st.info("You are using simulated data for demo purposes. "
            "Upload a CSV of real field data for your project results.")
else:
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is None:
        st.warning("Please upload a CSV file or switch to simulated data in the sidebar.")
        st.stop()
    df = pd.read_csv(uploaded)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        st.error("The CSV must contain at least 2 numeric columns.")
        st.stop()
    df = df[num_cols].dropna().reset_index(drop=True)
    c1, c2 = st.columns(2)
    target_col = c1.selectbox("Target column (Power output)", num_cols, index=len(num_cols) - 1)
    options = [c for c in num_cols if c != target_col]
    feature_cols = c2.multiselect("Input variables (features)", options, default=options)

if not feature_cols:
    st.warning("Select at least one input variable.")
    st.stop()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_data, tab_train, tab_predict, tab_export = st.tabs(
    ["📊 Data", "🧠 Training & Results", "🔮 Prediction", "💾 Export"]
)

with tab_data:
    st.subheader("Data preview")
    st.dataframe(df.head(100), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Number of samples", f"{len(df):,}")
    c2.metric("Number of inputs", len(feature_cols))
    c3.metric("Average power output", f"{df[target_col].mean():.1f} W")
    st.subheader("Summary statistics")
    st.dataframe(df[[*feature_cols, target_col]].describe().T, width="stretch")
    st.subheader("Correlation with power output")
    corr = df[[*feature_cols, target_col]].corr()[target_col].drop(target_col)
    st.bar_chart(corr)

with tab_train:
    st.subheader("Network architecture")
    st.pyplot(draw_architecture(len(feature_cols), hidden_layers))
    n_params = sum(
        a * b + b for a, b in zip((len(feature_cols), *hidden_layers), (*hidden_layers, 1))
    )
    st.caption(
        f"Structure: {len(feature_cols)} → "
        + " → ".join(map(str, hidden_layers))
        + f" → 1  |  Total parameters: {n_params:,}"
    )

    if st.button("🚀 Train ANN model", type="primary"):
        X = df[feature_cols].to_numpy()
        y = df[target_col].to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        params = dict(
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=int(random_state),
        )
        if solver != "lbfgs":
            params.update(learning_rate_init=learning_rate, early_stopping=early_stop)

        with st.spinner("Training model..."):
            model, sx, sy = train_ann(X_tr, y_tr, hidden_layers, params)

        st.session_state["result"] = {
            "model": model, "sx": sx, "sy": sy,
            "features": feature_cols, "target": target_col,
            "y_tr": y_tr, "p_tr": predict_ann(model, sx, sy, X_tr),
            "y_te": y_te, "p_te": predict_ann(model, sx, sy, X_te),
            "X_te": X_te, "p_rated": p_rated,
            "stats": {c: (float(df[c].min()), float(df[c].max()), float(df[c].median()))
                      for c in feature_cols},
        }

    res = st.session_state.get("result")
    if res is None:
        st.info("Press the button above to train the model.")
    else:
        m_tr = compute_metrics(res["y_tr"], res["p_tr"])
        m_te = compute_metrics(res["y_te"], res["p_te"])

        st.subheader("Model performance")
        metrics_df = pd.DataFrame({"Train": m_tr, "Test": m_te}).T
        st.dataframe(metrics_df.style.format("{:.4f}"), width="stretch")

        k1, k2, k3 = st.columns(3)
        k1.metric("R² (test)", f"{m_te['R²']:.4f}")
        k2.metric("RMSE (test)", f"{m_te['RMSE']:.2f} W")
        k3.metric("MAPE (test)", f"{m_te['MAPE (%)']:.2f} %")

        if m_te["R²"] >= target_r2 and m_te["MAPE (%)"] <= target_mape:
            st.success(f"✅ Targets achieved (R² ≥ {target_r2}, MAPE ≤ {target_mape}%).")
        else:
            st.error(
                f"❌ Targets not achieved (R² ≥ {target_r2}, MAPE ≤ {target_mape}%). "
                "Try adding neurons/layers, changing the learning rate, or adding more data."
            )

        gap = m_tr["R²"] - m_te["R²"]
        if gap > 0.05:
            st.warning("The gap between training and test R² is large – possible overfitting. "
                       "Try increasing alpha or reducing the network size.")

        st.pyplot(plot_results(res["y_te"], res["p_te"], res["model"]))

with tab_predict:
    res = st.session_state.get("result")
    if res is None:
        st.info("Train a model first in the 'Training & Results' tab.")
    else:
        st.subheader("Power output prediction (manual input)")
        cols = st.columns(3)
        values = []
        for i, name in enumerate(res["features"]):
            lo, hi, med = res["stats"][name]
            values.append(
                cols[i % 3].number_input(name, value=med, min_value=lo - abs(hi - lo),
                                         max_value=hi + abs(hi - lo), key=f"in_{name}")
            )

        if st.button("Predict power output"):
            p = predict_ann(res["model"], res["sx"], res["sy"], np.array([values]))[0]
            p = max(p, 0.0)
            r1, r2 = st.columns(2)
            r1.metric("Predicted power output", f"{p:.1f} W")
            r2.metric("Percentage of Pmax", f"{p / res['p_rated'] * 100:.1f} %")

        st.divider()
        st.subheader("Batch prediction (CSV)")
        batch = st.file_uploader("CSV with the same input columns", type=["csv"], key="batch")
        if batch is not None:
            bdf = pd.read_csv(batch)
            missing = [c for c in res["features"] if c not in bdf.columns]
            if missing:
                st.error(f"Columns missing from CSV: {missing}")
            else:
                bdf["P_predicted (W)"] = np.clip(
                    predict_ann(res["model"], res["sx"], res["sy"],
                                bdf[res["features"]].to_numpy()), 0, None)
                st.dataframe(bdf.head(100), width="stretch")
                st.download_button("⬇️ Download predictions (CSV)",
                                   bdf.to_csv(index=False).encode(),
                                   "batch_predictions.csv", "text/csv")

with tab_export:
    res = st.session_state.get("result")
    if res is None:
        st.info("Train a model first to export.")
    else:
        pred_df = pd.DataFrame(res["X_te"], columns=res["features"])
        pred_df["P_actual (W)"] = res["y_te"]
        pred_df["P_predicted (W)"] = res["p_te"]
        pred_df["Error (W)"] = pred_df["P_actual (W)"] - pred_df["P_predicted (W)"]

        st.download_button("⬇️ Test results (CSV)", pred_df.to_csv(index=False).encode(),
                           "test_results.csv", "text/csv")

        buf = io.BytesIO()
        joblib.dump({k: res[k] for k in ("model", "sx", "sy", "features", "target")}, buf)
        st.download_button("⬇️ Model ANN (.joblib)", buf.getvalue(),
                           "ann_solar_model.joblib", "application/octet-stream")
