"""
Solar PV Power Output Prediction using ANN (v3)
===============================================
Measured power output       = the TARGET the ANN must learn.
Computational power output  = calculated from the inputs (SEDA power equation).
The ANN output should be close to the measured target and, as far as the data
allows, close to the computational value too.

Run:  streamlit run app_v3.py
"""

import io
import warnings

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
st.set_page_config(page_title="Solar PV ANN Predictor (v3)", page_icon="☀️", layout="wide")

COMP_NAME = "P_computational (W)"


# ----------------------------------------------------------------------------
# Computational power (SEDA power equation, same as the database app)
# ----------------------------------------------------------------------------
def computational_power(irradiance, pv_temp, cfg):
    """Computational P_max (W) = SEDA Eq 5.7 - 5.10:

    P = Pstc x (1 + bifacial gain) x f_mm x f_degrad x f_temp x f_g x f_clean x f_unshade
    """
    g = np.asarray(irradiance, float)
    t = np.asarray(pv_temp, float)
    f_g = g / 1000                                         # Eq 5.9
    f_temp = 1 + cfg["gamma"] / 100 * (t - 25.0)           # Eq 5.10
    f_degrad = cfg["f_lid"] * cfg["f_age"]                 # Eq 5.8
    f_clean = (100 - cfg["dirt"]) / 100
    f_unshade = (100 - cfg["shading"]) / 100
    p = (cfg["p_stc"] * (1 + cfg["bifacial"] / 100) * cfg["f_mm"] * f_degrad
         * f_temp * f_g * f_clean * f_unshade)
    return np.clip(p, 0, None)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def generate_demo_data(n_samples: int, p_stc: float, gamma: float, seed: int) -> pd.DataFrame:
    """Simulated field data (for demo / system testing ONLY).

    Measured power = ideal power minus real-world losses (soiling, mismatch,
    low-light loss, noise), so it sits a few percent below the computational value.
    """
    rng = np.random.default_rng(seed)
    hour = rng.uniform(7, 19, n_samples)
    sun = np.clip(np.sin(np.pi * (hour - 7) / 12), 0, None)
    cloud = rng.uniform(0.35, 1.0, n_samples)

    irr = np.clip(1050 * sun * cloud + rng.normal(0, 15, n_samples), 0, None)
    t_amb = 26 + 7 * sun + rng.normal(0, 1.2, n_samples)
    t_pv = t_amb + irr / 800 * 25 + rng.normal(0, 1.0, n_samples)

    ideal = p_stc * (irr / 1000) * (1 + gamma / 100 * (t_pv - 25))
    meas = ideal * 0.93 * (1 - 0.05 * np.exp(-irr / 150)) * (1 + rng.normal(0, 0.01, n_samples))
    meas = np.clip(meas + rng.normal(0, 3, n_samples), 0, None)

    return pd.DataFrame({
        "Solar Irradiance (W/m2)": irr,
        "PV Temperature (C)": t_pv,
        "Ambient Temperature (C)": t_amb,
        "P_measured (W)": meas,
    })


# ----------------------------------------------------------------------------
# Model ANN
# ----------------------------------------------------------------------------
def train_ann(X_train, y_train, hidden_layers, params):
    """Train an ANN (MLP) with input & output scaling."""
    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    model = MLPRegressor(hidden_layer_sizes=hidden_layers, **params)
    model.fit(scaler_x.transform(X_train),
              scaler_y.transform(y_train.reshape(-1, 1)).ravel())
    return model, scaler_x, scaler_y


def predict_ann(model, scaler_x, scaler_y, X):
    pred = model.predict(scaler_x.transform(X))
    return scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true > 10  # avoid division by near-zero values
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.any() else np.nan)
    return {"R²": r2_score(y_true, y_pred),
            "RMSE (W)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE (W)": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": mape}


# ----------------------------------------------------------------------------
# Plots
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


def plot_results(res):
    meas, comp, ann = res["meas_te"], res["comp_te"], res["p_te"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    lim = [min(meas.min(), comp.min(), ann.min()), max(meas.max(), comp.max(), ann.max())]
    axes[0, 0].scatter(meas, comp, s=10, alpha=0.35, color="#f4a261", label="Computational")
    axes[0, 0].scatter(meas, ann, s=10, alpha=0.6, color="#2a9d8f", label="ANN")
    axes[0, 0].plot(lim, lim, "r--", lw=1, label="Ideal (= measured)")
    axes[0, 0].set(title="Predicted vs Measured (test set)",
                   xlabel="Measured power - target (W)", ylabel="Predicted power (W)")
    axes[0, 0].legend()

    axes[0, 1].hist(meas - ann, bins=30, alpha=0.7, color="#2a9d8f", label="Measured − ANN")
    axes[0, 1].hist(meas - comp, bins=30, alpha=0.5, color="#f4a261", label="Measured − Computational")
    axes[0, 1].set(title="Error distribution", xlabel="Error (W)", ylabel="Count")
    axes[0, 1].legend()

    if res["loss_curve"]:
        axes[1, 0].plot(res["loss_curve"], color="#e63946")
        axes[1, 0].set(title="Training loss curve", xlabel="Epoch", ylabel="Loss")
    else:
        axes[1, 0].text(0.5, 0.5, "No loss curve\n(lbfgs solver)", ha="center", va="center")
        axes[1, 0].axis("off")

    order = np.argsort(comp)
    x = np.arange(len(order))
    axes[1, 1].plot(x, meas[order], label="Measured (target)", color="#e76f51", lw=1.0, alpha=0.8)
    axes[1, 1].plot(x, comp[order], label="Computational", color="#264653", lw=1.2)
    axes[1, 1].plot(x, ann[order], label="ANN", color="#2a9d8f", lw=1.2)
    axes[1, 1].set(title="Measured vs Computational vs ANN (test set, sorted)",
                   xlabel="Test sample (sorted by computational)", ylabel="Power (W)")
    axes[1, 1].legend()
    fig.tight_layout()
    return fig


def show_results(res):
    """Accuracy against the measured target AND the computational value."""
    m_meas = compute_metrics(res["meas_te"], res["p_te"])
    m_comp = compute_metrics(res["comp_te"], res["p_te"])
    base = compute_metrics(res["meas_te"], res["comp_te"])

    st.subheader("Mean power (test set)")
    a1, a2, a3 = st.columns(3)
    a1.metric("Measured (target)", f"{res['meas_te'].mean():.1f} W")
    a2.metric("Computational", f"{res['comp_te'].mean():.1f} W")
    a3.metric("ANN", f"{res['p_te'].mean():.1f} W")

    st.subheader("1) ANN vs Measured (target)")
    k1, k2, k3 = st.columns(3)
    k1.metric("R²", f"{m_meas['R²']:.4f}")
    k2.metric("RMSE", f"{m_meas['RMSE (W)']:.2f} W")
    k3.metric("MAPE", f"{m_meas['MAPE (%)']:.2f} %")

    st.subheader("2) ANN vs Computational")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{m_comp['R²']:.4f}")
    c2.metric("RMSE", f"{m_comp['RMSE (W)']:.2f} W")
    c3.metric("MAPE", f"{m_comp['MAPE (%)']:.2f} %")

    near_t = m_meas["R²"] >= res["target_r2"] and m_meas["MAPE (%)"] <= res["target_mape"]
    near_c = m_comp["MAPE (%)"] <= res["target_mape_comp"]
    if near_t and near_c:
        st.success("✅ The ANN is close to BOTH the measured target and the computational value.")
    elif near_t:
        st.success(
            f"✅ The ANN is close to the measured target (MAPE {m_meas['MAPE (%)']:.2f}%). "
            f"It differs from the computational value by {m_comp['MAPE (%)']:.2f}% "
            f"(computational alone differs from measured by {base['MAPE (%)']:.2f}%).")
    else:
        st.error(
            f"❌ The ANN is not yet close enough to the measured target "
            f"(need R² ≥ {res['target_r2']} and MAPE ≤ {res['target_mape']}%). "
            "Try other neurons/layers, a different solver, or more data.")

    st.subheader("Comparison table (test set)")
    rows = {"ANN vs Measured (target)": m_meas,
            "ANN vs Computational": m_comp,
            "Baseline: Computational vs Measured": base}
    st.dataframe(pd.DataFrame(rows).T.style.format("{:.4f}"), width="stretch")
    st.caption("The baseline row is the gap between the calculation and the measurement "
               "before any ANN is used. The ANN cannot be exactly equal to both when "
               "this gap is large.")

    m_tr = compute_metrics(res["y_tr"], res["p_tr"])
    m_te = compute_metrics(res["y_te"], res["p_te"])
    if m_tr["R²"] - m_te["R²"] > 0.05:
        st.warning("The gap between training and test R² is large – possible overfitting. "
                   "Try a smaller network or a higher alpha.")
    if res["n_iter"] >= res["max_iter"]:
        st.warning("Training reached the maximum number of epochs without converging. "
                   "Increase 'Maximum epochs' or the learning rate.")
    st.pyplot(plot_results(res))


def _guess(cols, keys, default):
    for i, c in enumerate(cols):
        if any(k in c.lower() for k in keys):
            return i
    return default


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("☀️ PV Power Output Prediction using ANN")
st.caption("Measured power is the target. Computational power is calculated from the "
           "inputs. The ANN is judged against both.")

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("1. Data")
    source = st.radio("Data source", ["Simulated data (demo)", "Upload CSV / Excel"])
    if source == "Simulated data (demo)":
        n_samples = st.slider("Number of samples", 200, 5000, 1500, 100)
        seed = st.number_input("Random seed (data)", 0, 9999, 42)

    st.subheader("2. Computational model")
    with st.expander("PV module & SEDA factors", expanded=True):
        p_stc = st.number_input("P_max STC (W)", 100.0, 1000.0, 605.0, 5.0)
        gamma = st.number_input("γ_Pmax (%/°C)", -1.0, 0.0, -0.29, 0.01)
        dirt = st.slider("Dirt / soiling (%)", 0.0, 20.0, 2.0, 0.5)
        shading = st.slider("Shading (%)", 0.0, 100.0, 0.0, 1.0)
        f_lid = st.number_input("f_LID (light-induced degradation)", 0.5, 1.0, 0.975, 0.001)
        f_age = st.number_input("f_age (aging)", 0.5, 1.0, 1.0, 0.001)
        f_mm = st.number_input("f_mm (mismatch)", 0.5, 1.0, 1.0, 0.001)
        bifacial = st.number_input("Bifacial gain (%)", 0.0, 30.0, 0.0, 1.0)
    cfg = dict(p_stc=p_stc, gamma=gamma, dirt=dirt, shading=shading,
               f_lid=f_lid, f_age=f_age, f_mm=f_mm, bifacial=bifacial)

    st.subheader("3. ANN architecture")
    n_layers = st.slider("Number of hidden layers", 1, 5, 2)
    hidden_layers = tuple(
        st.slider(f"Neurons - Hidden layer {i + 1}", 2, 256, max(8, 64 // (2 ** i)), 2)
        for i in range(n_layers))
    activation = st.selectbox("Activation function", ["relu", "tanh", "logistic"])
    solver = st.selectbox("Solver / optimizer", ["adam", "lbfgs", "sgd"])
    learning_rate = st.select_slider(
        "Learning rate", [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], value=0.001)
    max_iter = st.slider("Maximum epochs", 50, 3000, 500, 50)
    alpha = st.select_slider("L2 regularization (alpha)",
                             [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], value=1e-4)
    early_stop = st.checkbox("Early stopping", value=True)

    st.subheader("4. Training & targets")
    test_size = st.slider("Test data ratio", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random seed (model)", 0, 9999, 42)
    target_r2 = st.slider("Near target: minimum R² (vs measured)", 0.80, 0.999, 0.95, 0.001)
    target_mape = st.slider("Near target: maximum MAPE % (vs measured)", 1.0, 20.0, 5.0, 0.5)
    target_mape_comp = st.slider("Near computational: maximum MAPE %", 1.0, 30.0, 10.0, 0.5)
    with st.expander("Advanced"):
        pull = st.slider("Pull ANN toward computational (%)", 0, 50, 0, 5)
        st.caption("0 = train on the measured target only. Higher values blend some "
                   "computational power into the training target.")

# ----------------------------------------------------------------------------
# Load data and choose columns
# ----------------------------------------------------------------------------
if source == "Simulated data (demo)":
    df = generate_demo_data(n_samples, p_stc, gamma, int(seed))
    num_cols = list(df.columns)
    st.info("You are using simulated data for demo purposes. "
            "Upload a CSV / Excel file of real field data for your project results.")
else:
    uploaded = st.file_uploader("Upload CSV / Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.warning("Please upload a file or switch to simulated data in the sidebar.")
        st.stop()
    raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    num_cols = raw.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 3:
        st.error("The file needs at least 3 numeric columns "
                 "(measured power, irradiance and PV temperature).")
        st.stop()
    df = raw[num_cols].dropna().reset_index(drop=True)

if len(df) < 10:
    st.error(f"Not enough valid records — only {len(df)}. Need at least 10.")
    st.stop()

c1, c2, c3 = st.columns(3)
meas_col = c1.selectbox("Measured power column (target)", num_cols,
                        index=_guess(num_cols, ["measured", "p_meas", "power", "p_out"], len(num_cols) - 1))
irr_col = c2.selectbox("Irradiance column", num_cols,
                       index=_guess(num_cols, ["irrad", "g_"], 0))
pvt_col = c3.selectbox("PV temperature column", num_cols,
                       index=_guess(num_cols, ["pv temp", "pv_temp", "module", "cell"], min(1, len(num_cols) - 1)))
if len({meas_col, irr_col, pvt_col}) < 3:
    st.error("Measured power, irradiance and PV temperature must be three different columns.")
    st.stop()

options = [c for c in num_cols if c != meas_col]
input_cols = st.multiselect("ANN input columns", options, default=options)
use_comp = st.checkbox("Also use computational power as an ANN input (recommended)", value=True)
if not input_cols and not use_comp:
    st.warning("Select at least one ANN input.")
    st.stop()

meas_all = df[meas_col].to_numpy(float)
comp_all = computational_power(df[irr_col], df[pvt_col], cfg)
feat_names = [*input_cols, *([COMP_NAME] if use_comp else [])]
need = list(dict.fromkeys([irr_col, pvt_col, *input_cols]))

st.markdown(f"**Target:** measured power (`{meas_col}`)  \n"
            f"**Computational power:** calculated from `{irr_col}` and `{pvt_col}`  \n"
            f"**ANN inputs:** {', '.join(feat_names)}")
if len(df) < 100:
    st.info("Small dataset: the 'lbfgs' solver with 1 hidden layer of 8–16 neurons "
            "usually works best. Large networks overfit easily on few records.")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_data, tab_train, tab_predict, tab_export = st.tabs(
    ["📊 Data", "🧠 Training & Results", "🔮 Prediction", "💾 Export"])

with tab_data:
    st.subheader("Data preview")
    st.dataframe(df.assign(**{COMP_NAME: comp_all}).head(100), width="stretch")
    base_all = compute_metrics(meas_all, comp_all)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Number of samples", f"{len(df):,}")
    d2.metric("Mean measured", f"{meas_all.mean():.1f} W")
    d3.metric("Mean computational", f"{comp_all.mean():.1f} W")
    d4.metric("Computational vs measured (MAPE)", f"{base_all['MAPE (%)']:.2f} %")
    st.subheader("Summary statistics")
    tmp = df[[*dict.fromkeys([*input_cols, irr_col, pvt_col]), meas_col]].copy()
    tmp[COMP_NAME] = comp_all
    st.dataframe(tmp.describe().T, width="stretch")
    st.subheader("Correlation with measured power")
    st.bar_chart(tmp.corr()[meas_col].drop(meas_col))

with tab_train:
    st.subheader("Network architecture")
    st.pyplot(draw_architecture(len(feat_names), hidden_layers))
    n_params = sum(a * b + b for a, b in zip((len(feat_names), *hidden_layers), (*hidden_layers, 1)))
    st.caption(f"Structure: {len(feat_names)} → " + " → ".join(map(str, hidden_layers))
               + f" → 1  |  Total parameters: {n_params:,}")

    if st.button("🚀 Train ANN model", type="primary"):
        lam = pull / 100
        X_df = df[input_cols].copy()
        if use_comp:
            X_df[COMP_NAME] = comp_all
        X = X_df.to_numpy(float)
        y = (1 - lam) * meas_all + lam * comp_all
        tr, te = train_test_split(np.arange(len(df)), test_size=test_size,
                                  random_state=int(random_state))
        params = dict(activation=activation, solver=solver, alpha=alpha,
                      max_iter=max_iter, random_state=int(random_state))
        if solver != "lbfgs":
            params.update(learning_rate_init=learning_rate,
                          early_stopping=bool(early_stop and len(tr) >= 50))
        with st.spinner("Training model..."):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model, sx, sy = train_ann(X[tr], y[tr], hidden_layers, params)

        st.session_state["result"] = {
            "model": model, "sx": sx, "sy": sy, "cfg": cfg,
            "input_cols": list(input_cols), "feat_names": feat_names, "use_comp": use_comp,
            "irr_col": irr_col, "pvt_col": pvt_col, "meas_col": meas_col, "need": need,
            "pull": pull, "p_stc": p_stc,
            "y_tr": y[tr], "p_tr": predict_ann(model, sx, sy, X[tr]),
            "y_te": y[te], "p_te": predict_ann(model, sx, sy, X[te]),
            "meas_te": meas_all[te], "comp_te": comp_all[te], "X_te": X[te],
            "target_r2": target_r2, "target_mape": target_mape,
            "target_mape_comp": target_mape_comp,
            "n_iter": int(model.n_iter_), "max_iter": int(max_iter),
            "loss_curve": list(getattr(model, "loss_curve_", [])),
            "stats": {c: (float(df[c].min()), float(df[c].max()), float(df[c].median()))
                      for c in need},
        }

    res = st.session_state.get("result")
    if res is None:
        st.info("Press the button above to train the model.")
    else:
        if res["input_cols"] != list(input_cols) or res["use_comp"] != use_comp:
            st.warning("These results are from earlier input settings. "
                       "Press **Train ANN model** to update.")
        show_results(res)

with tab_predict:
    res = st.session_state.get("result")
    if res is None:
        st.info("Train a model first in the 'Training & Results' tab.")
    else:
        st.subheader("Power output prediction (manual input)")
        cols = st.columns(3)
        vals = {}
        for i, name in enumerate(res["need"]):
            lo, hi, med = res["stats"][name]
            span = max(hi - lo, 1.0)
            vals[name] = cols[i % 3].number_input(
                name, value=med, min_value=lo - span, max_value=hi + span, key=f"in_{name}")

        if st.button("Predict power output"):
            p_c = float(computational_power(vals[res["irr_col"]], vals[res["pvt_col"]], res["cfg"]))
            row = [vals[c] for c in res["input_cols"]] + ([p_c] if res["use_comp"] else [])
            p = max(float(predict_ann(res["model"], res["sx"], res["sy"], np.array([row]))[0]), 0.0)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("ANN prediction", f"{p:.1f} W")
            r2.metric("Computational", f"{p_c:.1f} W")
            r3.metric("ANN vs computational", f"{(p - p_c) / p_c * 100:+.2f} %" if p_c else "—")
            r4.metric("Percentage of Pmax", f"{p / res['p_stc'] * 100:.1f} %")

        st.divider()
        st.subheader("Batch prediction (CSV / Excel)")
        st.caption("Needs columns: " + ", ".join(res["need"]) +
                   f".  Optional: {res['meas_col']} (to compare with the measured value).")
        batch = st.file_uploader("File with the same input columns", type=["csv", "xlsx"], key="batch")
        if batch is not None:
            bdf = pd.read_csv(batch) if batch.name.lower().endswith(".csv") else pd.read_excel(batch)
            missing = [c for c in res["need"] if c not in bdf.columns]
            if missing:
                st.error(f"Columns missing from file: {missing}")
            else:
                bdf = bdf.dropna(subset=res["need"]).reset_index(drop=True)
                pc = computational_power(bdf[res["irr_col"]], bdf[res["pvt_col"]], res["cfg"])
                Xb = bdf[res["input_cols"]].copy()
                if res["use_comp"]:
                    Xb[COMP_NAME] = pc
                bdf[COMP_NAME] = pc
                bdf["P_ANN (W)"] = np.clip(
                    predict_ann(res["model"], res["sx"], res["sy"], Xb.to_numpy(float)), 0, None)
                if res["meas_col"] in bdf.columns:
                    m = pd.to_numeric(bdf[res["meas_col"]], errors="coerce")
                    bdf["Error ANN vs measured (%)"] = ((bdf["P_ANN (W)"] - m)
                                                        / m.where(m > 10) * 100).round(2)
                st.dataframe(bdf.head(100), width="stretch")
                st.download_button("⬇️ Download predictions (CSV)",
                                   bdf.to_csv(index=False).encode(),
                                   "batch_predictions.csv", "text/csv")

with tab_export:
    res = st.session_state.get("result")
    if res is None:
        st.info("Train a model first to export.")
    else:
        out = pd.DataFrame(res["X_te"], columns=res["feat_names"])
        out["P_measured_target (W)"] = res["meas_te"]
        out["P_computational (W)"] = res["comp_te"]
        out["P_ANN (W)"] = res["p_te"]
        out["Error ANN − Measured (W)"] = out["P_ANN (W)"] - out["P_measured_target (W)"]
        out["Error ANN − Computational (W)"] = out["P_ANN (W)"] - out["P_computational (W)"]
        st.download_button("⬇️ Test results (CSV)", out.to_csv(index=False).encode(),
                           "test_results.csv", "text/csv")
        buf = io.BytesIO()
        joblib.dump({k: res[k] for k in ("model", "sx", "sy", "cfg", "input_cols", "feat_names",
                                         "use_comp", "irr_col", "pvt_col", "meas_col")}, buf)
        st.download_button("⬇️ Model ANN (.joblib)", buf.getvalue(),
                           "ann_solar_model.joblib", "application/octet-stream")
