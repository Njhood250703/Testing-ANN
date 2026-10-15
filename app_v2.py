"""
Solar PV ANN (v2) - Measured vs Computational (SEDA) power output
=================================================================
Standalone test harness for the ANN page of app.py.
It follows app.py: same field-data columns, same SEDA equations, same theme.

Run:  streamlit run app_v2.py
Needs: streamlit, pandas, numpy, scikit-learn, plotly, openpyxl, joblib

File layout
  [A] STANDALONE ONLY    page config, CSS, SEDA copies, demo data, data loading
  [B] PASTE INTO app.py  everything from "ANN CORE" down to render_ann_page()

To merge into app.py later (app.py is NOT changed by this file):
  1. Add to the imports of app.py:  import warnings, joblib
     from sklearn.neural_network import MLPRegressor
     from sklearn.preprocessing import StandardScaler
     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
     from sklearn.model_selection import train_test_split
  2. Paste section [B] above the page chain (after pct_diff).
  3. Replace the body of  `elif page == "🤖 ANN Prediction":`  with
         render_ann_page(df_meas)
     calc_factors / calc_outputs / noct_pv_temp / malaysia_pv_temp / pct_diff
     already exist in app.py, so section [A] copies are not needed there.
"""

import io
import warnings
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════════
#  [A] STANDALONE ONLY  (remove when merging into app.py)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Solar PV ANN (v2)", page_icon="☀️", layout="wide")

# Same theme as app.py (trimmed to what this page uses)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&display=swap');
:root { --gold:#FFB800; --orange:#FF6B00; --cyan:#00D4FF; --muted:#8892A4; --txt:#E8EAF0; }
html, body, [class*="css"] { font-family:'Exo 2', sans-serif; }
.stApp { background:linear-gradient(135deg,#0A0E1A 0%,#0D1525 60%,#0A1628 100%); color:var(--txt); }
.solar-header { background:linear-gradient(90deg,#0A0E1A,#1a2744,#0A0E1A); border-bottom:2px solid var(--gold);
    padding:1.4rem 2rem; margin:-1rem -1rem 1.5rem -1rem; text-align:center; }
.solar-header h1 { font-family:'Orbitron',monospace; font-size:2rem; font-weight:900; color:var(--gold);
    letter-spacing:4px; text-shadow:0 0 30px rgba(255,184,0,.5); margin:0; }
.solar-header p { color:var(--muted); font-size:.9rem; letter-spacing:2px; margin:.3rem 0 0; }
.sec { font-family:'Orbitron',monospace; font-size:1rem; color:var(--gold); letter-spacing:2px;
    border-left:3px solid var(--gold); padding-left:.8rem; margin:1.2rem 0 .8rem; }
.res-box { border-radius:10px; padding:1rem 1.2rem; margin:.4rem 0; }
.res-calculated { background:rgba(255,184,0,.08); border:1px solid rgba(255,184,0,.35); }
.res-title { font-family:'Orbitron',monospace; font-size:.78rem; letter-spacing:1px; margin-bottom:.4rem; }
.res-val { font-size:1.5rem; font-weight:700; font-family:'Orbitron',monospace; }
.res-formula { font-size:.75rem; color:var(--muted); margin-top:.3rem; font-style:italic; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0A0E1A,#0D1525); border-right:1px solid rgba(255,184,0,.2); }
.stButton>button { background:linear-gradient(135deg,var(--orange),var(--gold)); color:#0A0E1A;
    font-family:'Orbitron',monospace; font-weight:700; letter-spacing:1px; border:none; border-radius:8px; }
.stTabs [data-baseweb="tab"] { font-family:'Orbitron',monospace; font-size:.7rem; letter-spacing:1px; }
</style>
""", unsafe_allow_html=True)


# ── SEDA engine: verbatim copies from app.py (Ch 5.4–5.14) ─────────────────────
def calc_factors(irradiance, t_mod, t_stc,
                 gamma_pmax, beta_voc, alpha_isc,
                 dirt_pct, shading_pct, f_lid, f_age, f_mm):
    f_g       = irradiance / 1000                                    # Eq 5.9
    f_degrad  = f_lid * f_age                                        # Eq 5.8
    f_temp_p  = 1 + (gamma_pmax / 100) * (t_mod - t_stc)            # Eq 5.10
    f_temp_i  = 1 + (alpha_isc  / 100) * (t_mod - t_stc)            # Eq 5.12
    f_temp_v  = 1 + (beta_voc   / 100) * (t_mod - t_stc)            # Eq 5.14
    f_clean   = (100 - dirt_pct)    / 100
    f_unshade = (100 - shading_pct) / 100
    return dict(f_g=f_g, f_degrad=f_degrad, f_temp_p=f_temp_p,
                f_temp_i=f_temp_i, f_temp_v=f_temp_v,
                f_clean=f_clean, f_unshade=f_unshade, f_mm=f_mm,
                f_lid=f_lid, f_age=f_age)


def calc_outputs(p_stc, voc_stc, vmp_stc, isc_stc, imp_stc, factors,
                 bifacial_gain=0, n_s=1, n_p=1):
    F = factors
    p_max = (p_stc * (1 + bifacial_gain/100)
             * F["f_mm"] * F["f_degrad"] * F["f_temp_p"]
             * F["f_g"] * F["f_clean"] * F["f_unshade"]
             * n_s * n_p)                                            # Eq 5.7
    isc   = isc_stc * F["f_temp_i"] * F["f_g"] * F["f_clean"] * F["f_unshade"] * n_p  # Eq 5.11
    imp   = imp_stc * F["f_temp_i"] * F["f_g"] * F["f_clean"] * F["f_unshade"] * n_p
    voc   = voc_stc * F["f_temp_v"] * n_s                           # Eq 5.13
    vmp   = vmp_stc * F["f_temp_v"] * n_s
    return dict(p_max=p_max, isc=isc, imp=imp, voc=voc, vmp=vmp)


def malaysia_pv_temp(irradiance, t_amb):
    """Eq 5.6 — Malaysian climate empirical model (Zainuddin, H., 2014)."""
    return -15.76 + 0.02 * irradiance + 1.64 * t_amb


def noct_pv_temp(irradiance, t_amb, noct):
    """Eq 5.4 — NOCT temperature model."""
    return t_amb + irradiance * (noct - 20) / 800


def pct_diff(calc, meas):
    """Error = (P_meas - P_calc) / P_calc × 100%"""
    if calc and calc != 0:
        return (meas - calc) / abs(calc) * 100
    return None


# ── Demo data in the same schema as app.py's `measured_data` table ─────────────
@st.cache_data
def generate_demo_data(n_days, seed):
    """Simulated field data (testing only). Measured power = a 'true' SEDA state
    (dirt, shading, mismatch the user does not know) + low-light loss + noise."""
    rng = np.random.default_rng(seed)
    hours = np.arange(7, 19)
    day = np.repeat(np.arange(n_days), len(hours))
    hr = np.tile(hours, n_days)
    n = len(hr)

    ts = (pd.Timestamp(date.today()) - pd.Timedelta(days=n_days)
          + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hr, unit="h"))
    sun = np.clip(np.sin(np.pi * (hr - 6.5) / 12), 0, None)
    cloud = rng.uniform(0.4, 1.0, n_days)[day]

    irr = np.clip(1000 * sun * cloud * rng.uniform(0.85, 1.05, n) + rng.normal(0, 12, n), 0, None)
    t_amb = 26 + 6 * sun + rng.normal(0, 0.8, n)
    t_pv = t_amb + irr / 800 * 25 + rng.normal(0, 1.0, n)

    F = calc_factors(irr, t_pv, 25.0, -0.37, -0.28, 0.048, 4.0, 1.0, 0.975, 1.0, 0.98)
    p_true = calc_outputs(315.0, 40.5, 33.2, 10.05, 9.49, F)["p_max"]
    p_meas = np.clip(p_true * (1 - 0.05 * np.exp(-irr / 150)) + rng.normal(0, 2, n), 0, None)

    return pd.DataFrame({
        "recorded_at": ts, "site_name": "Demo Site",
        "irradiance": irr, "pv_temp": t_pv, "ambient_temp": t_amb,
        "power_measured": p_meas,
    })


def load_standalone_data():
    """Stands in for app.py's load_measured(). Returns a df_meas-like DataFrame."""
    with st.sidebar:
        st.markdown('<p class="sec">🧪 TEST DATA</p>', unsafe_allow_html=True)
        src = st.radio("Data source", ["Simulated (demo)", "Upload CSV / Excel"], key="sa_src")
        if src.startswith("Simulated"):
            days = st.slider("Days of data", 10, 120, 30, 5, key="sa_days")
            seed = st.number_input("Random seed (data)", 0, 9999, 42, key="sa_seed")
            df = generate_demo_data(int(days), int(seed))
            st.download_button("⬇ Demo data as CSV (app.py import format)",
                               df.to_csv(index=False).encode(), "demo_field_data.csv", "text/csv")
            st.caption("Simulated data for testing the ANN only. "
                       "Upload real field data for your results.")
            return df
        up = st.file_uploader("Field data (columns as in app.py import template)",
                              type=["csv", "xlsx", "xls"], key="sa_up")
    if up is None:
        st.warning("Upload a CSV / Excel file or switch to simulated data in the sidebar.")
        st.stop()
    try:
        df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  [B] PASTE INTO app.py  —  ANN CORE
# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit >= 1.50 uses width="stretch"; older versions use use_container_width
FW = ({"width": "stretch"} if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50)
      else {"use_container_width": True})

REQUIRED = ["irradiance", "pv_temp", "ambient_temp", "power_measured"]   # as in measured_data
ENV_LABELS = {"irradiance": "Solar Irradiance (W/m²)",
              "pv_temp": "PV Temperature (°C)",
              "ambient_temp": "Ambient Temperature (°C)"}
COMP = "p_computational"
LABEL = {**ENV_LABELS, COMP: "SEDA computational power (W)"}

MODE_A = "A. Correction — inputs + SEDA power → measured"
MODE_B = "B. Blended — inputs → w × measured + (1 − w) × SEDA"
MODE_C = "C. Emulator — inputs → SEDA power"

TEMP_MEASURED = "Measured PV temperature (column)"
TEMP_NOCT = "NOCT Model (Eq 5.4)"
TEMP_MY = "Malaysian Climate Model (Eq 5.6)"


def seda_power(env, cfg):
    """SEDA computational P_max (W) for every row of `env`, using app.py's engine."""
    g = env["irradiance"].to_numpy(float)
    ta = env["ambient_temp"].to_numpy(float)
    if cfg["temp_mode"] == TEMP_NOCT:
        tm = noct_pv_temp(g, ta, cfg["noct"])
    elif cfg["temp_mode"] == TEMP_MY:
        tm = malaysia_pv_temp(g, ta)
    else:
        tm = env["pv_temp"].to_numpy(float)
    F = calc_factors(g, tm, cfg["t_stc"], cfg["gamma"], cfg["beta_v"], cfg["alpha_i"],
                     cfg["dirt"], cfg["shading"], cfg["f_lid"], cfg["f_age"], cfg["f_mm"])
    # Only P_max is needed here, so the voltage/current STC arguments are placeholders
    out = calc_outputs(cfg["p_stc"], 1.0, 1.0, 1.0, 1.0, F,
                       cfg["bifacial"], cfg["n_s"], cfg["n_p"])
    return np.clip(np.asarray(out["p_max"], float), 0, None)


def build_X(env, comp, env_feats, mode):
    X = env[env_feats].copy()
    if mode == MODE_A:
        X[COMP] = comp
    return X


def train_ann(X_train, y_train, hidden_layers, params):
    sx = StandardScaler().fit(X_train)
    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    model = MLPRegressor(hidden_layer_sizes=hidden_layers, **params)
    model.fit(sx.transform(X_train), sy.transform(y_train.reshape(-1, 1)).ravel())
    return model, sx, sy


def predict_ann(model, sx, sy, X):
    pred = model.predict(sx.transform(X))
    return sy.inverse_transform(pred.reshape(-1, 1)).ravel()


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true > 10  # avoid division by near-zero values
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.any() else np.nan)
    return {"R²": r2_score(y_true, y_pred),
            "RMSE (W)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE (W)": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": mape}


def _style(fig, title="", h=350, xt=None, yt=None):
    """Same plot styling as app.py."""
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,21,37,0.8)",
                      font=dict(color="#8892A4", family="Exo 2"),
                      legend=dict(bgcolor="rgba(0,0,0,.5)"),
                      title=title, title_font=dict(color="#FFB800"),
                      xaxis_title=xt, yaxis_title=yt)
    fig.update_xaxes(gridcolor="rgba(255,184,0,.08)")
    fig.update_yaxes(gridcolor="rgba(255,184,0,.08)")
    return fig


def architecture_fig(n_in, hidden, max_nodes=8):
    sizes = [n_in, *hidden, 1]
    labels = ["Input"] + [f"Hidden {i + 1}" for i in range(len(hidden))] + ["Output"]
    colors = ["#00D4FF"] + ["#FFB800"] * len(hidden) + ["#FF6B00"]
    pos = []
    for x, s in enumerate(sizes):
        k = min(s, max_nodes)
        pos.append([(x, y) for y in np.linspace((k - 1) / 2, -(k - 1) / 2, k)])
    ex, ey = [], []
    for a, b in zip(pos[:-1], pos[1:]):
        for x1, y1 in a:
            for x2, y2 in b:
                ex += [x1, x2, None]
                ey += [y1, y2, None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                               line=dict(color="rgba(136,146,164,.25)", width=1)))
    for x, (p, c) in enumerate(zip(pos, colors)):
        fig.add_trace(go.Scatter(x=[q[0] for q in p], y=[q[1] for q in p], mode="markers",
                                 hoverinfo="skip", marker=dict(size=14, color=c)))
        extra = " ⋮" if sizes[x] > max_nodes else ""
        fig.add_annotation(x=x, y=max(q[1] for q in p) + 0.9, showarrow=False,
                           text=f"{labels[x]}<br>({sizes[x]}){extra}",
                           font=dict(color="#E8EAF0", size=11))
    _style(fig, h=300)
    fig.update_layout(showlegend=False, margin=dict(t=30, b=10))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _seda_settings():
    """Computational-model settings (same inputs and defaults as 🧮 SEDA Computation)."""
    with st.expander("🧮 Computational model (SEDA) — use the same datasheet values as 🧮 SEDA Computation"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_stc = st.number_input("P_max STC (W)", 100.0, 1000.0, 315.0, 0.1, key="ann2_pstc")
            noct = st.number_input("NOCT (°C)", 20.0, 60.0, 47.0, 0.1, key="ann2_noct")
            t_stc = st.number_input("T_STC (°C)", 0.0, 50.0, 25.0, 0.1, key="ann2_tstc")
            temp_mode = st.selectbox("PV temperature used in computation",
                                     [TEMP_MEASURED, TEMP_NOCT, TEMP_MY], key="ann2_tmode")
        with c2:
            gamma = st.number_input("γ_Pmax (%/°C)", -1.0, 0.0, -0.37, 0.001, key="ann2_gamma")
            beta_v = st.number_input("β_Voc (%/°C)", -1.0, 0.0, -0.28, 0.001, key="ann2_beta")
            alpha_i = st.number_input("α_Isc (%/°C)", 0.0, 1.0, 0.048, 0.001, key="ann2_alphai")
            f_lid = st.number_input("f_LID", 0.5, 1.0, 0.975, 0.001, key="ann2_flid")
        with c3:
            f_age = st.number_input("f_age", 0.5, 1.0, 1.0, 0.001, key="ann2_fage")
            f_mm = st.number_input("f_mm — Mismatch", 0.5, 1.0, 1.0, 0.001, key="ann2_fmm")
            dirt = st.slider("Dirt / Soiling (%)", 0.0, 20.0, 2.0, 0.5, key="ann2_dirt")
            shading = st.slider("Shading (%)", 0.0, 100.0, 0.0, 1.0, key="ann2_shade")
        d1, d2, d3, d4 = st.columns(4)
        bifacial = d1.number_input("Bifacial Gain (%)", 0.0, 30.0, 0.0, 1.0, key="ann2_bif")
        n_s = d2.number_input("N_s — modules in series", 1, 100, 1, 1, key="ann2_ns")
        n_p = d3.number_input("N_p — strings in parallel", 1, 100, 1, 1, key="ann2_np")
        min_irr = d4.number_input("Ignore records below (W/m²)", 0.0, 500.0, 0.0, 10.0, key="ann2_minirr")
    return dict(p_stc=p_stc, noct=noct, t_stc=t_stc, temp_mode=temp_mode, gamma=gamma,
                beta_v=beta_v, alpha_i=alpha_i, f_lid=f_lid, f_age=f_age, f_mm=f_mm,
                dirt=dirt, shading=shading, bifacial=bifacial, n_s=int(n_s), n_p=int(n_p),
                min_irr=min_irr)


def _summary(res):
    """Headline accuracy, target check and warnings for the trained model."""
    m = res["m_ref"]
    st.markdown(f"**Accuracy of the ANN against {res['ref_name']} (test set)**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R²", f"{m['R²']:.4f}")
    k2.metric("RMSE", f"{m['RMSE (W)']:.2f} W")
    k3.metric("MAE", f"{m['MAE (W)']:.2f} W")
    k4.metric("MAPE", f"{m['MAPE (%)']:.2f} %")
    if m["R²"] >= res["target_r2"] and m["MAPE (%)"] <= res["target_mape"]:
        st.success(f"✅ Targets achieved (R² ≥ {res['target_r2']}, MAPE ≤ {res['target_mape']}%).")
    else:
        st.error(f"❌ Targets not achieved (R² ≥ {res['target_r2']}, MAPE ≤ {res['target_mape']}%). "
                 "Try other neurons/layers, a different solver, or more data.")
    if res["m_tr"]["R²"] - res["m_te"]["R²"] > 0.05:
        st.warning("The gap between training and test R² is large – possible overfitting. "
                   "Try a smaller network or a higher alpha.")
    if res["n_iter"] >= res["max_iter"]:
        st.warning("Training reached the maximum number of epochs without converging. "
                   "Increase 'Maximum epochs' or the learning rate.")


def render_ann_page(df_meas):
    """ANN page. `df_meas` has the columns of app.py's measured_data table."""
    st.markdown('<p class="sec">🤖 ANN PREDICTION MODEL — MEASURED vs COMPUTATIONAL (SEDA)</p>',
                unsafe_allow_html=True)
    st.caption("Train an ANN with field-measured and SEDA-computed power output, "
               "then compare the ANN against both.")

    if df_meas is None or df_meas.empty:
        st.warning("No field measurement data found. Please add data in 🔬 Field Measurement first.")
        return
    missing = [c for c in REQUIRED if c not in df_meas.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    cfg = _seda_settings()

    data = df_meas.copy()
    for c in REQUIRED:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=REQUIRED)
    data = data[data["irradiance"] >= cfg["min_irr"]]
    if "recorded_at" in data.columns:
        data["recorded_at"] = pd.to_datetime(data["recorded_at"], errors="coerce")
        data = data.sort_values("recorded_at")
    data = data.reset_index(drop=True)
    if len(data) < 10:
        st.error(f"Not enough valid records — only {len(data)}. Need at least 10.")
        return
    data[COMP] = seda_power(data, cfg)
    p_rated = cfg["p_stc"] * cfg["n_s"] * cfg["n_p"]

    base = compute_metrics(data["power_measured"], data[COMP])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Field records", f"{len(data):,}")
    c2.metric("Mean measured", f"{data['power_measured'].mean():.1f} W")
    c3.metric("Mean SEDA computational", f"{data[COMP].mean():.1f} W")
    c4.metric("SEDA vs measured (MAPE)", f"{base['MAPE (%)']:.2f} %")
    if len(data) < 100:
        st.info("Small dataset: the 'lbfgs' solver with 1 hidden layer of 8–16 neurons "
                "usually works best. Large networks overfit easily on few records.")

    tab_train, tab_predict, tab_results = st.tabs(
        ["⚙️ Train Model", "🔮 Predict", "📊 Results & Metrics"])

    # ── TAB 1: TRAIN ──────────────────────────────────────────────────────────
    with tab_train:
        st.markdown("**1️⃣ What should the ANN learn?**")
        mode = st.radio("Training mode", [MODE_A, MODE_B, MODE_C], key="ann2_mode")
        w_meas = 0.5
        if mode == MODE_B:
            w_meas = st.slider("Weight of measured (w)", 0.0, 1.0, 0.5, 0.05, key="ann2_w")

        st.markdown("**2️⃣ Data configuration**")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Input features (X)**")
            use = {k: st.checkbox(lbl, value=True, key=f"ann2_use_{k}")
                   for k, lbl in ENV_LABELS.items()}
            if mode == MODE_A:
                st.caption("➕ SEDA computational power is added automatically as an input.")
        with d2:
            st.markdown("**Output target (Y)**")
            target_label = {MODE_A: "Measured power (W)",
                            MODE_B: f"{w_meas:.2f} × Measured + {1 - w_meas:.2f} × SEDA (W)",
                            MODE_C: "SEDA computational power (W)"}[mode]
            st.markdown(f"🎯 {target_label}")
            test_size = st.slider("Test set size (%)", 10, 40, 20, 5, key="ann2_test")
            random_state = st.number_input("Random seed (model)", 0, 9999, 42, key="ann2_seed")
        env_feats = [k for k, v in use.items() if v]
        n_in = len(env_feats) + (1 if mode == MODE_A else 0)

        st.markdown("**3️⃣ ANN architecture**")
        n_layers = st.slider("Hidden layers", 1, 5, 1 if len(data) < 100 else 2, key="ann2_nl")
        lc = st.columns(n_layers)
        hidden = tuple(int(lc[i].number_input(f"Layer {i + 1} neurons", 2, 256,
                                              max(8, (16 if len(data) < 100 else 64) // (2 ** i)),
                                              2, key=f"ann2_n{i}"))
                       for i in range(n_layers))
        a1, a2, a3 = st.columns(3)
        activation = a1.selectbox("Activation", ["relu", "tanh", "logistic"], key="ann2_act")
        solver = a2.selectbox("Solver / optimizer",
                              ["lbfgs", "adam", "sgd"] if len(data) < 100 else ["adam", "lbfgs", "sgd"],
                              key="ann2_solver")
        lr = a3.select_slider("Learning rate", [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                              value=0.001, key="ann2_lr")
        if n_in:
            st.plotly_chart(architecture_fig(n_in, hidden), key="ann2_arch", **FW)
            n_params = sum(a * b + b for a, b in zip((n_in, *hidden), (*hidden, 1)))
            st.caption(f"Structure: {n_in} → " + " → ".join(map(str, hidden))
                       + f" → 1  |  Total parameters: {n_params:,}")

        st.markdown("**4️⃣ Training parameters & targets**")
        p1, p2, p3, p4 = st.columns(4)
        max_iter = p1.number_input("Maximum epochs", 50, 3000, 500, 50, key="ann2_iter")
        alpha = p2.select_slider("L2 regularization (alpha)", [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                                 value=1e-4, key="ann2_alpha")
        target_r2 = p3.slider("Minimum R² target", 0.80, 0.999, 0.95, 0.001, key="ann2_tr2")
        target_mape = p4.slider("Maximum MAPE target (%)", 1.0, 20.0, 5.0, 0.5, key="ann2_tmape")
        early_stop = st.checkbox("Early stopping (adam / sgd, needs ≥ 50 training records)",
                                 value=True, key="ann2_es")

        if st.button("🚀 TRAIN ANN MODEL", key="ann2_train", **FW):
            if not env_feats:
                st.error("Select at least one input feature.")
            else:
                meas = data["power_measured"].to_numpy(float)
                comp = data[COMP].to_numpy(float)
                X_df = build_X(data, comp, env_feats, mode)
                feats = list(X_df.columns)
                X = X_df.to_numpy(float)
                y = {MODE_A: meas, MODE_B: w_meas * meas + (1 - w_meas) * comp, MODE_C: comp}[mode]
                tr, te = train_test_split(np.arange(len(data)), test_size=test_size / 100,
                                          random_state=int(random_state))
                params = dict(activation=activation, solver=solver, alpha=alpha,
                              max_iter=int(max_iter), random_state=int(random_state))
                if solver != "lbfgs":
                    params.update(learning_rate_init=lr,
                                  early_stopping=bool(early_stop and len(tr) >= 50))
                with st.spinner("Training in progress... please wait."):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model, sx, sy = train_ann(X[tr], y[tr], hidden, params)

                p_tr = predict_ann(model, sx, sy, X[tr])
                p_te = predict_ann(model, sx, sy, X[te])
                ref_te, ref_name = ((comp[te], "SEDA computational") if mode == MODE_C
                                    else (meas[te], "measured"))
                st.session_state["ann2"] = dict(
                    model=model, sx=sx, sy=sy, features=feats, env_feats=env_feats,
                    mode=mode, w=w_meas, target_label=target_label, cfg=cfg, p_rated=p_rated,
                    y_tr=y[tr], p_tr=p_tr, y_te=y[te], p_te=p_te,
                    meas_te=meas[te], comp_te=comp[te], X_te=X[te],
                    m_tr=compute_metrics(y[tr], p_tr), m_te=compute_metrics(y[te], p_te),
                    m_ref=compute_metrics(ref_te, p_te), ref_name=ref_name,
                    target_r2=target_r2, target_mape=target_mape,
                    n_train=len(tr), n_test=len(te), hidden=hidden,
                    n_iter=int(model.n_iter_), max_iter=int(max_iter),
                    loss_curve=list(getattr(model, "loss_curve_", [])),
                )
                st.success(f"✅ Training complete — {model.n_iter_} epochs. "
                           "Details are in the 📊 Results & Metrics tab.")
                _summary(st.session_state["ann2"])

    # ── TAB 2: PREDICT ────────────────────────────────────────────────────────
    with tab_predict:
        res = st.session_state.get("ann2")
        if res is None:
            st.warning("Train the model first in the ⚙️ Train Model tab.")
        else:
            st.info(f"Mode: **{res['mode']}**  ·  inputs: "
                    f"**{', '.join(LABEL[f] for f in res['features'])}**")
            st.caption("Computational power uses the SEDA settings that were active at training.")
            st.markdown("**Enter new input values:**")
            i1, i2, i3 = st.columns(3)
            irr = i1.number_input("Irradiance (W/m²)", 0.0, 1500.0, 800.0, 1.0, key="ann2_p_irr")
            pvt = i2.number_input("PV Temperature (°C)", -20.0, 120.0, 45.0, 0.1, key="ann2_p_pvt")
            amb = i3.number_input("Ambient Temp (°C)", -20.0, 60.0, 30.0, 0.1, key="ann2_p_amb")

            if st.button("🔮 PREDICT", key="ann2_predict", **FW):
                env1 = pd.DataFrame({"irradiance": [irr], "pv_temp": [pvt], "ambient_temp": [amb]})
                comp1 = seda_power(env1, res["cfg"])
                X1 = build_X(env1, comp1, res["env_feats"], res["mode"])[res["features"]]
                p_ann = max(float(predict_ann(res["model"], res["sx"], res["sy"],
                                              X1.to_numpy(float))[0]), 0.0)
                p_sed = float(comp1[0])
                st.markdown(f"""
                <div class="res-box res-calculated" style="text-align:center;padding:2rem">
                    <div class="res-title" style="color:#FFB800;font-size:1rem">🤖 ANN PREDICTED POWER OUTPUT</div>
                    <div class="res-val" style="color:#FFB800;font-size:2.5rem">{p_ann:.2f} W</div>
                    <div class="res-formula">Mode: {res['mode']}</div>
                </div>""", unsafe_allow_html=True)
                r1, r2, r3 = st.columns(3)
                r1.metric("SEDA computational", f"{p_sed:.2f} W")
                d = pct_diff(p_sed, p_ann)
                r2.metric("ANN vs SEDA", f"{d:+.2f} %" if d is not None else "—")
                r3.metric("Percentage of Pmax", f"{p_ann / res['p_rated'] * 100:.1f} %")

            st.markdown("---")
            st.markdown('<p class="sec">📅 BATCH PREDICTION — ALL DAYS</p>', unsafe_allow_html=True)
            st.caption("Predict many records at once and compare with SEDA and (if available) measured power.")
            b_src = st.radio("Input source", ["Field records in database", "Upload file (CSV / Excel)"],
                             horizontal=True, key="ann2_bsrc")
            df_in = None
            if b_src.startswith("Field"):
                df_in = data.copy()
                df_in["x_label"] = df_in["recorded_at"] if "recorded_at" in df_in.columns \
                    else np.arange(1, len(df_in) + 1)
            else:
                st.caption("Columns needed: irradiance, pv_temp, ambient_temp.  "
                           "Optional: date (or recorded_at), power_measured.")
                up = st.file_uploader("Upload file", type=["csv", "xlsx"], key="ann2_bup")
                if up is not None:
                    try:
                        df_in = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
                        df_in.columns = [str(c).strip().lower() for c in df_in.columns]
                        tcol = next((c for c in ("recorded_at", "date") if c in df_in.columns), None)
                        df_in["x_label"] = (pd.to_datetime(df_in[tcol], errors="coerce")
                                            if tcol else np.arange(1, len(df_in) + 1))
                    except Exception as e:
                        st.error(f"Could not read file: {e}")
                        df_in = None

            if df_in is not None and not df_in.empty:
                need = ["irradiance", "pv_temp", "ambient_temp"]
                miss = [c for c in need if c not in df_in.columns]
                if miss:
                    st.error(f"Missing column(s): {', '.join(miss)}")
                else:
                    for c in need:
                        df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
                    df_b = df_in.dropna(subset=need).reset_index(drop=True)
                    comp_b = seda_power(df_b, res["cfg"])
                    Xb = build_X(df_b, comp_b, res["env_feats"], res["mode"])[res["features"]]
                    ann_b = np.clip(predict_ann(res["model"], res["sx"], res["sy"],
                                                Xb.to_numpy(float)), 0, None)

                    out = pd.DataFrame({"Date / Day": df_b["x_label"].values})
                    for c in need:
                        out[c] = df_b[c].values
                    out["SEDA Computational (W)"] = np.round(comp_b, 2)
                    out["ANN Predicted (W)"] = np.round(ann_b, 2)
                    has_meas = "power_measured" in df_b.columns and \
                        pd.to_numeric(df_b["power_measured"], errors="coerce").notna().any()
                    if has_meas:
                        pm = pd.to_numeric(df_b["power_measured"], errors="coerce").to_numpy(float)
                        out["Measured (W)"] = pm
                        out["Error (%)"] = [round((m - p) / p * 100, 2)
                                            if (p != 0 and not np.isnan(m)) else np.nan
                                            for m, p in zip(pm, ann_b)]

                    fig_b = go.Figure()
                    if has_meas:
                        fig_b.add_trace(go.Scatter(x=out["Date / Day"], y=out["Measured (W)"],
                                                   name="Measured", mode="lines+markers",
                                                   line=dict(color="#00D4FF", width=2)))
                    fig_b.add_trace(go.Scatter(x=out["Date / Day"], y=out["SEDA Computational (W)"],
                                               name="SEDA Computational", mode="lines",
                                               line=dict(color="#00E676", width=1.5, dash="dot")))
                    fig_b.add_trace(go.Scatter(x=out["Date / Day"], y=out["ANN Predicted (W)"],
                                               name="ANN Predicted", mode="lines+markers",
                                               line=dict(color="#FFB800", width=2, dash="dash")))
                    _style(fig_b, "Power — Measured vs SEDA vs ANN", 350, "Date", "Power Output (W)")
                    st.plotly_chart(fig_b, key="ann2_batch_fig", **FW)
                    st.dataframe(out, hide_index=True, **FW)
                    if b_src.startswith("Field"):
                        st.warning("Records used to train the model look more accurate than they really are. "
                                   "For a fair check use the test-set results in 📊 Results & Metrics.")
                    buf_b = io.BytesIO()
                    with pd.ExcelWriter(buf_b, engine="openpyxl") as w:
                        out.to_excel(w, sheet_name="Batch Prediction", index=False)
                    st.download_button("⬇ Export Batch Prediction to Excel", buf_b.getvalue(),
                                       f"ANN_Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       key="ann2_batch_dl", **FW)

    # ── TAB 3: RESULTS ────────────────────────────────────────────────────────
    with tab_results:
        res = st.session_state.get("ann2")
        if res is None:
            st.warning("Train the model first in the ⚙️ Train Model tab.")
        else:
            if res["mode"] != st.session_state.get("ann2_mode"):
                st.warning("These results are from a different training mode. "
                           "Press **Train ANN model** to update.")
            st.markdown('<p class="sec">📊 MODEL PERFORMANCE METRICS</p>', unsafe_allow_html=True)
            st.caption(f"Mode: {res['mode']}  ·  Target: {res['target_label']}  ·  "
                       f"Trained on {res['n_train']} · tested on {res['n_test']} records  ·  "
                       f"Architecture: {list(res['hidden'])}")
            _summary(res)

            rows = {"ANN vs Measured": compute_metrics(res["meas_te"], res["p_te"]),
                    "ANN vs SEDA computational": compute_metrics(res["comp_te"], res["p_te"]),
                    "Baseline: SEDA vs Measured": compute_metrics(res["meas_te"], res["comp_te"])}
            if res["mode"] == MODE_B:
                rows = {"ANN vs blended target": res["m_te"], **rows}
            st.markdown("**Test-set accuracy against each reference**")
            st.dataframe(pd.DataFrame(rows).T.style.format("{:.4f}"), **FW)
            st.caption("The baseline row shows how far SEDA is from the measured data before the ANN is used.")
            st.markdown("**Train vs test (against the training target)**")
            st.dataframe(pd.DataFrame({"Train": res["m_tr"], "Test": res["m_te"]}).T
                         .style.format("{:.4f}"), **FW)

            st.markdown('<p class="sec">📈 MEASURED vs SEDA vs ANN (TEST SET)</p>', unsafe_allow_html=True)
            order = np.argsort(res["comp_te"])
            x = np.arange(len(order))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=res["meas_te"][order], name="Measured",
                                     line=dict(color="#00D4FF", width=2)))
            fig.add_trace(go.Scatter(x=x, y=res["comp_te"][order], name="SEDA Computational",
                                     line=dict(color="#00E676", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=x, y=res["p_te"][order], name="ANN Predicted",
                                     line=dict(color="#FFB800", width=2, dash="dash")))
            _style(fig, "Test samples sorted by SEDA power", 350, "Test sample", "Power Output (W)")
            st.plotly_chart(fig, key="ann2_fig1", **FW)

            st.markdown('<p class="sec">📐 PARITY PLOT</p>', unsafe_allow_html=True)
            allv = np.concatenate([res["meas_te"], res["comp_te"], res["p_te"]])
            lo, hi = float(allv.min()) * 0.95, float(allv.max()) * 1.05
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], name="Ideal line",
                                      line=dict(color="#FFB800", dash="dash", width=2)))
            fig2.add_trace(go.Scatter(x=res["meas_te"], y=res["comp_te"], mode="markers",
                                      name="SEDA vs measured",
                                      marker=dict(color="#00E676", size=7, opacity=0.6)))
            fig2.add_trace(go.Scatter(x=res["meas_te"], y=res["p_te"], mode="markers",
                                      name="ANN vs measured",
                                      marker=dict(color="#00D4FF", size=7, opacity=0.8)))
            _style(fig2, "Parity — points on the gold line = perfect agreement", 380,
                   "Measured Power (W)", "Predicted / Computed Power (W)")
            st.plotly_chart(fig2, key="ann2_fig2", **FW)

            e1, e2 = st.columns(2)
            with e1:
                fig3 = go.Figure(go.Histogram(x=res["y_te"] - res["p_te"], nbinsx=30,
                                              marker_color="#FF6B00"))
                _style(fig3, "Error distribution (target − ANN)", 300, "Error (W)", "Count")
                st.plotly_chart(fig3, key="ann2_fig3", **FW)
            with e2:
                fig4 = go.Figure()
                if res["loss_curve"]:
                    fig4.add_trace(go.Scatter(y=res["loss_curve"], name="Training loss",
                                              line=dict(color="#FF6B00", width=2)))
                    _style(fig4, "Training loss per epoch", 300, "Epoch", "Loss")
                else:
                    _style(fig4, "No loss curve (lbfgs solver)", 300)
                st.plotly_chart(fig4, key="ann2_fig4", **FW)

            st.markdown("---")
            out = pd.DataFrame(res["X_te"], columns=[LABEL[f] for f in res["features"]])
            out["P_measured (W)"] = res["meas_te"]
            out["P_SEDA (W)"] = res["comp_te"]
            out["P_ANN (W)"] = res["p_te"]
            out["Error ANN − Measured (W)"] = out["P_ANN (W)"] - out["P_measured (W)"]
            out["Error ANN − SEDA (W)"] = out["P_ANN (W)"] - out["P_SEDA (W)"]
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                out.to_excel(w, sheet_name="Test Results", index=False)
                pd.DataFrame(rows).T.reset_index().rename(columns={"index": "Comparison"}) \
                    .to_excel(w, sheet_name="Metrics Summary", index=False)
            x1, x2 = st.columns(2)
            x1.download_button("⬇ Export ANN Results to Excel", buf.getvalue(),
                               f"ANN_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="ann2_dl_x", **FW)
            mb = io.BytesIO()
            joblib.dump({k: res[k] for k in ("model", "sx", "sy", "features", "env_feats",
                                             "mode", "w", "target_label", "cfg")}, mb)
            x2.download_button("⬇ Model ANN (.joblib)", mb.getvalue(), "ann_solar_model.joblib",
                               "application/octet-stream", key="ann2_dl_m", **FW)


# ═══════════════════════════════════════════════════════════════════════════════
#  [A] STANDALONE ONLY — page shell
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="solar-header">
  <h1>☀ SOLAR PV ANN — v2</h1>
  <p>MEASURED vs SEDA COMPUTATIONAL · ANN TEST BENCH FOR app.py</p>
</div>
""", unsafe_allow_html=True)

render_ann_page(load_standalone_data())
