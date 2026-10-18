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
TARGET_LABEL = "Power output (measured)"
PV_RATING = 605.0  # W (fixed, single module)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})


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
                         max_iter=2000, early_stopping=len(X_tr) >= 50,
                         random_state=42)
    model.fit(X_tr_s, y_tr)
    pred_te = model.predict(X_te_s)
    pred_tr = model.predict(X_tr_s)
    metrics = {
        "r2_train": r2_score(y_tr, pred_tr),
        "r2_test": r2_score(y_te, pred_te) if len(y_te) >= 2 else float("nan"),
        "rmse_test": float(np.sqrt(mean_squared_error(y_te, pred_te))),
        "n_train": len(X_tr), "n_test": len(X_te),
        "loss_curve": list(model.loss_curve_),
        "val_curve": list(getattr(model, "validation_scores_", None) or []),
    }
    return model, scaler, metrics, y_te.values, pred_te, X_te["irradiance"].values


def _std_name(col):
    """Map a column name from the user's file to the standard name used by the app."""
    c = str(col).lower().replace("\n", " ").strip()
    if "irradiance" in c:
        return "irradiance"
    if "ambient" in c:
        return "ambient_temp"
    if "pv temp" in c or "pv_temp" in c or "module temp" in c:
        return "pv_temp"
    if "rating" in c:
        return "pv_rating"
    if c in ("power", "power output (measured)") or ("power" in c and "measured" in c):
        return "power"
    return str(col)


def load_table(file):
    """Read CSV or Excel. For Excel, find the real header row automatically
    (report files often have title rows above the table)."""
    name = getattr(file, "name", "").lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(file)
    else:
        probe = pd.read_excel(file, header=None)
        header_row = 0
        for i in range(min(len(probe), 30)):
            row = " ".join(str(v).lower() for v in probe.iloc[i].tolist())
            if "irradiance" in row:
                header_row = i
                break
        file.seek(0)
        raw = pd.read_excel(file, header=header_row)
    raw.columns = [_std_name(c) for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.duplicated()]
    return raw


# ---------------- Plot helpers ----------------
def draw_ann(n1, n2, max_nodes=8):
    """Simple ANN structure diagram: 4 inputs -> n1 -> n2 -> 1 output."""
    sizes = [4, n1, n2, 1]
    shown = [min(s, max_nodes) for s in sizes]
    xs = [0, 1, 2, 3]
    pos = []
    for x, k in zip(xs, shown):
        ys = np.linspace((k - 1) / 2, -(k - 1) / 2, k) if k > 1 else np.array([0.0])
        pos.append([(x, y) for y in ys])

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for L in range(3):                      # connections
        for (x1, y1) in pos[L]:
            for (x2, y2) in pos[L + 1]:
                ax.plot([x1, x2], [y1, y2], color="0.75", lw=0.5, zorder=1)
    colors = ["#4C9BE8", "#F2A03D", "#F2A03D", "#E85C5C"]
    for L in range(4):                      # nodes
        for (x, y) in pos[L]:
            ax.scatter(x, y, s=220, color=colors[L], edgecolor="k", lw=0.6, zorder=2)
        if sizes[L] > max_nodes:            # too many neurons to draw
            ax.text(xs[L], -(shown[L] - 1) / 2 - 0.9,
                    f"... ({sizes[L]} neurons)", ha="center", fontsize=7)

    names = ["Irradiance", "Ambient temp", "PV rating", "PV temp"]
    for (x, y), nm in zip(pos[0], names):
        ax.text(x - 0.12, y, nm, ha="right", va="center", fontsize=8)
    ax.text(3.12, 0, "Power output\n(ANN)", ha="left", va="center", fontsize=8)

    top = max((shown[1] - 1) / 2, (shown[2] - 1) / 2, 1.5) + 0.9
    titles = ["Input layer", f"Hidden layer 1\n({n1} neurons)",
              f"Hidden layer 2\n({n2} neurons)", "Output layer"]
    for x, t in zip(xs, titles):
        ax.text(x, top, t, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlim(-1.2, 4.1)
    ax.set_ylim(-top - 0.8, top + 1.0)
    ax.axis("off")
    fig.tight_layout()
    return fig


def small_fig():
    return plt.subplots(figsize=(3.6, 2.8))


# ---------------- Sidebar: data ----------------
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload measured data (CSV or Excel)",
                              type=["csv", "xlsx", "xls"])
st.sidebar.caption("Needs columns for irradiance, ambient temp, PV temp and "
                   "measured power (target). Names like 'Irradiance (W/m²)' and "
                   "'Power Measured (W)' are recognised. If PV rating is missing, "
                   "605 W is used.")
test_size = st.sidebar.slider("Test data (%)", 10, 40, 20) / 100

if up is not None:
    df = load_table(up)
    if "pv_rating" not in df.columns:
        df["pv_rating"] = PV_RATING
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"Could not find these columns in the file: {missing}. "
                 f"Columns found: {list(df.columns)}")
        st.stop()
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 10:
        st.error(f"Only {len(df)} usable rows. Need more data to train an ANN.")
        st.stop()
    data_label = f"Measured data (uploaded, {len(df)} rows)"
else:
    df = demo_data()
    data_label = "DEMO data (synthetic)"

# ---------------- Main page (single section) ----------------
st.title("ANN PV Power Prediction")
st.caption(f"Bifacial PV module, {PV_RATING:.0f} W rating  |  Data: {data_label}")
if up is None:
    st.warning("Using synthetic DEMO data. Results are not real. "
               "Upload your measured site data (CSV or Excel) in the sidebar.")
elif len(df) < 50:
    st.warning(f"Only {len(df)} rows of measured data. This is too few for a "
               "reliable ANN: the test set is tiny, so R² and the graphs can be "
               "misleading. Collect more measurements (ideally a few hundred).")

with st.expander("Show data"):
    st.write(f"{len(df)} rows")
    st.dataframe(df.rename(columns={TARGET: TARGET_LABEL}).head(100))

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
_, mid, _ = st.columns([1, 3, 1])
mid.pyplot(draw_ann(n1, n2))
st.caption("ANN structure (activation: ReLU in both hidden layers). "
           "Only the first 8 neurons of a layer are drawn.")

(model, scaler, m, y_te, pred_te, irr_te) = train_model(df, n1, n2, test_size)

st.divider()

# ===== 3. OUTPUT ANN =====
st.subheader("3. Output ANN")
x_new = pd.DataFrame([[irr, amb, PV_RATING, pvt]], columns=FEATURES)
p = max(float(model.predict(scaler.transform(x_new))[0]), 0.0)  # no negative power
o1, o2 = st.columns(2)
o1.metric("Power output (ANN)", f"{p:.1f} W")
o2.metric("Percent of rating", f"{p / PV_RATING * 100:.1f} %")
lo_i, hi_i = df["irradiance"].min(), df["irradiance"].max()
if not (lo_i <= irr <= hi_i):
    st.info(f"Irradiance is outside the training range ({lo_i:.0f} to {hi_i:.0f} "
            "W/m²). Prediction may be unreliable.")

st.divider()

# ===== 4. TARGET OUTPUT (MEASURED VALUE) =====
st.subheader(f"4. Target output: {TARGET_LABEL}")
meas = st.number_input(f"{TARGET_LABEL}, W", 0.0, 1000.0, 0.0, 1.0,
                       help="Enter the measured value at the same time as the "
                            "inputs above. Leave 0 to skip.")
if meas > 0:
    err = meas - p
    t1, t2 = st.columns(2)
    t1.metric("Error (measured - ANN)", f"{err:.1f} W")
    t2.metric("Error (%)", f"{abs(err) / meas * 100:.1f} %")

st.markdown("**ANN performance on test data**")
a, b, c = st.columns(3)
a.metric("R² (test)", f"{m['r2_test']:.4f}")
b.metric("RMSE (test)", f"{m['rmse_test']:.2f} W")
c.metric("R² (train)", f"{m['r2_train']:.4f}")
st.caption(f"Train samples: {m['n_train']}  |  Test samples: {m['n_test']}. "
           "R² train and test should be close. A large gap means overfitting.")

st.markdown("**ANN graphs**")
lo, hi = float(y_te.min()), float(y_te.max())
resid = y_te - pred_te

g1, g2, g3 = st.columns(3)
with g1:                                        # 1. Actual vs predicted
    fig, ax = small_fig()
    ax.scatter(y_te, pred_te, s=8)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Perfect")
    ax.set_xlabel(f"{TARGET_LABEL} (W)")
    ax.set_ylabel("Power output (ANN) (W)")
    ax.set_title("Measured vs ANN")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
with g2:                                        # 2. Training loss curve
    fig, ax = small_fig()
    ax.plot(m["loss_curve"], color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss curve")
    ax.set_yscale("log")
    fig.tight_layout()
    st.pyplot(fig)
with g3:                                        # 3. Validation score curve
    if m["val_curve"]:
        fig, ax = small_fig()
        ax.plot(m["val_curve"], color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation R²")
        ax.set_title("Validation score curve")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Validation curve is not available for small data "
                "(fewer than 50 rows).")

g4, g5, g6 = st.columns(3)
with g4:                                        # 4. Error histogram
    fig, ax = small_fig()
    ax.hist(resid, bins=20, color="tab:orange", edgecolor="k", lw=0.4)
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Error: measured - ANN (W)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    fig.tight_layout()
    st.pyplot(fig)
with g5:                                        # 5. Error vs irradiance
    fig, ax = small_fig()
    ax.scatter(irr_te, resid, s=8, color="tab:purple")
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Irradiance (W/m²)")
    ax.set_ylabel("Error (W)")
    ax.set_title("Error vs irradiance")
    fig.tight_layout()
    st.pyplot(fig)
with g6:                                        # 6. Test samples, measured vs ANN
    fig, ax = small_fig()
    k = min(50, len(y_te))
    ax.plot(range(k), y_te[:k], "o-", ms=3, lw=1, label="Measured")
    ax.plot(range(k), pred_te[:k], "s--", ms=3, lw=1, label="ANN")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Power (W)")
    ax.set_title("First test samples")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
