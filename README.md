# ☀️ Solar PV Power Output Prediction using ANN

A Streamlit app that trains an Artificial Neural Network (ANN) to predict the power output of a solar PV module from field measurements, and checks how close the ANN result is to both the **measured** power and the **computational** power.

## Concept

| Quantity | Role | Source |
|---|---|---|
| **Measured power output** | Target the ANN must learn | Field measurement (column in your data) |
| **Computational power output** | Reference the ANN should stay close to | Calculated in the SEDA app (column in your data) |
| **ANN power output** | Result of the model | Predicted by this app |

The ANN is trained on the measured power. It is then judged against two references:

1. **Near the target:** ANN vs measured (the main goal).
2. **Near the computational value:** ANN vs computational.

When the measured and computational values differ a lot, the ANN cannot be equal to both. The app shows this gap as a **baseline** (computational vs measured), so the result is easy to interpret.

## Features

- Upload your own CSV / Excel data, or use built-in simulated data for testing
- Choose the measured (target) column, the computational column and the ANN input columns
- Optionally use the computational power as an extra ANN input
- Configurable ANN: hidden layers (1–5), neurons per layer, activation, solver, learning rate, epochs, L2 regularization, early stopping
- Accuracy targets: minimum R² and maximum MAPE vs measured, maximum MAPE vs computational
- Metrics: R², RMSE, MAE, MAPE, with a comparison table and baseline row
- Plots: predicted vs measured, error distribution, training loss curve, measured vs computational vs ANN
- Manual prediction and batch prediction (CSV / Excel)
- Export test results (CSV) and the trained model (`.joblib`)

## Data format

A CSV or Excel file with numeric columns only, one row per record. Column names are free, since you choose them in the app. Example:

| irradiance | pv_temp | ambient_temp | p_computational | p_measured |
|---|---|---|---|---|
| 812.4 | 54.3 | 31.2 | 468.1 | 452.7 |
| 640.9 | 48.1 | 30.4 | 372.5 | 361.0 |

- At least 3 numeric columns are needed: measured power, computational power and one or more inputs.
- At least 10 valid records are needed. More than 100 is recommended.
- Typical inputs: solar irradiance (W/m²), PV temperature (°C), ambient temperature (°C).
- Computational power comes from the SEDA app, so combine its export with your field data into one file.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app_v3.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** and choose **Deploy a public app from GitHub**.
4. Set the repository, the branch (`main`) and the main file path to `app_v3.py`.
5. Click **Deploy**.

Every `git push` redeploys the app automatically.

## How to use the app

1. **Data:** pick simulated data or upload a file in the sidebar.
2. **Columns:** choose the measured column, the computational column and the ANN inputs.
3. **ANN settings:** set layers, neurons, solver and the other training options in the sidebar.
4. **Train:** open the *Training & Results* tab and press **Train ANN model**.
5. **Read the result:** check the accuracy against the measured target and the computational value.
6. **Predict:** use the *Prediction* tab for a single case or a batch file.
7. **Export:** download the test results and the trained model from the *Export* tab.

## Reading the results

| Status | Meaning |
|---|---|
| ✅ Close to both | ANN meets the target accuracy vs measured and the computational limit |
| ✅ Close to the target | ANN meets the target accuracy vs measured, but differs from the computational value by more than the limit |
| ❌ Not yet close | ANN does not meet the target accuracy vs measured. Change the architecture, the solver or add data |

Tips:

- **Small datasets (under 100 records):** use the `lbfgs` solver with 1 hidden layer of 8–16 neurons. Large networks overfit easily.
- **Overfitting warning:** a large gap between training and test R² means the network is too large for the data. Reduce the neurons or raise alpha.
- **Pull ANN toward computational (Advanced):** blends part of the computational power into the training target. Keep it at 0 to train on the measured target only.

## Project files

| File | Purpose |
|---|---|
| `app_v3.py` | Main Streamlit app |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

Earlier versions of the app (`app.py`, `app_v2.py`), if present, are kept for reference only.

## Notes

- The simulated data is for testing the app only. Use real field data for project results.
- The ANN is a scikit-learn `MLPRegressor`. Inputs and output are standardized before training.
- Predictions are clipped at 0 W in the prediction tab, because power cannot be negative.
