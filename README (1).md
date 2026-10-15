# ☀️ Solar PV Power Output Prediction using ANN

A Streamlit GUI for training an Artificial Neural Network (MLP) to predict solar panel power output (e.g. a 605 W module).

## Features
- Inputs: Solar Irradiance, PV Temperature, Ambient Temperature (selectable)
- ANN settings: number of hidden layers, neurons per layer, activation, solver, learning rate, epochs, alpha, early stopping
- Targets: power output column, plus minimum R² and maximum MAPE targets
- Simulated demo data or upload your own field-data CSV
- Metrics: R², RMSE, MAE, MAPE; actual vs predicted, residual and loss-curve plots
- Manual and batch (CSV) prediction; export results and trained model (.joblib)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV format
Numeric columns only, for example:
`Solar Irradiance (W/m2), PV Temperature (C), Ambient Temperature (C), P_output (W)`

## Deploy
Push to GitHub, then deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) with `app.py` as the main file.
