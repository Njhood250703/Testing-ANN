# ANN PV Power Prediction

Streamlit GUI to predict PV module power output using an Artificial Neural Network (ANN).

- Inputs: irradiance, ambient temperature, PV temperature (PV rating fixed at 605 W)
- Output: predicted power (W)
- Upload measured site data (CSV) in the sidebar to train on real data

Run locally:

    pip install -r requirements.txt
    streamlit run app.py
