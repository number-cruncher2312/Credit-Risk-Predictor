# Credit Risk Analytics Dashboard

An end-to-end credit risk analytics platform built with Python and Streamlit that combines machine learning, Basel IRB regulatory capital calculations, portfolio analytics, stress testing, and model drift monitoring into a single interactive dashboard.

The project demonstrates how modern credit risk models can be developed, validated, monitored, and interpreted throughout their lifecycle.

###  [View Live App](https://credit-risk-predictor-kzwx6mz7hky2xc3ucvqgjj.streamlit.app/)
---

## Features

### Machine Learning Credit Risk Model

- Train and evaluate an XGBoost credit risk model
- Probability of Default (PD) prediction
- Feature importance using SHAP
- Model performance metrics
- Interactive prediction interface

---

### Portfolio Risk Analytics

- Portfolio-level Expected Loss (EL)
- Probability of Default (PD)
- Loss Given Default (LGD)
- Exposure at Default (EAD)
- Portfolio summaries
- Risk visualizations

---

### Basel IRB Regulatory Capital

Implements the Basel Internal Ratings-Based (IRB) capital framework.

Features include:

- Asset Correlation (R)
- Maturity Adjustment (b)
- Capital Requirement (K)
- Regulatory Capital
- Risk-Weighted Assets (RWA)

Portfolio analytics:

- Loan-level Basel metrics
- Top RWA contributors
- Portfolio concentration analysis
- Interactive capital dashboard

---

### Stress Testing

Simulate deteriorating credit environments by applying configurable PD stress scenarios.

Includes:

- Portfolio-wide PD stress multiplier
- Live Basel recalculation
- Base vs stressed comparison
- Capital impact
- Expected Loss impact
- RWA impact

---

### Model Drift Monitoring

Production-style monitoring framework for model health.

Implemented metrics:

#### Population Stability Index (PSI)

- Distribution comparison
- Drift severity classification
- Historical trend tracking

#### Characteristic Stability Index (CSI)

- Feature-level drift analysis
- SHAP-driven feature monitoring
- Drift prioritization

Monitoring includes:

- Green / Yellow / Red status classification
- Configurable thresholds
- Investigation recommendations
- Trend visualizations
- Distribution comparison charts

---

## Project Structure

```
.
├── app.py
├── train_model.py
├── reg_capital.py
├── drift.py
├── drift_simulation.py
├── drift_monitoring_tab.py
├── models.py
├── config.py
├── assets/
├── data/
└── model/
```

---

## Tech Stack

- Python
- Streamlit
- XGBoost
- SHAP
- Pandas
- NumPy
- Scikit-learn
- Plotly
- SciPy

---

## Dashboard Modules

- Credit Risk Prediction
- Portfolio Analytics
- Basel IRB Capital
- Stress Testing
- Model Drift Monitoring

---

## Future Improvements

- Automated retraining pipeline
- Champion-Challenger model comparison
- Time-series portfolio monitoring
- Explainability reports
- Additional drift metrics
- IFRS 9 Expected Credit Loss implementation
- Scenario management
- Model governance dashboard

---

## Running Locally

Clone the repository

```bash
git clone git@github.com:number-cruncher2312/Credit-Risk-Predictor.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Educational Purpose

This project was developed as a practical implementation of concepts commonly used in quantitative risk management and banking, including:

- Credit Risk Modeling
- Basel III Internal Ratings-Based (IRB) Framework
- Expected Loss Modeling
- Regulatory Capital
- Stress Testing
- Model Monitoring
- Population Stability Index (PSI)
- Characteristic Stability Index (CSI)
- Explainable AI using SHAP

---

## License

MIT License
