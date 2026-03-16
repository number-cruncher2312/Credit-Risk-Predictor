# 📊 Credit Risk Predictor

An end-to-end credit risk modeling project using **XGBoost** to predict the probability of a borrower experiencing financial distress. 

### 🚀 [View Live App](https://credit-risk-predictor-kzwx6mz7hky2xc3ucvqgjj.streamlit.app/)

---

## 📈 Model Performance
Based on the *Kaggle "Give Me Some Credit"* dataset, the model achieves strong predictive power:

- **AUC-ROC**: `0.8653`
- **Gini Coefficient**: `0.7306`
- **KS Statistic**: `0.5766`

---

## 🛠️ Tech Stack
- **Modeling**: XGBoost, Scikit-Learn
- **Dashboard**: Streamlit, Plotly
- **Data Engineering**: Pandas, NumPy
- **Persistence**: Joblib

## 📁 Repository Structure
- `train_model.py`: End-to-end training pipeline (Cleaning -> Imputation -> Oversampling -> Training -> Evaluation).
- `app.py`: Interactive Streamlit dashboard.
- `model/`: Serialized XGBoost model (`xgb_model.pkl`).
- `requirements.txt`: Environment dependencies for deployment.

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/number-cruncher2312/Credit-Risk-Predictor.git
   cd Credit-Risk-Predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

---
*Created with ❤️ by Antigravity.*
