<h1 align="center">Oil Price Prediction</h1>

<p align="center">
  <b>Time-Series Forecasting with LSTM & Simple RNN</b><br>
  Developed during my internship at <b>Saudi Aramco</b>
</p>

<p align="center">
  <img src="results/lstm_rnn_comparison.png" width="90%" alt="Actual vs. predicted oil prices using LSTM and Simple RNN">
</p>

<p align="center">
  <a href="[STREAMLIT_APP_URL](https://yasir-oil-price-prediction.streamlit.app/)"><strong>Try the Live App ↗</strong></a>
  &nbsp; · &nbsp;
  <a href="Oil_Price_Prediction_LSTM_RNN_Comparison.ipynb">Explore the Notebook</a>
  &nbsp; · &nbsp;
  <a href="Final%20Report%20Oil%20Price%20Prediction.pdf">Project Report</a>
</p>

---

## Overview

An end-to-end deep learning project developed during my internship at **Saudi Aramco**, comparing **LSTM** and **Simple RNN** for next-observation oil closing-price forecasting.

The project combines data preprocessing, model training, evaluation, and an interactive **Streamlit dashboard** that runs inference using the saved models.

## Interactive Dashboard

| Page | Features |
|---|---|
| **Overview** | Historical prices, dataset coverage, and performance summary |
| **Historical Evaluation** | Actual vs. predicted prices, date filters, model comparisons, and CSV export |
| **Interactive Forecast** | Generate a next-observation forecast using either trained model |
| **About** | Project background, methodology, and limitations |

## Data & Methodology

| Item | Details |
|---|---|
| **Data** | Historical crude oil dataset sourced from Kaggle |
| **Coverage** | 6,073 observations · August 2000–October 2024 |
| **Input** | Previous 60 closing-price observations |
| **Target** | Next observation’s closing price, in USD per barrel |
| **Split** | Chronological: 70% training, 15% validation, 15% testing |
| **Scaling** | MinMaxScaler fitted on training data only |
| **Training** | Adam · MSE loss · Batch size 32 · Up to 50 epochs |
| **Early stopping** | Independent callback per model, with best weights restored |

Models were trained in **Google Colab**. The dashboard loads the saved models and scaler to generate predictions without retraining.

## Results

Performance on **912 test observations**, from March 2021 to October 2024. MAE and RMSE are measured in USD per barrel.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| LSTM | 1.7431 | 2.3079 | 0.9634 |
| Simple RNN | 1.8538 | 2.4342 | 0.9593 |
| Persistence baseline | **1.5011** | **2.0710** | **0.9705** |

**LSTM achieved approximately 5% lower RMSE than Simple RNN in this run.** However, neither model outperformed the persistence baseline, which predicts that the next closing price equals the latest observed close.

## Tech Stack

**Python · TensorFlow · Keras · scikit-learn · Pandas · NumPy · Plotly · Matplotlib · Streamlit**

## Run Locally

Use **Python 3.12** and install the application dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The repository includes the data, trained models, scaler, and evaluation outputs needed to run the dashboard.

## Project Files

| Path | Contents |
|---|---|
| `app.py` | Streamlit application entry point |
| `app_pages/` | Dashboard pages |
| `oil_dashboard/` | Data processing, inference, charts, and shared UI |
| `artifacts/` | Trained models, scaler, metrics, and predictions |
| `data/` | Historical price data |
| `results/` | Evaluation figures |
| `assets/` | Dashboard illustration |
| `scripts/` | Model import and verification utilities |
| `Oil_Price_Prediction_LSTM_RNN_Comparison.ipynb` | Training and evaluation notebook |

## Scope

Forecasts use a **static historical dataset** and cover **one observation ahead**. The models use closing prices only and do not incorporate market news or economic indicators. Results are intended for learning and demonstration, not investment decisions.

This is a personal project showcase, not an official Saudi Aramco product or endorsement.

---

<p align="center">
  <b>Yasir Abdullah Aladwani</b><br>
  B.Sc. in Artificial Intelligence · AI Engineer
</p>
