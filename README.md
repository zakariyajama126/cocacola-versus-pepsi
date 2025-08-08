KO vs PEP Stock Performance Predictor
Machine learning model that predicts which stock will perform better monthly: Coca-Cola (KO) or PepsiCo (PEP).
Performance

Test Accuracy: 75-80% on monthly predictions
Cross-validation: 5-fold CV with hyperparameter optimization
Timeframe: 2018-2024 data with 2022+ held for testing

Features
The model uses 60+ engineered features including:

Technical Indicators: RSI, Bollinger Bands, Moving Averages, Momentum
Relative Analysis: Price ratios, volatility comparisons, RSI differences
Market Context: SPY returns, VIX volatility, market regime indicators
Historical Performance: 6-month lagged returns and rolling statistics

Algorithms
Automatically selects best performer from:

XGBoost (gradient boosting)
Random Forest
Logistic Regression

Uses GridSearchCV for hyperparameter optimization across all models.

Sample Output
Enhanced KO vs PEP Prediction Model
==================================================
Downloading comprehensive dataset...
Engineering advanced features...
Dataset shape: (72, 61)
Target distribution: {1: 38, 0: 34}
Training samples: 48 | Test samples: 24

Training and optimizing models...
   Optimizing XGBoost...
     Best CV score: 0.729
   Optimizing Random Forest...
     Best CV score: 0.708
   Optimizing Logistic Regression...
     Best CV score: 0.688

Best model: XGBoost (CV score: 0.729)

Prediction Accuracy: 79.17% (19/24)

Top 10 Most Important Features:
    1. KO_RSI                   : 0.0847
    2. Relative_Price_Ratio     : 0.0731
    3. Market_VIX_Change        : 0.0629
    4. KO_return_lag_2          : 0.0584
    5. PEP_Momentum_10          : 0.0541

Recent Predictions (with confidence):
   ✅ 2023-08: PEP won | Predicted: PEP wins (conf: 73.2%)
   ❌ 2023-09: KO won | Predicted: PEP wins (conf: 62.1%)
   ✅ 2023-10: KO won | Predicted: KO wins (conf: 81.4%)
   ✅ 2023-11: PEP won | Predicted: PEP wins (conf: 69.7%)

Model Summary:
   • Best Algorithm: XGBoost
   • Features Used: 60
   • Cross-validation Score: 0.729
   • Test Accuracy: 0.792
Quick Start
bashpip install yfinance pandas numpy xgboost scikit-learn
python stock_predictor.py
Files

stock_predictor.py - Main prediction script
requirements.txt - Package dependencies
README.md - This file

Data Sources

Stock prices: Yahoo Finance (yfinance)
Market data: S&P 500 (SPY) and VIX volatility index
Timeframe: 2015-2024 for feature engineering, 2018+ for modeling
