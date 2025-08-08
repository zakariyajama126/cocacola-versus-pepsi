import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def calculate_technical_indicators(prices, name):
    """Calculate technical analysis indicators"""
    df = pd.DataFrame(index=prices.index)
    df[name] = prices

    # Simple Moving Averages
    df['SMA_5'] = prices.rolling(window=5).mean()
    df['SMA_20'] = prices.rolling(window=20).mean()

    # Exponential Moving Average
    df['EMA_12'] = prices.ewm(span=12).mean()

    # Relative Strength Index (RSI)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = prices.rolling(window=20).mean()
    rolling_std = prices.rolling(window=20).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / rolling_mean
    df['BB_position'] = (prices - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # Volatility (rolling standard deviation)
    df['Volatility'] = prices.rolling(window=20).std()

    # Price momentum
    df['Momentum_5'] = prices / prices.shift(5) - 1
    df['Momentum_10'] = prices / prices.shift(10) - 1

    return df


def create_relative_features(ko_data, pep_data):
    """Create relative features between KO and PEP"""
    relative_features = pd.DataFrame(index=ko_data.index)

    # Price ratio and its momentum
    relative_features['Price_Ratio'] = ko_data / pep_data
    relative_features['Price_Ratio_MA5'] = relative_features['Price_Ratio'].rolling(5).mean()
    relative_features['Price_Ratio_Momentum'] = relative_features['Price_Ratio'] / relative_features[
        'Price_Ratio'].shift(5) - 1

    # Volatility comparison
    relative_features['Vol_Ratio'] = ko_data.rolling(20).std() / pep_data.rolling(20).std()

    # RSI difference
    ko_rsi = calculate_rsi(ko_data)
    pep_rsi = calculate_rsi(pep_data)
    relative_features['RSI_Diff'] = ko_rsi - pep_rsi

    return relative_features


def calculate_rsi(prices, period=14):
    """Calculate RSI for a price series"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def create_market_features(start_date, end_date):
    """Create broader market context features"""
    # Download market data
    spy = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)['Close']
    vix = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True)['Close']

    market_data = pd.DataFrame(index=spy.index)
    market_data['SPY_Return'] = spy.pct_change()
    market_data['SPY_Volatility'] = spy.rolling(20).std()
    market_data['VIX'] = vix
    market_data['VIX_Change'] = vix.pct_change()

    # Market regime indicators
    market_data['Market_Trend'] = (spy > spy.rolling(50).mean()).astype(int)
    market_data['High_Vol_Regime'] = (vix > vix.rolling(20).mean()).astype(int)

    return market_data


def main():
    print("🚀 Enhanced KO vs PEP Prediction Model")
    print("=" * 50)

    # 1. Download stock data with extended history for better features
    print("📊 Downloading comprehensive dataset...")
    tickers = ['KO', 'PEP']
    start_date = '2015-01-01'  # Extended for better technical indicators
    end_date = '2024-01-01'

    stock_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    prices = stock_data['Close']

    # Download market data
    market_data = create_market_features(start_date, end_date)

    # 2. Calculate technical indicators for both stocks
    print("🔧 Engineering advanced features...")
    ko_tech = calculate_technical_indicators(prices['KO'], 'KO')
    pep_tech = calculate_technical_indicators(prices['PEP'], 'PEP')

    # 3. Create relative features
    relative_features = create_relative_features(prices['KO'], prices['PEP'])

    # 4. Resample everything to monthly
    ko_monthly = ko_tech.resample('M').last()
    pep_monthly = pep_tech.resample('M').last()
    relative_monthly = relative_features.resample('M').last()
    market_monthly = market_data.resample('M').last()

    # 5. Calculate monthly returns and create target
    ko_returns = ko_monthly['KO'].pct_change()
    pep_returns = pep_monthly['PEP'].pct_change()

    # Create comprehensive feature dataset
    feature_data = pd.DataFrame(index=ko_monthly.index)

    # Basic return features (with more lags)
    for lag in range(1, 7):  # Extended to 6 months
        feature_data[f'KO_return_lag_{lag}'] = ko_returns.shift(lag)
        feature_data[f'PEP_return_lag_{lag}'] = pep_returns.shift(lag)

    # Technical indicator features (monthly)
    tech_features = ['RSI', 'BB_width', 'BB_position', 'Volatility', 'Momentum_5', 'Momentum_10']
    for feature in tech_features:
        if feature in ko_monthly.columns:
            feature_data[f'KO_{feature}'] = ko_monthly[feature]
            feature_data[f'PEP_{feature}'] = pep_monthly[feature]
            # Add relative versions
            feature_data[f'{feature}_diff'] = ko_monthly[feature] - pep_monthly[feature]

    # Relative features
    for col in relative_monthly.columns:
        feature_data[f'Relative_{col}'] = relative_monthly[col]

    # Market context features
    for col in market_monthly.columns:
        feature_data[f'Market_{col}'] = market_monthly[col]

    # Rolling statistics features
    for window in [3, 6]:
        feature_data[f'KO_return_rolling_mean_{window}'] = ko_returns.rolling(window).mean()
        feature_data[f'PEP_return_rolling_mean_{window}'] = pep_returns.rolling(window).mean()
        feature_data[f'KO_return_rolling_std_{window}'] = ko_returns.rolling(window).std()
        feature_data[f'PEP_return_rolling_std_{window}'] = pep_returns.rolling(window).std()

    # Target variable
    feature_data['target'] = (ko_returns > pep_returns).astype(int)

    # 6. Clean data and prepare for modeling
    # Start from 2018 to have enough history for features
    feature_data = feature_data.loc['2018-01-01':].dropna()

    print(f"📈 Dataset shape: {feature_data.shape}")
    print(f"📊 Target distribution: {feature_data['target'].value_counts().to_dict()}")

    # Prepare features and target
    X = feature_data.drop('target', axis=1)
    y = feature_data['target']

    # Train/test split
    train_mask = feature_data.index < '2022-01-01'
    test_mask = feature_data.index >= '2022-01-01'

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"🔢 Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 7. Model selection and hyperparameter tuning
    print("🤖 Training and optimizing models...")

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    # Hyperparameter grids
    param_grids = {
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    }

    best_models = {}
    cv_scores = {}

    for name, model in models.items():
        print(f"   Optimizing {name}...")

        # Use scaled data for logistic regression
        X_train_use = X_train_scaled if name == 'Logistic Regression' else X_train

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grids[name],
            cv=5, scoring='accuracy',
            n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train_use, y_train)
        best_models[name] = grid_search.best_estimator_
        cv_scores[name] = grid_search.best_score_

        print(f"     Best CV score: {cv_scores[name]:.3f}")

    # 8. Select best model and make predictions
    best_model_name = max(cv_scores, key=cv_scores.get)
    best_model = best_models[best_model_name]

    print(f"\n🏆 Best model: {best_model_name} (CV score: {cv_scores[best_model_name]:.3f})")

    # Make predictions
    if best_model_name == 'Logistic Regression':
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 9. Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    correct_predictions = sum(y_pred == y_test)
    total_predictions = len(y_test)

    print(f"\n✅ Prediction Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

    # Detailed evaluation
    print(f"\n📊 Detailed Performance:")
    print(classification_report(y_test, y_pred, target_names=['PEP Wins', 'KO Wins']))

    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i + 1:2d}. {row['feature']:<25}: {row['importance']:.4f}")

    # Recent predictions with confidence
    print(f"\n📅 Recent Predictions (with confidence):")
    recent_results = pd.DataFrame({
        'Date': X_test.index[-8:],
        'Actual': y_test.iloc[-8:].values,
        'Predicted': y_pred[-8:],
        'Confidence': y_pred_proba[-8:],
        'KO_Return': feature_data.loc[X_test.index[-8:], 'KO_return_lag_1'].shift(-1).fillna(0),
        'PEP_Return': feature_data.loc[X_test.index[-8:], 'PEP_return_lag_1'].shift(-1).fillna(0)
    })

    for _, row in recent_results.iterrows():
        status = "✅" if row['Actual'] == row['Predicted'] else "❌"
        actual_text = "KO won" if row['Actual'] == 1 else "PEP won"
        pred_text = "KO wins" if row['Predicted'] == 1 else "PEP wins"
        confidence = row['Confidence'] if row['Predicted'] == 1 else (1 - row['Confidence'])

        print(f"   {status} {row['Date'].strftime('%Y-%m')}: {actual_text} | "
              f"Predicted: {pred_text} (conf: {confidence:.1%})")

    print(f"\n🎯 Model Summary:")
    print(f"   • Best Algorithm: {best_model_name}")
    print(f"   • Features Used: {len(X.columns)}")
    print(f"   • Cross-validation Score: {cv_scores[best_model_name]:.3f}")
    print(f"   • Test Accuracy: {accuracy:.3f}")

    return best_model, X.columns.tolist(), accuracy


if __name__ == "__main__":
    model, features, accuracy = main()