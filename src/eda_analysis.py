import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def analyze_distributions(df, column, title, save_path):
    """Plots the distribution of a column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def compare_sales_during_holidays(df, date_column, sales_column, holidays):
    """Compares sales during holidays."""
    df['is_holiday'] = df[date_column].isin(holidays)
    holiday_sales = df.groupby('is_holiday')[sales_column].mean()
    return holiday_sales

def plot_time_series(df, date_column, value_column, save_path):
    """Plots time series data."""
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=date_column, y=value_column, data=df)
    plt.savefig(save_path)
    plt.close()

def train_model(data, target_column):
    # Split the data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipeline
    pipeline = Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    return pipeline

def custom_loss_function(y_true, y_pred):
    # Mean Absolute Percentage Error (MAPE)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def analyze_predictions(model, data, target_column):
    # Feature Importance
    feature_importances = model.named_steps['model'].feature_importances_
    features = data.drop(columns=[target_column]).columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    print(importance_df.sort_values(by='Importance', ascending=False))

    # Confidence Interval Estimation (example: 95% CI)
    predictions = model.predict(data.drop(columns=[target_column]))
    std_dev = np.std(predictions)
    mean_prediction = np.mean(predictions)
    ci_lower = mean_prediction - 1.96 * std_dev
    ci_upper = mean_prediction + 1.96 * std_dev
    print(f"95% Confidence Interval: [{ci_lower}, {ci_upper}]")


