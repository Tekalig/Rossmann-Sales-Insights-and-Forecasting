import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
