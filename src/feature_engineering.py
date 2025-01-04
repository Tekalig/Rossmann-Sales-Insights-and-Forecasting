import pandas as pd
import logging

def extract_date_features(df):
    """Extract date-related features from the 'Date' column."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    logging.info("Date features extracted.")
    return df
