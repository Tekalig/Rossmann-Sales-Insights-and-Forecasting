import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def clean_data(df):
    """Perform data cleaning tasks such as handling missing values and outliers."""
    # Example: Drop rows with missing target values
    df_cleaned = df.dropna(subset=['Sales'])
    logging.info("Missing values handled.")
    # Additional cleaning steps...
    return df_cleaned

def handle_missing_data(df):
    """Handles missing data using mean imputation."""
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def detect_outliers(df, column, threshold=3):
    """Detects outliers using Z-score."""
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[(z_scores > -threshold) & (z_scores < threshold)]

def save_cleaned_data(df, file_path):
    """Saves cleaned data to a file."""
    df.to_csv(file_path, index=False)

def preprocess_data(data):
    # Handle missing values
    data.fillna(0, inplace=True)

    # Extract features from datetime columns
    data['weekday'] = data['date'].dt.weekday
    data['weekend'] = data['date'].dt.weekday.isin([5, 6]).astype(int)
    data['days_to_holiday'] = (data['next_holiday'] - data['date']).dt.days
    data['days_after_holiday'] = (data['date'] - data['last_holiday']).dt.days
    data['month_period'] = data['date'].dt.day.apply(lambda x: 'start' if x <= 10 else 'mid' if x <= 20 else 'end')

    # Generate more features (customizable)
    data['is_month_start'] = data['date'].dt.is_month_start.astype(int)
    data['is_month_end'] = data['date'].dt.is_month_end.astype(int)

    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=['month_period'], drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data
