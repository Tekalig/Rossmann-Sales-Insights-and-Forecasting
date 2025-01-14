from src.data_processing import load_data, handle_missing_data, detect_outliers, save_cleaned_data, preprocess_data
from src.eda_analysis import analyze_distributions, compare_sales_during_holidays, plot_time_series, custom_loss_function, train_model, analyze_predictions
from src.data_visualization import plot_sales_distribution, check_outlier_plot, plot_sales_behavior, plot_sales_vs_customers, plot_correlation_heatmap, plot_promo_effect, plot_assortment_effect
from src.logger import setup_logging
from src.feature_engineering import train_lstm, extract_date_features, serialize_model

def main():
    logger = setup_logging()
    logger.info("Task 1: Exploration of Customer Purchasing Behavior started.")

    # Load data
    raw_data_path = 'data/dataset.csv'
    df = load_data(raw_data_path)
    logger.info(f"Data loaded from {raw_data_path}. Shape: {df.shape}")

    # Data Cleaning
    df = handle_missing_data(df)
    logger.info("Missing data handled.")

    # Visualization for outliers
    check_outlier_plot(df)
    logger.info("Outlier plot generated.")

    df = detect_outliers(df, 'sales')
    logger.info("Outliers detected and removed.")

    cleaned_data_path = 'data/processed/cleaned_data.csv'
    save_cleaned_data(df, cleaned_data_path)
    logger.info(f"Cleaned data saved to {cleaned_data_path}. Shape: {df.shape}")

    # EDA
    analyze_distributions(df, 'sales', 'Sales Distribution', 'plots/sales_distribution.png')
    logger.info("Sales distribution plot saved.")

    holidays = ['2023-12-25', '2024-01-01']
    holiday_sales = compare_sales_during_holidays(df, 'date', 'sales', holidays)
    logger.info(f"Holiday sales comparison: {holiday_sales}")

    plot_time_series(df, 'date', 'sales', 'plots/sales_time_series.png')
    logger.info("Time series plot saved.")

    plot_sales_distribution(df)
    logger.info("Plot the distribution of sales")

    plot_sales_behavior(df, 'date', 'sales', holidays)
    logger.info("Plot sales behavior around holidays")

    plot_sales_vs_customers(df, 'sales', 'customers')
    logger.info("Plot the relationship between sales and number of customers")

    plot_correlation_heatmap(df)
    logger.info("Plot correlation heatmap")

    plot_promo_effect(df, 'promo', 'sales')
    logger.info("Plot the effect of promotions on sales")

    plot_assortment_effect(df, 'assortment', 'sales')
    logger.info("Plot the effect of assortment type on sales")

    logger.info("Task 1 completed successfully.")

    logger.info("Task 2: Training a model to predict sales started.")
    # Feature Engineering
    df_features = extract_date_features(df)
    logger.info("Date features extracted.")

    # Preprocessing
    df_processed = preprocess_data(df_features)
    logger.info("Data preprocessed.")

    # Train Model
    model = train_model(df_processed, 'sales')
    logger.info("Model trained.")

    # Custom Loss Function
    loss = custom_loss_function(df_processed['sales'], model.predict(df_processed.drop(columns=['sales'])))
    logger.info(f"Custom loss: {loss}")

    # Analyze Predictions
    analyze_predictions(model, df_processed, 'sales')
    logger.info("Predictions analyzed.")



if __name__ == "__main__":
    main()
