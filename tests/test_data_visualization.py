import pandas as pd
import numpy as np
from src.data_visualization import plot_assortment_effect, plot_correlation_heatmap, plot_sales_distribution, plot_promo_effect, plot_sales_behavior, plot_sales_vs_customers

# Create sample data for testing
def create_sample_data():
    np.random.seed(42)
    data = {
        "date": pd.date_range(start="2025-01-01", periods=100),
        "sales": np.random.randint(100, 1000, size=100),
        "customers": np.random.randint(10, 100, size=100),
        "promo": np.random.choice([0, 1], size=100),
        "assortment": np.random.choice(["basic", "extended", "premium"], size=100),
        "competitor_distance": np.random.randint(1, 50, size=100)
    }
    return pd.DataFrame(data)

# Test visualization functions
def test_visualization_functions():
    # Generate sample data
    df = create_sample_data()

    # Test distribution plot
    print("Testing plot_distribution...")
    plot_sales_distribution(df)

    # Test correlation heatmap
    print("Testing plot_correlation_heatmap...")
    plot_correlation_heatmap(data=df)

    # Test sales behavior plot
    print("Testing plot_sales_behavior...")
    holidays = ["2025-01-15", "2025-02-10", "2025-03-25"]
    plot_sales_behavior(data=df, date_column="date", sales_column="sales", holidays=holidays)

    # Test sales vs customers scatter plot
    print("Testing plot_sales_vs_customers...")
    plot_sales_vs_customers(data=df, sales_column="sales", customers_column="customers")

    # Test promo effect plot
    print("Testing plot_promo_effect...")
    plot_promo_effect(data=df, promo_column="promo", sales_column="sales")

    # Test assortment effect plot
    print("Testing plot_assortment_effect...")
    plot_assortment_effect(data=df, assortment_column="assortment", sales_column="sales")

if __name__ == "__main__":
    test_visualization_functions()
