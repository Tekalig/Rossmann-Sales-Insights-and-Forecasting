import matplotlib.pyplot as plt
import seaborn as sns
import logging

def plot_sales_distribution(df):
    """Plot the distribution of sales."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sales'], bins=50, kde=True)
    plt.title('Sales Distribution')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    logging.info("Sales distribution plot generated.")

def check_outlier_plot(df):
    """Check for outliers in the 'Sales' column."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Sales', data=df)
    plt.title('Sales Outliers')
    plt.grid(True)
    plt.show()
    logging.info("Sales outlier plot generated.")


def plot_correlation_heatmap(data, title="Correlation Heatmap"):
    """
    Plot a heatmap of correlations between numerical columns.
    Args:
        data (pd.DataFrame): The dataset.
        title (str): The title of the heatmap.
    """
    logging.info("Plotting correlation heatmap")
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()


def plot_sales_behavior(data, date_column, sales_column, holidays=None):
    """
    Plot sales behavior before, during, and after holidays.
    Args:
        data (pd.DataFrame): The dataset.
        date_column (str): The date column.
        sales_column (str): The sales column.
        holidays (list): List of holiday dates (optional).
    """
    logging.info("Plotting sales behavior around holidays")
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data, x=date_column, y=sales_column, label="Sales")

    if holidays:
        for holiday in holidays:
            plt.axvline(x=pd.to_datetime(holiday), color='red', linestyle='--', label=f"Holiday: {holiday}")

    plt.title("Sales Behavior Before, During, and After Holidays")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()


def plot_sales_vs_customers(data, sales_column, customers_column):
    """
    Plot the relationship between sales and number of customers.
    Args:
        data (pd.DataFrame): The dataset.
        sales_column (str): The sales column.
        customers_column (str): The customers column.
    """
    logging.info(f"Plotting sales vs customers: {sales_column} and {customers_column}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=customers_column, y=sales_column, color='purple')
    plt.title("Sales vs Customers")
    plt.xlabel("Number of Customers")
    plt.ylabel("Sales")
    plt.show()


def plot_promo_effect(data, promo_column, sales_column):
    """
    Plot the effect of promotions on sales.
    Args:
        data (pd.DataFrame): The dataset.
        promo_column (str): The promo column.
        sales_column (str): The sales column.
    """
    logging.info(f"Plotting promo effect: {promo_column} on {sales_column}")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=promo_column, y=sales_column, palette="Set2")
    plt.title("Effect of Promotions on Sales")
    plt.xlabel("Promo")
    plt.ylabel("Sales")
    plt.show()


def plot_assortment_effect(data, assortment_column, sales_column):
    """
    Plot the effect of assortment type on sales.
    Args:
        data (pd.DataFrame): The dataset.
        assortment_column (str): The assortment type column.
        sales_column (str): The sales column.
    """
    logging.info(f"Plotting assortment effect: {assortment_column} on {sales_column}")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=assortment_column, y=sales_column, palette="Set3")
    plt.title("Effect of Assortment Type on Sales")
    plt.xlabel("Assortment Type")
    plt.ylabel("Sales")
    plt.show()