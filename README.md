# Rossmann Sales Insights and Forecasting

This project analyzes historical sales data from Rossmann Pharmaceuticals to generate insights and forecast future sales trends. The project includes data cleaning, exploratory data analysis (EDA), and data visualization to understand key metrics and relationships.

---

## Project Structure

```
├── data/
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data files
│
├── notebooks/
│   ├── exploratory_analysis.ipynb  # EDA and insights
│
├── scripts/                   # Utility scripts
│   ├── exploratory_analysis.py       # EDA script
│
├── src/
│   ├── tasks/
│   │   ├── feature_engineeing.py        # Functions for feature engineering
│   │   ├── data_visualization.py   # Functions for visualizing data
│   │   ├── data_processing.py      # Functions for data transformation
│   │   ├── logger.py     # Logging configuration
│   
├── logs/
│   ├── eda.log               # Log file for EDA
│   
│
├── tests/
│   ├── test_data_cleaning.py       # Test script for data cleaning
│   ├── test_data_visualization.py  # Test script for data visualization
│
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── .gitignore                      # Files to ignore in version control
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Features

1. **Data Cleaning**
   - Handles missing values and outliers.
   - Ensures data consistency and quality.

2. **Data Visualization**
   - Generates plots to analyze sales trends, promotional impacts, and customer behavior.
   - Correlation heatmaps, scatter plots, and time series visualizations.

3. **Logging**
   - Logs every step for traceability using the `logger` library.

4. **Modular Design**
   - Code is organized into reusable modules for better maintainability.

---

## Usage

1. **Exploratory Data Analysis (EDA)**
   - Run the notebook `notebooks/exploratory_analysis.ipynb` to perform EDA and generate insights.

2. **Data Cleaning**
   - Use `src/tasks/data_cleaning.py` to clean raw data:
     ```python
     from src.tasks.data_cleaning import clean_data
     clean_data("data/raw/data.csv", "data/processed/cleaned_data.csv")
     ```

3. **Data Visualization**
   - Visualize key metrics with `src/tasks/data_visualization.py`:
     ```python
     from src.tasks.data_visualization import plot_distribution
     plot_distribution(data, column="sales", title="Sales Distribution")
     ```

4. **Testing**
   - Run test scripts to validate functionality:
     ```bash
     pytest tests/
     ```

---

## Key Insights to Explore

- **Sales Trends**:
  - How do holidays, promotions, and seasonal events affect sales?
- **Customer Behavior**:
  - What is the relationship between sales and the number of customers?
- **Promotions**:
  - How effective are promotions in driving sales and customer engagement?
- **Competitor Impact**:
  - How does competitor distance affect sales, especially in urban areas?

---
## Future Work

- Build machine learning models to predict sales trends.
- Integrate a dashboard for real-time visualization and insights.
- Deploy the project using containerization and CI/CD pipelines.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributors

- **Tekalign Mesfin**  
  4th Year Computer Engineering Student  
  Marketing Analytics Engineer at AlphaCare Insurance Solutions  
    Addis Ababa, Ethiopia