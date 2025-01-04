import unittest
from src.data_processing import clean_data
import pandas as pd

class TestDataPreprocessing(unittest.TestCase):
    def test_clean_data(self):
        # Create a sample dataframe with missing values
        df = pd.DataFrame({'Sales': [100, None, 200]})
        df_clean = clean_data(df)
        # Assert that rows with missing 'Sales' are dropped
        self.assertEqual(len(df_clean), 2)

if __name__ == '__main__':
    unittest.main()
