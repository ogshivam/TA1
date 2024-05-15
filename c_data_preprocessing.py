# data_processing.py
import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv("/Users/shivampratapwar/Desktop/Customer_Churn_Prediction.csv")

def preprocess_data(df):
    """Perform data preprocessing steps."""
    print(df.shape)
    df.info()
    df.describe()
    df.isnull().sum()
    # Handle missing values, remove outliers, etc.
    return df
