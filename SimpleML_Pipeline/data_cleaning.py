import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from data_load import getdata


def clean_data(df):
    
    # Handle Missing Values
    df = df.dropna()  

    # Feature Engineering
    current_year = datetime.now().year
    df["Startup Age"] = current_year - df["Year Founded"]
    df["Funding Efficiency Ratio"] = df["Investment Amount (USD)"] / df["Number of Investors"]
    df["Funding Rounds Per Year"] = df["Funding Rounds"] / (df["Startup Age"] + 1)
    df["Investment Per Round"] = df["Investment Amount (USD)"] / df["Funding Rounds"]
    df["Investor Density"] = df["Number of Investors"] / df["Funding Rounds"]

    # Industry & Country Normalization
    df["Growth Rate Relative to Industry"] = df.groupby("Industry")["Growth Rate (%)"].transform(lambda x: x / x.median())
    df["Investment Relative to Country"] = df.groupby("Country")["Investment Amount (USD)"].transform(lambda x: x / x.median())

    # Define Features for Preprocessing
    numerical_features = [
        "Funding Rounds", "Investment Amount (USD)", "Number of Investors", "Startup Age", 
        "Funding Efficiency Ratio", "Funding Rounds Per Year", "Investment Per Round", 
        "Investor Density", "Growth Rate (%)", "Growth Rate Relative to Industry", "Investment Relative to Country"
    ]
    categorical_features = ["Industry", "Country"]

    # Log Transformation to Reduce Skewness
    log_features = ["Investment Amount (USD)", "Valuation (USD)", "Funding Efficiency Ratio", "Investment Per Round", "Investor Density", "Investment Relative to Country"]
    for feature in log_features:
        df[feature] = np.log1p(df[feature])  # log(1 + x) to avoid log(0) issues

    # Standardization & Encoding
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])
    
    X = df.drop(columns=["Startup Name", "Valuation (USD)"])  # Features
    y = df["Valuation (USD)"]  # Target

    return preprocessor, X, y
# print(clean_data(getdata()))
