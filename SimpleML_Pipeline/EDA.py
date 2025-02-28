import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df):  
    # umerical Features
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols].hist(figsize=(12, 8), bins=30, edgecolor='black')
    plt.suptitle("Distribution of Numerical Features", fontsize=14)
    plt.show()

    # Pairplot to observe feature relationships
    # sns.pairplot(df[numeric_cols].sample(500), diag_kind="kde")
    # plt.show()
    
    

