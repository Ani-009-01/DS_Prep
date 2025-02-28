import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_load import getdata

def plot_visualizations(df):
    """Generate visualizations for startup data analysis."""
    
    # Set style for seaborn
    sns.set_style("whitegrid")
    
    # Histogram 
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Valuation (USD)"], bins=30, kde=True, color="blue")
    plt.title("Distribution of Startup Valuation")
    plt.xlabel("Valuation (USD)")
    plt.ylabel("Frequency")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=["number"])  # Select only numeric columns
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Scatter Plot: Funding Rounds vs. Valuation
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["Funding Rounds"], y=df["Valuation (USD)"], alpha=0.7, color="green")
    plt.title("Funding Rounds vs. Valuation")
    plt.xlabel("Funding Rounds")
    plt.ylabel("Valuation (USD)")
    plt.show()

    # Scatter Plot: Investment Amount vs. Valuation
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["Investment Amount (USD)"], y=df["Valuation (USD)"], alpha=0.7, color="red")
    plt.title("Investment Amount vs. Valuation")
    plt.xlabel("Investment Amount (USD)")
    plt.ylabel("Valuation (USD)")
    plt.show()


    # Box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Industry", y="Growth Rate (%)", data=df)
    plt.xticks(rotation=45)
    plt.title("Growth Rate Distribution by Industry")
    plt.xlabel("Industry")
    plt.ylabel("Growth Rate (%)")
    plt.show()
    
   

# plot_visualizations(getdata())