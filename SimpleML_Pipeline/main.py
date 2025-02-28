from data_load import getdata
from EDA import perform_eda
from data_visualization import plot_visualizations
from data_cleaning import clean_data
from model import train_and_compare_models

# Load dataset
df = getdata()

# plot visualizations
plot_visualizations(df)

# Perform EDA
perform_eda(df)

# Clean & preprocess data
preprocessor, X, y = clean_data(df)

# Train & Compare Models
best_model_name, best_model = train_and_compare_models(preprocessor, X, y)