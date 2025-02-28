import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Import regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def train_and_compare_models(preprocessor, X, y):
    

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model building
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=5),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}

    # train models
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipeline.fit(X_train, y_train)  # Train

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Compute Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"RMSE": rmse, "R² Score": r2}

    # select best 
    best_model = min(results, key=lambda k: results[k]["RMSE"])
    print(f"\nBest Model: {best_model} with RMSE = {results[best_model]['RMSE']:.4f} and R² = {results[best_model]['R² Score']:.4f}")

    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results).T
    print("\n Model Performance Summary:")
    print(results_df.sort_values(by="RMSE"))

    return best_model, models[best_model]
