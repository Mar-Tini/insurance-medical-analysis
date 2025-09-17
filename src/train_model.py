import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error ,r2_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor


class Train_Model: 
    def __init__(self)->None:
        pass

    def models(self): 
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "SVR": SVR(kernel="rbf"),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
            ),
            "XGBoost": XGBRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
            ),
        }
        
        return models
    
    
    def select_model(self, models, X_train, y_train,X_test, y_test):
        preprocessor = joblib.load("../preprocessor/preprocessor.pkl")

        results = {}

        for name, model in models.items():
            pipe = Pipeline([
                ("preprocess", preprocessor),
                ("regressor", model)
            ])
            
            pipe.fit(X_train, y_train.values.ravel())
            y_pred = pipe.predict(X_test)  
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {"MSE": mse, "R2": r2}

        results_df = pd.DataFrame(results).T
        print(results_df)