import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

class Evaluate_Model: 
    def __init__(self)->None:
        pass
        
    def evaluation_model(self,X_train, y_train , preprocessor):
        

        xgb_pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", XGBRegressor(random_state=42))
        ])

        param_grid = {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [3, 5, 7, 10],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0]
        }


        grid = GridSearchCV(
            estimator=xgb_pipe,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )


        grid.fit(X_train, y_train)

        print("Meilleurs paramètres :", grid.best_params_)
        print("Meilleur R² en validation croisée :", grid.best_score_)

        return grid.best_estimator_