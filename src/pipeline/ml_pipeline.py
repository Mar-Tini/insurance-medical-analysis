import pandas as pd # type: ignore 
import joblib # type: ignore 

from src.features.engineering import feature_engineering
from src.models.train import train_model
from src.models.evaluate import evaluate_model, test_homoscedasticity
from src.models.visualize import plot_predictions, plot_residuals, compare_variance_patterns
from src.data.config import load_config
import statsmodels.api as sm # type: ignore 
from statsmodels.stats.diagnostic import het_breuschpagan # type: ignore 

def run_pipeline():
    data_config, model_config = load_config()

    # -- Load data --
    train_df = pd.read_csv(data_config["data"]["processed"]["train"])
    test_df = pd.read_csv(data_config["data"]["processed"]["test"])

    target = model_config["model"]["target"]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # -- Feature engineering --
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    # -- Load preprocessor --
    preprocessor = joblib.load(
        model_config["artefacte"]["preprocessing"]
    )

    # -- Preprocessing --
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # -- Training --
    model = train_model(X_train_processed, y_train)

    # -- Evaluation --
    metrics = evaluate_model(model, X_test_processed, y_test)

    # -- Visualization --
    plot_predictions(y_test, metrics["y_pred"])
    plot_residuals(y_test, metrics["y_pred"])
    # plot_residuals_by_group(y_test.reset_index(drop=True), metrics["y_pred"], X_test["smoker"].reset_index(drop=True),)

    # -- Simulate heteroscedasticity for learning --
    compare_variance_patterns(y_test, metrics["y_pred"])

    # -- Residuals --
    residuals = y_test.to_numpy() - metrics["y_pred"]

    # -- Breusch-Pagan test --
    test_homoscedasticity(X_test_processed, residuals)
            

if __name__ == "__main__":
    run_pipeline()
