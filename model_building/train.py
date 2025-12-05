
import pandas as pd
import os
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb

# Hugging Face Hub
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow CONFIGURATION

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

api = HfApi()

# -------------------------------------------------------
# LOAD TRAIN + TEST DATA FROM HUGGINGFACE DATASET HUB
# -------------------------------------------------------
Xtrain_path = "hf://datasets/Amitgupta2982/Tourism-Package-Prediction/Xtrain.csv"
Xtest_path  = "hf://datasets/Amitgupta2982/Tourism-Package-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/Amitgupta2982/Tourism-Package-Prediction/ytrain.csv"
ytest_path  = "hf://datasets/Amitgupta2982/Tourism-Package-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest  = pd.read_csv(ytest_path).values.ravel()

print("Dataset successfully loaded from HuggingFace 🚀")

# IDENTIFY NUMERIC + CATEGORICAL FEATURES

categorical_features = Xtrain.select_dtypes(include="object").columns.tolist()
numeric_features     = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()

# PREPROCESSING PIPELINE (SCALING + ONE-HOT ENCODING)

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)


# DEFINE BASE MODEL — XGBoost

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


# HYPERPARAMETER GRID

param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__subsample": [0.7, 1.0],
}

# MODEL PIPELINE

model_pipeline = make_pipeline(preprocessor, xgb_model)

# START MLflow TRACKING

with mlflow.start_run():
    print("Starting GridSearch hyperparameter tuning...")

    grid = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="accuracy"
    )

    # Train models
    grid.fit(Xtrain, ytrain)

    # Log all trials
    results = grid.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])

    # Log best params
    mlflow.log_params(grid.best_params_)

 
    # MODEL EVALUATION

    best_model = grid.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest,  y_pred_test,  output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "test_accuracy":  test_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "test_precision":  test_report["1"]["precision"],
        "train_recall":    train_report["1"]["recall"],
        "test_recall":     test_report["1"]["recall"],
        "train_f1_score":  train_report["1"]["f1-score"],
        "test_f1_score":   test_report["1"]["f1-score"],
    })

    # SAVE BEST MODEL LOCALLY
 
    model_path = "tourism_xgb_best_model_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"Model saved locally: {model_path}")

    mlflow.log_artifact(model_path, artifact_path="model")

    # PUSH MODEL TO HUGGINGFACE MODEL HUB

    repo_id = "Amitgupta2982/Tourism-Package-Model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print("Model repo already exists — updating it.")
    except RepositoryNotFoundError:
        print("Model repo not found — creating a new one...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="tourism_xgb_best_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

    print("Model successfully uploaded to HuggingFace Model Hub 🚀")

print("Production training completed ✔")
