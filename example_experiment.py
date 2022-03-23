import logging
import os
import random
import warnings
from typing import List, Tuple

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow

# Here, we specify the local URL of our MLflow server, which is running on Docker
mlflow.set_tracking_uri("http://localhost:5000")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

# Simple logging facility
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual: List[float], pred: List[float]) -> Tuple[float, float, float]:
    """ Compute different scoring metrics.
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # A toy dataset about wine quality
    csv_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
    data = pd.read_csv(csv_url, sep=";")

    # Set the MLflow experiment (ie. project) name (to be created if not exist)
    expr_name = "project_4"
    s3_uri = "s3://mlflow"
    if mlflow.get_experiment_by_name(expr_name) is None:
        print("Creating experiment: {}".format(expr_name))
        mlflow.create_experiment(expr_name, artifact_location=s3_uri)
    mlflow.set_experiment(expr_name)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=0.2, train_size=0.8)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # A run is a given task in a whole experiment (model_1 for the project_1 eg.)
    with mlflow.start_run(run_name="model_1_optimisation") as model_1:

        # Multiples runs of the same experiment
        for _ in range(10):

            # Each try for this model_1 will be a sub-run, thus nested=True
            with mlflow.start_run(run_name=f"optimisation_{_}", nested=True) as model_1_try:

                # We gonna generate random param values (should be HyperOpt, gridsearch, BBO, etc)
                alpha = random.uniform(0.1, 0.9)
                l1_ratio = random.uniform(0.1, 0.9)

                # We train the model with those random parametres
                lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                lr.fit(train_x, train_y)
                # And evaluate it
                predicted_qualities = lr.predict(test_x)
                (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                # Different MLflow logging facilities for params and results
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Model registry does not work with file store
                mlflow.sklearn.log_model(lr, "model", registered_model_name="MyFirstModel")
