import os
import mlflow
from mlflow import log_metric, log_param 
from mlflow_utils import create_mlflow_experiment, get_mlflow_experiment
from dotenv import load_dotenv

# Set tracking URI to your Heroku application
load_dotenv()

MLFLOW_URI = os.environ["MLFLOW_URI"]

EXPERIMENT_NAME="my-experiment"
mlflow.set_tracking_uri(MLFLOW_URI)


if __name__ == "__main__":

    run_descrition = "this is a test."
    experiment_id = create_mlflow_experiment(EXPERIMENT_NAME)
    experiment = get_mlflow_experiment(experiment_id)

    with mlflow.start_run(
        run_name = "parent",
        experiment_id = experiment.experiment_id,
        description= run_descrition
        ) as run:

        with mlflow.start_run(run_name= "child", nested=True):
            log_param("un",1)
            log_metric("metric1", .9)
            log_metric("metric2", .9)
