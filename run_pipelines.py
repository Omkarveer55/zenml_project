from pipelines import training_pipeline
from zenml.client import Client
import mlflow


if __name__ == '__main__':
    tracking_uri = Client().active_stack.experiment_tracker.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    training_pipeline.train_pipeline(r'C:\Users\DELL3\zenml_pipelines\data\iris.csv')
    