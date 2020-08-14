import argparse

import joblib
from azureml import core

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_folder', type=str, dest='model_folder', default="diabetes_model",
    help='model location'
)
args = parser.parse_args()
model_folder = args.model_folder

run = core.Run.get_context()

print("Loading model from", model_folder)
model_file = model_folder + "/model.pkl"
model = joblib.load(model_file)

core.Model.register(
    workspace=run.experiment.workspace,
    model_path=model_file,
    model_name='diabetes_model',
    tags={'Training context': 'Pipeline'},
)

run.complete()
