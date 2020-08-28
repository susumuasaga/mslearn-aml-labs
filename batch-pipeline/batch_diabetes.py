import os

import joblib
import numpy as np
from azureml import core
from sklearn import linear_model

model: linear_model.LogisticRegression


def init():
    global model

    model_path = core.Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)


def run(mini_batch):
    result_list = []
    for f in mini_batch:
        data = np.genfromtxt(f, delimiter=',')
        prediction = model.predict(data.reshape(1, -1))
        result_list.append(f'{os.path.basename(f)}: {prediction[0]}')
    return result_list
