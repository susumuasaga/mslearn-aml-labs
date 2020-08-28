import json
import joblib
import numpy as np
from azureml.core.model import Model
from sklearn.linear_model import LogisticRegression

model: LogisticRegression


def init():
    """
    Initialize `model`
    """
    global model
    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)


def run(raw_data):
    """
    Score `model`
    """
    data = json.loads(raw_data)['data']
    np_data = np.array(data)
    predictions = model.predict(np_data)
    log_text = f'Data: {data} - Predictions: {predictions}'
    print(log_text)
    classnames = ['not-diabetic', 'diabetic']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    return json.dumps(predicted_classes)
