import argparse
import os

import joblib
import pandas as pd
import numpy as np
from azureml import core
from sklearn import model_selection, linear_model, metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    '--regularization',
    type=float,
    dest='reg_rate',
    default=0.01,
    help='regularization rate',
)
parser.add_argument(
    '--data-folder', type=str, dest='data_folder', help='data folder reference'
)
args = parser.parse_args()
reg = args.reg_rate

run = core.Run.get_context()

data_folder = args.data_folder
print("Loading data from", data_folder)
all_files = os.listdir(data_folder)
diabetes = pd.concat(
    (pd.read_csv(os.path.join(data_folder, csv_file)) for csv_file in all_files)
)

X = diabetes[
        [
            'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
            'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree',
            'Age'
        ]
    ].to_numpy()
y = diabetes['Diabetic'].to_numpy()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.30, random_state=0
)

print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  reg)
model = (
    linear_model.LogisticRegression(C=1/reg, solver="liblinear", verbose=1)
    .fit(X_train, y_train)
)
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', acc)

y_scores = model.predict_proba(X_test)
auc = metrics.roc_auc_score(y_test, y_scores[:, 1])
print('AUC:', auc)
run.log('AUC', auc)

os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

run.complete()
