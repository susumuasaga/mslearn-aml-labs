import argparse
import os

import joblib
import pandas as pd
import numpy as np
from azureml import core
from sklearn import model_selection, tree, metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_folder', type=str, dest='output_folder',
    default='diabetes_model', help='output folder'
)
args = parser.parse_args()
output_folder = args.output_folder

run = core.Run.get_context()

print("Loading Data...")
diabetes: pd.DataFrame = (
    run.input_datasets['diabetes_train'].to_pandas_dataframe()
)
X = diabetes[
        [
            'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
            'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree',
            'Age',
        ]
    ].to_numpy()
y = diabetes['Diabetic'].to_numpy()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.30, random_state=0
)

print('Training a decision tree model')
model = tree.DecisionTreeClassifier().fit(X_train, y_train)

y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', acc)

y_scores = model.predict_proba(X_test)
auc = metrics.roc_auc_score(y_test, y_scores[:, 1])
print('AUC:', auc)
run.log('AUC', auc)

os.makedirs(output_folder, exist_ok=True)
output_path = output_folder + "/model.pkl"
joblib.dump(value=model, filename=output_path)

run.complete()
