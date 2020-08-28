import os
import pandas as pd
import numpy as np
import joblib
from azureml.contrib.interpret.explanation.explanation_client import (
    ExplanationClient,
)
from interpret_community import TabularExplainer
from sklearn import model_selection, metrics

from azureml import core

from sklearn.tree import DecisionTreeClassifier

run = core.run.Run.get_context()

print("Loading Data...")
data = pd.read_csv('diabetes.csv')

features = [
    'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
    'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age',
]
labels = ['not-diabetic', 'diabetic']

X = data[features].to_numpy()
y = data['Diabetic'].to_numpy()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.30, random_state=0
)

print('Training a decision tree model')
model: DecisionTreeClassifier = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', acc)

y_scores = model.predict_proba(X_test)
auc = metrics.roc_auc_score(y_test, y_scores[:, 1])
run.log('AUC', auc)

os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetes.pkl')

explainer = TabularExplainer(model, X_train, features=features, classes=labels)
explanation = explainer.explain_global(X_test)

explain_client = ExplanationClient.from_run(run)
explain_client.upload_model_explanation(
    explanation, comment='Tabular Explanation'
)

run.complete()
