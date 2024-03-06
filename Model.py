import pickle
import pandas as pd

import csv as csv

from pycaret.classification import load_model, predict_model
loaded_model = load_model('excel_neuralnet_modell')
test = pd.read_csv("test.csv")


predictions = predict_model(loaded_model, data=test)
predictions=predictions.drop(columns=['date','state','store','product'])


if not predictions.empty:
    predictions.to_csv('sample.csv',sep=',',encoding='utf-8', index=False)
    
    print("Predictions saved to sample.csv.")
else:
    print("No predictions were generated.")



