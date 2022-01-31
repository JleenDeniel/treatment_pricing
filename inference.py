import pandas as pd
from catboost import CatBoostRegressor
import numpy as np

def load_model():
    model = CatBoostRegressor()
    model.load_model('../models/catboostregressor_v1.cbm')
    return model


def make_prediction(model: CatBoostRegressor, case: np.array()):
    predictions = model.predict(case)
    print(predictions)


if __name__ == "__main__":
    test_cases = pd.read_csv('../data/test.csv')
    make_prediction(load_model(), np.array(test_cases))


