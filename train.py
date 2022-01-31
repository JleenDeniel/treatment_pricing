from catboost import CatBoostRegressor, Pool
import pandas as pd
from sklearn.model_selection import train_test_split


def load_train_df():
    try:
        april_1_15 = pd.read_excel('../data/april_1-15.xlsx')
        april_16_30 = pd.read_excel('../data/april_16-30.xlsx')
        may_1_15 = pd.read_excel('../data/may_1-15.xlsx')
        may_16_31 = pd.read_excel('../data/may_16-31.xlsx')
        june_1_15 = pd.read_excel('../data/june_1-15.xlsx')
        june_16_30 = pd.read_excel('../data/june_16-30.xlsx')
        file = pd.concat([april_1_15, april_16_30, may_1_15, may_16_31, june_1_15, june_16_30], ignore_index=True)
    except FileNotFoundError:
        print('No such file!')
        return None
    return file


def split_on_train_val_test(file: pd.DataFrame):
    file.fillna(0, inplace=True)
    X = file[:200000].drop(['money', 'patient_ID'], axis=1)
    y = file[:200000]['money']
    test = file[200000:]
    X_test = test.drop(['money', 'patient_ID'], axis=1)
    y_test = test['money']
    X_train, X_val, y_train, y_val = train_test_split(X.astype('int'), y, random_state=1234)
    trainPool = Pool(X_train, y_train)
    evalPool = Pool(X_val, y_val)
    testPool = Pool(X_test, y_test)
    return trainPool, evalPool, testPool


def fit(trainPool, evalPool):
    reg = CatBoostRegressor(learning_rate=0.1, iterations=1000, depth=8, use_best_model=True)
    reg.fit(trainPool, eval_set=evalPool)
    print('Model Fitted')
    reg.save_model('../models/catboostregressor.cbm')


def get_test_score(testpool):
    model = CatBoostRegressor()
    model.load_model('../models/catboostregressor.cbm')
    print(model.score(testpool))


if __name__ == "__main__":
    df = load_train_df()
    trainPool, evalPool, testPool = split_on_train_val_test(df)
    fit(trainPool, evalPool)
    print('test score is' + get_test_score(testPool))
