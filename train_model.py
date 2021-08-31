import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

preprocessed_data_path = 'CarPricePredict/data/preprocessed_data.pickle'
abs_prep_data_path = os.path.normpath(os.path.join(os.getcwd(), preprocessed_data_path))
with open(abs_prep_data_path, 'rb') as f:
    df_data = pickle.load(f, encoding='bytes')
print(df_data.shape)

label_data_path = 'CarPricePredict/data/label_data.pickle'
abs_lbl_data_path = os.path.normpath(os.path.join(os.getcwd(), label_data_path))
with open(abs_lbl_data_path, 'rb') as f:
    df_label = pickle.load(f, encoding='bytes')
print(df_label.shape)

#########
## setup lgb
cat_features = ['fueltype', 'aspiration', 'doornumber',
                'carbody', 'drivewheel', 'enginelocation',
                'enginetype', 'cylindernumber', 'fuelsystem']

X_train, X_test, y_train, y_test = train_test_split(df_data, df_label, test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'regression',
    'boosting': 'gbdt',
    'num_iterations': 100,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'min_data_in_leaf': 10,
    'colsample_bytree': 0.8
}
res = lgb.cv(params, train_data,
       num_boost_round=100,
       nfold=5,
       categorical_feature='auto',
       early_stopping_rounds=50,
       seed=0,
       eval_train_metric=False,
       stratified=False,
       metrics='rmse')

## plot cv train rmse error
plt.plot(res['rmse-mean'])
plt.show()

model = lgb.train(params, train_set=train_data)
test_predict = model.predict(test_data)



