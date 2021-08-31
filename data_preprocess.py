import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

data_path = 'CarPricePredict/data/CarPrice_Assignment.csv'
abs_data_path = os.path.normpath(os.path.join(os.getcwd(), data_path))
data_car = pd.read_csv(abs_data_path)
print(data_car.head())
print(data_car.columns)

index_cols = ['car_ID', 'symboling', 'CarName']
label_col = ['price']
data_car = data_car.set_index(index_cols)

df_label = data_car[label_col]
df_data = data_car.drop(label_col, axis=1)
print(df_data.dtypes)
print(df_data.info())

### Features
df_data['fueltype'].value_counts() # -- fuel and gas

df_data['aspiration'].value_counts() # -- std and turbo

df_data['doornumber'].value_counts() # -- four and two

df_data['carbody'].value_counts() # -- sedan, hatchback, wagon, hardtop, convertible

df_data['drivewheel'].value_counts() # -- fwd, rwd, 4wd

df_data['enginelocation'].value_counts() # -- front, rear

df_data['enginetype'].value_counts() # -- ohc, ohcf, ohcv, dohc, l, rotor

df_data['cylindernumber'].value_counts() # -- four, six, five, eight, two, twelve

df_data['fuelsystem'].value_counts() # -- mpfi, 2bbl, idi, 1bbl

cat_features = ['fueltype', 'aspiration', 'doornumber',
                'carbody', 'drivewheel', 'enginelocation',
                'enginetype', 'cylindernumber', 'fuelsystem']

for col in cat_features:
    df_data[col] = pd.Series(df_data[col], dtype="category")

cont_features = ['carlength', 'carwidth', 'carheight',
         'curbweight', 'enginesize', 'boreratio',
         'stroke', 'compressionratio', 'horsepower',
         'peakrpm', 'citympg', 'highwaympg']
df_data[cont_features].describe()

## plot features
fig, axs = plt.subplots(3,4, figsize=(15, 12), sharex=False, sharey=False)
for i, ax in enumerate(axs.flat):
    ax.hist(df_data[cont_features[i]])
    ax.set_title(f'{cont_features[i]}')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

## plot label price
df_label.hist()
plt.show()

################# Save File #################
import pickle

preprocessed_data_path = 'CarPricePredict/data/preprocessed_data.pickle'
abs_prep_data_path = os.path.normpath(os.path.join(os.getcwd(), preprocessed_data_path))
with open(abs_prep_data_path, 'wb') as f:
    pickle.dump(df_data, f)

label_data_path = 'CarPricePredict/data/label_data.pickle'
abs_lbl_data_path = os.path.normpath(os.path.join(os.getcwd(), label_data_path))
with open(abs_lbl_data_path, 'wb') as f:
    pickle.dump(df_label, f)