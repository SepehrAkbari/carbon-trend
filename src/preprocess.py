import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/annual_global_CO2_levels', 
                 delimiter = "\t", 
                 names = ["Year", "CO2 Level"], 
                 skiprows = 1)

def preprocess(df: pd.DataFrame):
    x_values = df['Year'].values.astype(np.float32)
    x_values = (x_values - x_values.min()) / (x_values.max() - x_values.min())

    y_values = df['CO2 Level'].values.astype(np.float32)
    y_values = (y_values - y_values.min()) / (y_values.max() - y_values.min())

    x_train_val, x_test, y_train_val, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=13)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=13)

    return x_train_val, x_test, y_train_val, y_test, x_train, x_val, y_train, y_val