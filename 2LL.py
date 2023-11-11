import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score


df = pd.read_csv("insclass_train.csv")

features = df.drop('target', axis=1)
target_variable = df['target']
RANDOM_STATE = 42

features.info()

data_types = df.dtypes
print(data_types)

numeric_columns = features.select_dtypes(include='number').columns

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(features[numeric_columns])
features[numeric_columns] = imputer.transform(features[numeric_columns])

scaler = StandardScaler()
features[numeric_columns] = scaler.fit_transform(features[numeric_columns])
features["variable_0"] = np.ones((features.shape[0]))

features = features.drop(['variable_7', 'variable_9', 'variable_15'], axis=1)

categorical_columns = ['variable_1', 'variable_5', 'variable_20', 'variable_21', 'variable_22', 'variable_28']

encoder = ce.TargetEncoder()

features[categorical_columns] = encoder.fit_transform(features[categorical_columns], target_variable)

features_train, features_test, target_train, target_test = train_test_split(features, target_variable, test_size=0.15, stratify=target_variable, random_state=RANDOM_STATE)

target_train = np.array(target_train).reshape(-1, 1)
target_test = np.array(target_test).reshape(-1, 1)