# Classify the different types of penguin based on different parameters.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)  # to check if the model is working properly or not


df = pd.read_csv("penguins.csv")
# print(df.head()) # .head prints only frst 5 rows

# drop missing values

df.dropna(axis=0, how="any", subset=None, inplace=True)  # there is a parameter called thresh which says the amount of Nan values that are acceptable in each row.


# one-hot encoding
df = pd.get_dummies(df, columns=["island", "sex"])
print(df.head())

# assign X and y Variables
X = df.drop("species", axis=1)
y = df["species"]

# spilt data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# Algorithm
model = LogisticRegression()

# assign values
model.fit(X_train, y_train)

# formula y = ax+b
# y intercept - b
print(model.intercept_)

# x coefficient - a
print(model.coef_)

# run algorithm to test
mode_test = model.predict(X_test)

# evaluate result
print(confusion_matrix(y_test, mode_test))
print(classification_report(y_test, mode_test))


# make predictions
feature_names = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island_Biscoe",
    "island_Dream",
    "island_Torgersen",
    "sex_FEMALE",
    "sex_MALE",
]

penguin = [
    39,
    18.5,
    180,
    3750,
    0,
    0,
    1,
    1,
    0,
]

penguin_data = pd.DataFrame([penguin], columns=feature_names)
predict = model.predict(penguin_data)
print(predict)
