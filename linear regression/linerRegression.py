# Predict the value of tips using the linear regression model

#libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error # to check if the model is working properly or not


# import dataset
df  = pd.read_csv('tips.csv')
df.head()
del df['smoker'] #3 this is not required snce this is not adding value to the dataset

## dataset analysis using heatmap - check correlation
df_numeric = df.select_dtypes(include=['number'])
df_corr = df_numeric.corr()
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.show() # without plt this, we will not be able to show the graph. 

 
## dataset analysis using pairplot

sns.pairplot(df)
plt.show()

# convert non-numeric to one-hoy encoding
df = pd.get_dummies(df, columns=['time', 'day','sex'])
print(df.head())

# check correlation between the dependent and the independent variable
df_corr = df.corr()
print(df_corr)

# assigna X (independent) and y(dependent) variable
# axis 1 -> Column based | axis 0 -> row based
X = df.drop(['tip'], axis=1) # try making changes by removing the values with less correlation. 
y = df['tip']

# split data into test/train set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=True) # test = 30%: train = 70%

# Algorithm 
model = LinearRegression()

# Link model to X and y
model.fit(X_train, y_train)


# find y intercept
print(model.intercept_)

# find X coefficients

print(model.coef_)


# mean abolute error
mae_train = mean_absolute_error(y_train, model.predict(X_train))
print(" train set MAE: %.2f" %mae_train)

mae_test = mean_absolute_error(y_test, model.predict(X_test))
print(" test set MAE: %.2f" %mae_test)


# make predictions
feature_names = ['total_bill', 'size', 'time_Dinner', 'time_Lunch', 'day_Fri', 'day_Sat', 'day_Sun', 'day_Thur', 'sex_Female', 'sex_Male']

jamie = [
    30, #total_bill       
    2,# size  
    1,# time_Dinner  
    0,# time_Lunch   
    0,# day_Fri   
    0,# day_Sat   
    1,# day_Sun  
    0,# day_Thur  
    0,# sex_Female  
    1,# sex_Male
]
jamie_data = pd.DataFrame([jamie], columns=feature_names)
jamie_tip = model.predict(jamie_data)
print(jamie_tip)