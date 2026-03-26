# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and remove unnecessary columns.

2.Convert categorical data into numerical form and split features and target variable.

3.Standardize the data and divide it into training and testing sets.

4.Train Ridge, Lasso, and ElasticNet models using Polynomial Features.

5.Predict test values and evaluate models using MSE and R² score, then visualize results.


## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: ROSHAN V
RegisterNumber:  25004228 // 212225240124
*/


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("caras.csv")
print(data.head())

data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

X = data.drop('price', axis=1)
y = data['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}
results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results[name] = {'MSE': mse, 'R² Score': r2}

print("\nName: ROSHAN V ") 
print("Reg No: 212225240124")

for model_name, metrics in results.items():
    print(f"\n{model_name} Model")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"R² Score: {metrics['R² Score']:.2f}")

results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.barplot(x='Model', y='MSE', data=results_df)
plt.title("Mean Squared Error Comparison")
plt.xticks(rotation=45)

plt.subplot(1,2,2)
sns.barplot(x='Model', y='R² Score', data=results_df)
plt.title("R² Score Comparison")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
