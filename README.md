![insulin-level-monitor-hero](https://user-images.githubusercontent.com/122539866/233172446-8c0b2dc7-e0fc-4e54-af07-bcfc63ee3374.jpg)

-----


## About

The diabetes dataset can be used to predict whether a patient has diabetes or not based on certain features such as Number of times pregnant, BMI, age, and more. The dataset can be used for various purposes such as:

- Demonstrating the basic steps of a machine learning workflow, including data loading, data preprocessing, model training, and model evaluation.


- Evaluating the performance of various classification models, such as logistic regression, decision trees, random forests, and more, on the diabetes dataset.
This dataset can also be said to provide real-world problem with well-defined features and a clear objective.

## Dateset

This dataset is initially from pycaret libraries and then saved as a `csv file`. 

```bash python

# import PyCaret classification module and load diabetes dataset
from pycaret.classification import *
diabetes_data = get_data('diabetes')

# convert diabetes dataset to CSV
diabetes_data.to_csv('diabetes.csv', index=False)

```

## Creating a machine learning pipeline with scikit learn(sklearn)

The following steps were taken:

🔔 Set up the environment, Load the dataset and Check for nulls


```bash python

# load sample dataset
import pandas as pd
import seaborn as sns

from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from yellowbrick.regressor import PredictionError

df = pd.read_csv('./diabetes.csv')
print(df.shape)
df


# Renaming the column Class variable
df = df.rename(columns={'Class variable': 'Class_variable'})
df


# simple check for nulls
df.isna().sum()[df.isna().sum() > 0]


profile = ProfileReport(df)
profile.to_notebook_iframe()
profile.to_file('./reg_diabetes.html')

```

🔔 Set and save unseen data


```bash python

# set aside and save unseen data set
data_unseen = df.sample(n=100, random_state=42)
data        = df.drop(data_unseen.index)
print(f'Data for model: {data.shape},\nData for unseen predictions: {data_unseen.shape}')
data_unseen.to_csv('./diabetes_unseen.csv', index=False)

```


```bash python

# data.columns!='Class_variable'
X = data.loc[: , data.columns!='Class_variable']
y = data.loc[: , data.columns=='Class_variable']


X
```


🔔 Split the data into training and test

```bash python

# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

```bash python

# encoding 
# get the categorical and numeric column names
num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(num_cols, '\n', cat_cols)

```

🔔 Pipelines(numerical and categorical columns)

```bash python

# pipeline for numerical columns
num_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)
num_pipe



# pipeline for categorical columns
cat_pipe = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='N/A'),
    OneHotEncoder(handle_unknown='ignore', sparse=False)
)
cat_pipe

```

🔔 Combining both pipelines

```bash python

# combine both the pipelines
full_pipe = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])
full_pipe

```

🔔 Build the model

```bash python

# build the model
gbr_diabetes = make_pipeline(full_pipe, GradientBoostingRegressor(random_state=42))
gbr_diabetes

```

🔔 Train the model

```bash python

# train the model
gbr_diabetes.fit(X_train, y_train)

```


🔔 Predict the model

```bash python

# make predictions on the test set
y_pred = gbr_diabetes.predict(X_test)

```


```bash python

# measure accuracy
print('R2:', r2_score(y_test, y_pred))


# done manually to break out the example above
y_test['y_pred'] = y_pred
test_scores = y_test.copy()
test_scores

```

```bash python

r2 = r2_score(test_scores['Class_variable'], test_scores['y_pred'])
mae = mean_absolute_error(test_scores['Class_variable'], test_scores['y_pred'])
mean_act = test_scores['Class_variable'].mean()
mean_pred = test_scores['y_pred'].mean()
mape = mean_absolute_percentage_error(test_scores['Class_variable'], test_scores['y_pred'])
print(f'R2: {r2}\nmae: {mae}\nact_mean: {mean_act}\npred_mean: {mean_pred}\nmape: {mape}')

```

```bash python
import joblib
joblib.dump(gbr_diabetes, './diabetes.pkl')
print(gbr_diabetes)

```

----

## [Click To View My Notebook](https://nbviewer.org./github/Ofomn/pipeline_with_Sklearn/blob/cba97ef0caa65c7a9ba8e5d3f13263604b974bc9/Pipeline-Regression-sklearn.ipynb)

-----



# Technologies Used

* [skLearn](https://scikit-learn.org/stable/index.html)
* [pandas](https://pandas.pydata.org/)
* [seaborn](https://scikit-learn.org/stable/index.html)
* [Yellowbrick](https://www.scikit-yb.org/en/latest/)


---

 ## License 
 Licensed under GPL-3.0 license

© Ofomnbuk 2023 🇨🇦😘🇳🇬


