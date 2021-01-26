import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.set_context('paper')

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# GET THE DATA
# import training data 
data = pd.read_csv('titanic/train.csv')
print(data.head())
columns = data.columns
index = data.index

# check the data and dtypes
print(data.describe()) # age columns missing some data
print(data.describe(include=['bool', 'object']))
df_dtypes = data.dtypes.reset_index()
df_dtypes.columns = ['Count', 'Data Type']
print(df_dtypes.groupby("Data Type").aggregate('count').reset_index())



# EDA
data.info() 
# CATEGORICAL DATA
data_cat = data.select_dtypes(exclude='number')
col = data_cat.columns
unique_values = pd.DataFrame.from_records([(col, data[col].nunique()) for col in data.columns],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
print(unique_values)

# Examine the infuence of sex and embarked on survived (=1)
data.groupby('Sex')['Survived'].count()
# Visualise the number of survivers
ax = sns.countplot(data=data[data['Survived'] == 1], x='Sex')
ax.set_ylabel("Number of survivers")
plt.show()
# Visualise deaths
# ax = sns.countplot(data=data[data['Survived'] == 0], x='Sex')
# ax.set_ylabel("Number of deaths")
# plt.show()

# View categorical feature relationship to deaths
features = ['Pclass', 'SibSp', 'Embarked', 'Parch', 'Sex']
fig, ax = plt.subplots(3, 2, figsize=(15, 10))
for var, subplot in zip(features, ax.flatten()):
    sns.countplot(x=var, data=data[data['Survived'] == 0], ax=subplot)
    subplot.set_ylabel("Number of Deaths")
plt.show()


# NUMERICAL DATA
data_num = data.select_dtypes(include='number')
corr_matrix = data.corr(method='pearson')
attrib = corr_matrix.nlargest(len(corr_matrix), 'Survived').index
sns.heatmap(corr_matrix, annot=True, cbar=True, square=True, 
            fmt='.2f', yticklabels=attrib.values, xticklabels=attrib.values)
plt.show()

# Scatter matrix
pd.plotting.scatter_matrix(data=data_num, alpha=0.3, diagonal='kde')
plt.show()

# PREPROCESSING 
# Drop the columns with excessive missing and unhelpful features 
data_temp = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# Check for missing values
incomplete_rows = data_temp[data_temp.isnull().any(axis=1)]
print(incomplete_rows) # 179 incomplete rows with NaN's in the age column
median_age = data_temp['Age'].median() 
data_temp['Age'].fillna(median_age, inplace=True) # fill with median age 
data_temp = data_temp.dropna()

# Drop the labels from data 
X = data_temp.drop('Survived', axis=1) # create data minus labels 
y = data_temp['Survived'].copy() # create labels 

# Encode categorical data 
X_cat = X[['Sex', 'Embarked']]
encoder = OrdinalEncoder()
X_cat_enc = pd.DataFrame(encoder.fit_transform(X_cat), columns=X_cat.columns, index=X_cat.index)
# Standardize numerical data 
X_num = X.drop(['Sex', 'Embarked'], axis=1)
scaler = StandardScaler()
X_scal = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns, index=X_num.index)

# Join categorical and numerical data, create validation set
X_trans = X_cat_enc.join(X_scal)
print(X_trans.info())
# Create a holdout set 
X_train, X_val, y_train, y_val = train_test_split(X_trans, y, test_size=0.2, random_state=42)

# MODEL SELECTION and CROSS VALIDATION
# Set the scorer 
acc = make_scorer(accuracy_score)
# Function for the scores, mean and std
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Selecting and instantiating the model
log_reg = LogisticRegression()
# Cross validation
cross_val = cross_val_score(estimator=log_reg,
                     X=X_train,
                     y=y_train,
                     cv=5,
                     scoring=acc)
# Lin Regression scores, mean and std
display_scores(cross_val)

# Selecting and instantiating the model
lin_svm = LinearSVC()
# Cross validation
cv_svm = cross_val_score(estimator=lin_svm,
                     X=X_train,
                     y=y_train,
                     cv=5,
                     scoring=acc)
# Lin SVM scores, mean and std
display_scores(cv_svm)

"""
Try SVC
"""

# Selecting and instantiating the model
rfc = RandomForestClassifier(n_estimators=300, random_state=0)
# Cross validation
cv_rfc = cross_val_score(estimator=rfc,
                     X=X_train,
                     y=y_train,
                     cv=5,
                     scoring=acc)
# Rand Forest scores, mean and std
display_scores(cv_rfc)

# HYPERPARAMETER TUNING 
# LogReg Hyperparameters
parameters = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'solver': ["newton-cg", "lbfgs", "liblinear"]
    }
# GridSearch
grid_search = GridSearchCV(estimator=log_reg, 
                        param_grid=parameters,
                        scoring=acc,
                        cv=5)    
grid_result = grid_search.fit(X_train, y_train)
# Results summary
# View best hyperparameters
print('Best Penalty:', grid_result.best_estimator_.get_params()['penalty'])
print('Best C:', grid_result.best_estimator_.get_params()['C'])
print('Best C:', grid_result.best_estimator_.get_params()['solver'])
best_lr_params = grid_result.best_params_

# Analysing Errors on Validation set
lr = LogisticRegression(**best_lr_params) # Instantiate the model
lr.fit(X_train, y_train) # Fit our model
lr_predict = lr.predict(X_val) # Predict the holdout values
c_matrix = (pd.DataFrame(confusion_matrix(y_val, lr_predict))
 .rename_axis('Actual')
 .rename_axis('Predicted', axis='columns'))
sns.heatmap(c_matrix, annot=True)
plt.show()



# Try model on unseen test set
# Import the test data
test = pd.read_csv('test.csv')
print(test.head())

# Try model
test_lr = LogisticRegression(**best_lr_params)
test_lr.fit(X_test, y_test)
test_predictions_probs = lr.predict_proba(X_test)
test_predictions = lr.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
print("My predictions for the survival rate onboard Titanic have an accuracy of: {1:.2f}".format(test_accuracy))
"""