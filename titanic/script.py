import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.set_context('paper')

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

# GET THE DATA
# import training data 
data = pd.read_csv('Challenges/titanic/train.csv')
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
# Check for missing values 
data.isna().sum()
incomplete_rows = data[data.isnull().any(axis=1)]
print(incomplete_rows)
# Drop the columns with excessive missing values
data = data.drop(['Cabin'], axis=1) 
data = data.set_index('PassengerId') # set as index
# Fill Age NaN's
median_age = data['Age'].median() 
data['Age'].fillna(median_age, inplace=True) # fill with median age 
# Fill Embarked NaNs
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].value_counts().index[0])

# Create label 
y = data['Survived'].copy() # create labels 
data = data.drop('Survived', axis=1) # create data minus labels 

# Encode categorical data 
data_cat = data.select_dtypes(exclude='number')
encoder = OneHotEncoder()
data_cat_enc = pd.DataFrame(encoder.fit_transform(data_cat), columns=data_cat.columns, index=data_cat.index)

# Standardize numerical data 
data_num = data.select_dtypes(include='number')
scaler = StandardScaler()
data_scal = pd.DataFrame(scaler.fit_transform(data_num), columns=data_num.columns, index=data_num.index)

# Create label and training datasets 
X = pd.concat([data_scal, data_cat_enc], axis=1)
print(f"Processed Training DataFrame Shape: {X.shape}")
print(f"Labels (target) shape: {y.shape}")
# Create a holdout set 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




# MODEL SELECTION and CROSS VALIDATION
# Set the scorer 
acc = make_scorer(accuracy_score)
# Function for the scores, mean and std
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# LinReg model
log_reg = LogisticRegression()
# Cross validation
cross_val = cross_val_score(estimator=log_reg, X=X_train, y=y_train,
                     cv=5, scoring=acc)
# Lin Regression scores, mean and std
display_scores(cross_val)

# LinSVM model
lin_svm = LinearSVC()
# Cross validation
cv_svm = cross_val_score(estimator=lin_svm, X=X_train, y=y_train,
                     cv=5, scoring=acc)
# Lin SVM scores, mean and std
display_scores(cv_svm)

# Ensemble model
rfc = RandomForestClassifier(n_estimators=300, random_state=0)
# Cross validation
cv_rfc = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=5,
                     scoring=acc)
# Rand Forest scores, mean and std
display_scores(cv_rfc)



# HYPERPARAMETER TUNING: RandomizedSearchCV
# Current parameters in use
base_rfc_params = rfc.get_params()
print(base_rfc_params)
# RFR Hyperparameter Tuning
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
print(random_grid)

# RandomizedSearchCV
rand_search = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,
                        scoring=acc, cv=5, n_iter = 100, )    
result = rand_search.fit(X_train, y_train)
# Results summary
# View best hyperparameters
best_rfc_params = result.best_params_
print(best_rfc_params)

# Evaluate Randomized Search on Validation Set
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

# Baseline model accuracy
rfc.fit(X_train, y_train)
base_accuracy = evaluate(rfc, X_val, y_val)

# Best model
best_rand_model = result.best_estimator_
best_accuracy = evaluate(best_rand_model, X_val, y_val)

# Analysing Errors on Validation set
rfc_opt = RandomForestClassifier(**best_rfc_params) # Instantiate the model
rfc_opt.fit(X_train, y_train) # Fit our model
preds = rfc_opt.predict(X_val) # Predict the holdout values
c_matrix = (pd.DataFrame(confusion_matrix(y_val, preds))
 .rename_axis('Actual')
 .rename_axis('Predicted', axis='columns'))
sns.heatmap(c_matrix, annot=True)
plt.show()
rfc_opt.score(X_val, y_val)




# TEST ON UNSEEN DATA
# Import the test data
X_test = pd.read_csv('Challenges/titanic/test.csv')
print(f"Test data shape: {X_test.shape}")

# Check for missing values 
X_test.isna().sum()
X_test = X_test.drop(['Cabin'], axis=1)
X_test = X_test.set_index('PassengerId') # set as index
# Fill Age NaN's
median_age = X_test['Age'].median() 
X_test['Age'].fillna(median_age, inplace=True) # fill with median age 
# Fill Fare NaN
X_test['Fare'].fillna(data['Fare'].value_counts().index[0])

# Encode categorical data 
test_cat = X_test.select_dtypes(exclude='number')
enc = OrdinalEncoder()
test_cat_enc = pd.DataFrame(enc.fit_transform(test_cat), columns=test_cat.columns, index=test_cat.index)

# Standardize numerical data 
test_num = X_test.select_dtypes(include='number')
scaler = StandardScaler()
test_scal = pd.DataFrame(scaler.fit_transform(test_num), columns=test_num.columns, index=test_num.index)

# Create label and training datasets 
test_labels = pd.concat([test_scal, test_cat_enc], axis=1)

# Try model
test_labels['predictions'] = rfc.predict(test_labels)
print(test_labels.head())

# CSV for submission 
survival_submission = test_labels['predictions']
survival_submission.to_csv('Titanic Submission File.csv')
