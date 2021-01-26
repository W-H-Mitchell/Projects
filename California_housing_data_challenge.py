import os
import tarfile
import urllib
import urllib.request
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Common imports
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rc('axes', labelsize=12)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)

np.random.seed(42)

# GETTING DATA
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(housing.head())
print(f"Data shape: {housing.shape}")

# HOLDOUT SET
# from sklearn use random sampling method 
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head()


# EXPLORATORY DATA ANALYSIS
# Check data types and info
df_dtypes = housing.dtypes.reset_index()
df_dtypes.columns = ['Count', 'Data Type']
df_dtypes.groupby("Data Type").aggregate('count').reset_index() 

# Describe data. Nominal or ordinal categorical data? 
print(housing.describe()) # Numerical data
print(housing.describe(include=['object'])) # Categorical data

# CATEGORICAL DATA 
housing_cat = housing.select_dtypes(exclude='number')
# Examine the variation in frequency of different neighborhoods
sns.countplot(housing_cat['ocean_proximity'])
plt.show()
# Check relationship to Median House Value
ax = sns.violinplot(x="ocean_proximity", y="median_house_value", data=housing)
plt.xticks(rotation=90)
plt.show()




# NUMERICAL DATA 
# Examine target variable
sns.distplot(
   housing['median_house_value'], norm_hist=False, kde=False, bins=40, hist_kws={"alpha": 1}
   ).set(xlabel='House Price', ylabel='Count')
plt.show()

# Scatter plots
# Location fig; radius of circles = population and color = median house price
fig1 = housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False) # jet is a predefined color map with hot and cold colors
plt.legend()
plt.show()
# fig1.savefig("housing_prices_scatterplot.pdf")

# Correlations witin the data
corr_matrix = housing.corr(method='pearson')
attributes = corr_matrix.nlargest(5, "median_house_value").index
fig3 = pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 12))
plt.show()
# fig3.savefig("scatter_matrix_plot.pdf")

# Heatmap of correlation matrix 
corr_matrix["median_house_value"].sort_values(ascending=False)
sns.heatmap(corr_matrix, annot=True, )



# PREPROCESSING - Preparing the data
# drop target labels for training set
housing_prepared = train_set.drop("median_house_value", axis=1) # working with train set ONLY
housing_labels = train_set["median_house_value"].copy()

# Fill incomplete rows
imputer = SimpleImputer(strategy='median')
housing_num = housing_prepared.drop(['ocean_proximity'], axis=1) # imputer can only work on numerical attributes
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_imputed = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# Standardization using StandardScaler
scaler = StandardScaler()
housing_scaled = pd.DataFrame(scaler.fit_transform(housing_imputed), columns=housing_imputed.columns, index=housing_imputed.index)
# Label encoder; convert categorical data to numerical data


# MODEL SELECTION
# Kfolds
kfold = StratifiedKFold(n_splits=5)

# Choose method to score model 
mse = make_scorer(mean_squared_error)
# Function for the scores, mean and std
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Selecting and instantiating the model
lin_reg = LinearRegression()
# Evaluate the model using cross validation
cross_val = cross_val_score(estimator=lin_reg, X=housing_scaled, y=housing_labels,
                     cv=kfold,scoring=mse)
# Lin Regression scores, mean and std
display_scores(cross_val)

# Selecting a second model
rfc = RandomForestRegressor(random_state=42)
# Evaluate the model; don't need .fit() and .predict() stages
# Cross Validation: set up cross_val_score
cv = cross_val_score(estimator=rfc,
                     X=housing_scaled,
                     y=housing_labels,
                     cv=10,
                     scoring=mse)
# RandomForest scores, mean and std
display_scores(cv)

# Selecting a third model
tree_reg = DecisionTreeRegressor()
tree_cv = cross_val_score(estimator=tree_reg, X=housing_scaled, y=housing_labels,
                     cv=10,
                     scoring=mse)
# DecisionTree scores, mean and std
display_scores(tree_cv)

# Hyperparameter Tuning (dropped Lin_reg)
# Parameter_dictionary for RandomForestRegressor
param_dict = {
    "max_depth": [2, 4, 6, 8],
    "max_features": [2, 4, 6, 8, 10],
    "min_samples_split": [2, 4, 8, 16]
    }   
grid_search = GridSearchCV(rfc, param_dict, cv=5,
                           scoring=mse,
                           return_train_score=True)
grid_search.fit(housing_scaled, housing_labels)
grid_search.best_estimator_
grid_search.best_score_

# Parameter_dictionary for DecisionTreeRegressor
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
# train across 5 folds; total of 90 training rounds 
grid_search_tree = GridSearchCV(tree_reg, param_grid, cv=5,
                           scoring=mse,
                           return_train_score=True)
grid_search_tree.fit(housing_scaled, housing_labels)
grid_search_tree.best_estimator_
grid_search_tree.best_score_

# Finally, test models on hold out set 
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()