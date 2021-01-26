# Imports
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=12)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)
sns.set(style='darkgrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.set_context('paper')


# Collecting training data 
data = pd.read_csv('datasets/real-estate/data.csv') 
columns = [data.columns]
index = data.index
print(data.head())
print(f"Data shape: {data.shape}")

# EXPLORATORY DATA ANALYSIS
# Check general data types and number 
print(data.info()) 
# Calculate the number of each dtype
dtype_df = data.dtypes.reset_index() # reset_index() labels each row with and converts to a column
dtype_df.columns = ["Count", "Column type"]
print(dtype_df.groupby("Column type").aggregate('count').reset_index()) 

# Describe data. Any data capped? Nominal or ordinal categorical data? 
print(data.describe()) # Numerical data
print(data.describe(include=['object', 'bool'])) # Categorical data

# Examine relationships with target variable in the data
correlations = data.corr(method='pearson')
largest_correlations = correlations.nlargest(40, 'SalePrice').index
print(largest_correlations)
data = data[largest_correlations]



"""
# CATEGORICAL DATA
data_cat = data.select_dtypes(exclude='number')
print(data_cat.describe(include=['object', 'bool']))
# data_cat = data_cat.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = "columns")
# Unique values in a column
unique_counts = pd.DataFrame.from_records([(col, data_cat[col].nunique()) for col in data_cat.columns],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
print(unique_counts) # Max unique categories is neighborhood

# Examine the variation in frequency of different neighborhoods
sns.countplot(data_cat['Neighborhood'])
plt.show()
# Check relationship to SalePrice
ax = sns.boxplot(x="Neighborhood", y="SalePrice", data=data)
plt.xticks(rotation=90)
plt.show()

# Visualise some categorical features and their number counts
cat_features = ['Condition1', 'Condition2', 'HouseStyle', 'SaleType', 'SaleCondition', 'LotShape']
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
for variable, subplot in zip(cat_features, ax.flatten()):
    sns.countplot(data[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.show()
"""



# NUMERICAL DATA 
# Examine target variable
sns.distplot(
   data['SalePrice'], norm_hist=False, kde=False, bins=40, hist_kws={"alpha": 1}
   ).set(xlabel='Sale Price', ylabel='Count')
plt.show()

# Examine numerical data
# View numerical data with most correlated to sale price
data_num = data.select_dtypes(include='number')
corr = data_num.corr(method='pearson')
columns = corr.nlargest(10, 'SalePrice').index
print(columns)
# Unique values in a column
num_unique_counts = pd.DataFrame.from_records([(col, data_num[col].nunique()) for col in data_num.columns],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
print(num_unique_counts)

# View Multiple Feature relationship to Sale Price
features = ['OverallQual', 'GarageCars', 'YrSold', 'OverallCond', 'SaleCondition', 'FullBath']
fig, ax = plt.subplots(3, 2, figsize=(15, 10))
for var, subplot in zip(features, ax.flatten()):
    sns.violinplot(x=var, y='SalePrice', data=data, ax=subplot)
plt.show()

# Scatter_matrix plot
scatter_feat = ['SalePrice', 'GrLivArea', 'LotArea', '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'YearBuilt']
pd.plotting.scatter_matrix(data[scatter_feat], alpha=0.3, diagonal='hist', figsize=(12,12))
plt.show()

# Heatmap of correlation matrix 
corr = data[scatter_feat].corr(method='pearson')
correlation_map = np.corrcoef(data[scatter_feat].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)
plt.show()




# PREPROCESSING - Preparing the data
# Get rid of NaN
data_incomplete_rows = data[data.isna().any(axis=1)]
y = data['SalePrice'].copy()
data = data.drop("SalePrice", axis='columns')
# data = data.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley', 'PID', 'Order'], axis=1) # drop unnecessary columns

# Numerical data for imputer 
data_num = data.select_dtypes(include='number')
imputer = SimpleImputer(strategy='median')
imputer.fit(data_num)
X = imputer.transform(data_num)
data_imputed = pd.DataFrame(X, columns=data_num.columns, index=data_num.index)

# Standardization using StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns, index=data_imputed.index)


"""
# Encoding categorical data 
X_cat = data.select_dtypes(include='number')
print(X_cat.isna().sum()) # check NaNs under 10%
# Fill NaNs with most common value for each col
X_cat = X_cat.apply(lambda x: x.fillna(x.value_counts().index[0]))
# OrdinalEncoder 
encoder = OrdinalEncoder()
X_cat_enc = pd.DataFrame(encoder.fit_transform(X_cat), columns=X_cat.columns, index=X_cat.index) 
X_cat_enc.head()

# Join two dataframes
X = pd.concat([X_cat_enc, data_scaled], axis=1)
X.head()
# Check the dtypes
dtype_df = X.dtypes.reset_index() # reset_index() labels each row with and converts to a column
dtype_df.columns = ["Count", "Column type"]
print(dtype_df.groupby("Column type").aggregate('count').reset_index()) 
"""




# MODEL SELECTION
# Kfolds
kfold = StratifiedKFold(n_splits=5)

# Choose method to score model 
# scorer = make_scorer(mean_squared_error)
# Function for the scores, mean and std
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
# LinRegression
lr = LinearRegression()
lr_cv = cross_val_score(lr, X, y, cv=kfold, scoring='r2')
display_scores(lr_cv)

# Ridge
ridge = Ridge()
ridge_cv = cross_val_score(ridge, X, y, cv=kfold, scoring='r2')
display_scores(ridge_cv)

# Baynesian Ridge
BRidge = BayesianRidge()
BRidge_cv = cross_val_score(BRidge, X, y, cv=kfold,scoring='r2')
display_scores(BRidge_cv)

# SupportVector
svr = SVR(kernel='linear')
svr_cv = cross_val_score(svr, X, y, cv=kfold, scoring='r2')
display_scores(svr_cv)

# Rfr
rfr = RandomForestRegressor()
rfr_cv = cross_val_score(rfr, X, y, cv=kfold,scoring='r2')
display_scores(rfr_cv)

  