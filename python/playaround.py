##########################################
##### Notes 
# methods like StandardScaler() / pca() etc. have a fit() and transform() method as they transform the data 
# whereas kmean() etc. assigns labels, thus have a fit() and predict() method 


# .fit() is equal to training - learns statistical learning parameters - i.e. coefficient estimates
# .fit_predict()
# .fit_transform() 

gbt_imp.models_setup

### load packages ####

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical/Inference packages # 
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
# Predictive Modelling packages # 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score, roc_curve

# Hyperparameter Tuning # 
from sklearn.model_selection import GridSearchCV 

titanic = pd.read_csv(r"C:\Users\Owner\OneDrive\Stuart\LearningPython\VSCode_Python\DummyDatasets\titanicTrain.txt")

# Quick EDA #
titanic.info()
titanic.describe()
titanic.columns
titanic.drop(columns = ['PassengerId'], inplace = True)
sns.heatmap(titanic.corr())

# EDA # 
_ = titanic.hist()
plt.show(_)

# Preprocessing - outliers / missing values / dtypes
# change dtypes 
titanic = titanic.astype({'Pclass':'category'})

titanic_numeric = titanic._get_numeric_data()
# handle NAs #
titanic_numeric.isna().sum() # NAs per column 
titanic_numeric['Age'] = titanic_numeric['Age'].fillna(titanic_numeric['Age'].mean())
titanic_numeric = titanic_numeric.apply(lambda x: x.fillna(x.mean()),axis=0) # column-wise mean

# Feature Engineering 

# multicollinearity #
titanic_numeric.corr() # no sign of multicollinearity 
# Variance Inflation Factor - where 1=not correlated / >5 is highly correlated N.B. Requires intercept termvif = pd.DataFrame() #
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]; print(vif)


# transformations #
np.log1p(colName)   # handles 0s in the column as log0 is undefined and ln(x) approaches -inf as x approaches 0. log1p removes danger of large negative numbers 

# Dim reduction 

# 1. NMF - Non-Negative Matrix Factorisation # 
from sklearn.decomposition import NMF 
NMF_components = 40
nmf = NMF(n_components = NMF_components).fit(input_df)
W = nmf.transform(input_df)# W.shape gives 15900,40 array where m=no.rows, n=components
H = nmf.components_ # Factor Loadings for 40 components      
# N.B. could do A = W*H as part of sample reconstruction to approx A, our original matrix 
channel_nmf_df = pd.DataFrame(nmf.transform(input_df), index=input_df.index, columns=['Name your components here']) # Transforming my dataset and producing a dataframe output 

# 2. PCA 
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('reducer',PCA(n_components=10))]) 
principal_components = pipe.fit(df)
print(principal_components.steps[1][1].explained_variance_ratio_.cumsum())   # Proportion of variance each PC explains 
print(principal_components.steps[1][1].explained_variance_)                  # eigenvalues for each PC i.e. Magnitude of variance explained by each 
print(pd.DataFrame(principal_components.steps[1][1].components_, columns=list(df.columns))) 
# scree plot #
plt.plot(principal_components.steps[1][1].explained_variance_ratio_)
plt.title('Scree Plot')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(0,11, step=5))
plt.show()
# Once happy with your number of components, transform dataframe:
principal_components = pipe.fit_transform(df)
pca_df = pd.DataFrame(principal_components, columns = ['col1', 'col2'.............])



# train/test split 
# manual approach - 80% of rows taken at random - issue with this is cannot be certain that equal distribution of target variable ....
np.random.seed(345)
rand_rows = np.random.randint(len(titanic_numeric), size = round(len(titanic_numeric)*0.8)) 
titanic_train = titanic_numeric.iloc[rand_rows]

# sklearn approach # 
titanic_numeric = sm.add_constant(titanic_numeric) # for sm.logit, needs an intercept col
X = titanic_numeric.drop(columns = ['Survived'])
y = titanic_numeric['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 123)

# modelling #
mod = sm.Logit(y_train, X_train).fit()   
mod.summary() # inference - N.B. non-standardised coefficients so cannot compare like for like 
train_probpreds = mod.predict(X_train)     # rf.mod(X_train)[:, 1] ??? difference??
train_classpreds = (mod.predict(X_train) > 0.5).astype(int)
test_probpreds = mod.predict(X_test)
test_classpreds = (mod.predict(X_test) > 0.5).astype(int)


# model evaluation 
y_test.value_counts(dropna = False)
# Manual confusion matrix - this works, because order maintained in each series - so comparing elementwise
confusionMatrix = pd.crosstab(train_classpreds, y_train, rownames = ['ClassPreds'], colnames = ['Actual'], margins = False); confusionMatrix # margins = True gives row/column sums
confusionMatrix/confusionMatrix.sum(axis = 0) # proportional confusion matrix - need to ensure you read this correctly 
confusion_matrix(y_test, test_classpreds) # using sklearn - quite unneat though as array form

roc_auc_score(y_train, train_probpreds) # train AUC
roc_auc_score(y_test, test_probpreds) # test AUC

(train_classpreds == 1 & y_train == 1)
foo = map(lambda x,y: (x == 1 & y == 1).sum(), train_classpreds, y_train)

accuracy =  # TP + TN / (TP + TN + FP + FN)
classification_report(y_test, y_pred)

# accuracy = TP + TN / (TP + TN + FP + FN)
# precision = TP / TP + FP # how often the classifier is correct when it predicts positive 
# recall = TP / TP + FN  # how often the classifier is correct for all positive instances
# F1 score =  # how often the 
# ROC-AUC = 2 * (precision * recall ) / (precision + recall)




#### Random Forest - Feature Selection #######

model = RandomForestClassifier(n_estimators=100, criterion = 'gini', max_depth=8, random_state=12)
model.fit(x_train, y_train)
importances = model.feature_importances_
feature_subset = SelectFromModel(model, max)  # by default, will select those features which have importance score greater than the mean importance of all the features
Fit = feature_subset.fit_transform(df_dummies.drop(columns = 'canx', axis = 1), df_dummies['canx'])   # target variable - train['canx']
param_grid = {'n_estimators': [50, 100, 250],
              'min_samples_split': [2, 6, 10],
              'max_depth': [5, 15, 25],
              'criterion': ['entropy', 'gini']}


##### Linear Regression #######

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.coef_
lr.score(X_test, y_test)

##### regularised linear regression ###########
from sklearn.linear_model import Lasso
lasso_grid = {'alpha' = np.arange(0,1.1,0.1).tolist(), 'maxiter': [1000,1100]}
lasso = Lasso(alpha = 0.05)
lasso.fit(X_train, y_train)
lasso.coef_
lasso.score(X_test, y_test) # should return coefficient of determination 

LassoCV() # for hyperparameter tuning 

##### logistic regression ###########

X_train_std = StandardScaler().fit_transform(X_train)

############## ANOVA #####################

mod = ols('viewing_mins ~ affluence + package', df).fit()
anova_table = anova_lm(mod, typ=2).loc[:, ['F', 'PR(>F)']].drop('Residual').reset_index().rename(columns={'index': 'variable'})
# Apply multiple tests correction
multipletests(anova_table['PR(>F)']) # this might not work - but general idea is to adddress multiple testign problem i.e. bonferroni style correction that shrinks p-values

# Create a Support Vecotr Machine Classifier (SVM)
from sklearn.svm import SVC 
svc = SVC()
svc.fit(X_train, y_train) 

####### XAI ################
#! pip install SHAP 
import shap
