###### Hyperparameter Tuning  ########

## CONTENTS :

# 1. GRIDSEARCHCV
# 2. RANDOMSEARCHCV
# 3. BAYES OPTIMISATION

#### 1. GridSearchCV Approach  ###

lgbm_clf = lgbm.LGBMClassifier()
params = {
   'class_weight':[None,'balanced'],
   'feature_fraction':[0.3,0.4,0.5], # i.e. number of features to randomly choose between at a given split #
    'learning_rate':[0.03,0.08],
    'random_state':[100],
    'max_depth':[6,8,10],
    'eval_metric':['auc'],
    'n_estimators':[100,150],
    'objective': ['binary']
}
best_clf = GridSearchCV(lgbm_clf, params, n_jobs=-1, cv=5)  #  refit=True i.e. default means that once best hyperparameters found, will retrain it on the whole training set - in this way improves performance as more data
best_clf.fit(x_train, y_train)
print('Best parameters found:\n', best_clf.best_params_)
# Having found the best architecture for our model, we fit it on the entirety of the training set, and then can assess it against our test set  
best_clf.predict ### etc

#### 1b GridSearchCV Approach using a pipeline (might be necessary where handling nulls involved mean imputation. In which case, would want to apply this on each fold rather than before to avoid data leakage!) ###

lgbm_clf = lgbm.LGBMClassifier()
params = {
   'class_weight':[None,'balanced'],
   'feature_fraction':[0.3,0.4,0.5], # i.e. number of features to randomly choose between at a given split #
    'learning_rate':[0.03,0.08],
    'random_state':[100],
    'max_depth':[6,8,10],
    'eval_metric':['auc'],
    'n_estimators':[100,150],
    'objective': ['binary']
}

pipe = Pipeline([
    ('mean', SimpleImputer(missing_values=np.nan, strategy='mean'),
    ('clf', RandomForestClassifier(random_state=2))])
params = {
   'clf__n_estimators':[2, 5],
   'clf__max_depth':[3, 4]}
grid_search = GridSearchCV(pipe, param_grid=params, cv=5)
## then fit it then ..
grid_search.best_params_

# Two consecutive dictionaries 
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},   # will first try 3*4 possible combinations
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}, # then try 2*3 possible combinations this time without bootstrapping
 ]

#### 2. Optuna Approach #####

## Most recent approach : for content tiering optimisation ###

import optuna
optuna.logging.set_verbosity(0)

# Define objective function for hyperparameter optimization
def objective(trial):
    
    variable_params = {
        
        'n_estimators':trial.suggest_int('n_estimators', 60, 180),
        'max_features':trial.suggest_float('max_features', 0.2, 0.8),
        'max_depth':trial.suggest_int('max_depth', 3, np.round(np.sqrt(len(top_features))+2))
        
    }
    
    clf_opt = RandomForestRegressor(**variable_params)
    clf_opt.fit(X_train, y_train_continuous)
    
    y_preds_test_rounded = np.round(clf_opt.predict(X_test)).astype('int64')
    test_accuracy = accuracy_score(y_test_continuous, y_preds_test_rounded)
    
    return test_accuracy

study = optuna.create_study(study_name='optimize_model', direction='maximize')
study.optimize(objective, timeout=12000, show_progress_bar=True)
best_params_optuna = study.best_params
best_test = study.best_value
df_trials = study.trials_dataframe().set_index('number')
# Save determined best params
with open('config/best_optuna_params.pickle', 'wb') as file:
    pickle.dump(best_params_optuna, file, protocol=pickle.HIGHEST_PROTOCOL) # The higher the protocol used, the more recent the version of Python needed to read the pickle produced - so here uses most the version of python you haveinstalled. Means less backward compatibility. 


    ######################

# https://towardsdatascience.com/optuna-a-flexible-efficient-and-scalable-hyperparameter-optimization-framework-d26bc7a23fff 

import optuna
optuna.logging.set_verbosity(0)

from sklearn.model_selection import KFold, cross_validate

#### OPTUNA WITH CROSS VALIDATION TO ENSURE THAT WE ARE NOT PEAKING INTO TEST SET ######
def objective(trial, X_train, y_train, cv, scoring):

    # Suggest values of the hyperparameters using a trial object.
    param = {
        
        'learning_rate':trial.suggest_loguniform('learning_rate', 0.001, 0.2),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'n_estimators':trial.suggest_int('n_estimators',600,2000),
        'subsample':trial.suggest_float('subsample', 0.5, 0.9),
        
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        
    }
    
    ## Using the training API rather than the sklearn API - believe it is quicker - but need to pass in lgbm dataset ## 
    lgbm_tuning = lgbm.LGBMClassifier(**param,random_state=random_state,metric='auc')
    scores = cross_validate(lgbm_tuning, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1,error_score='raise')
    mean_scores_folds = scores["test_score"].mean() # want the mean score across the folds
    
    return mean_scores_folds

# Create study that maximises AUC
study = optuna.create_study(study_name='cv_optimisation',direction="maximize")
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
func = lambda trial: objective(trial, X_train, y_train, cv=kf, scoring=('roc_auc'))
study.optimize(func, n_trials=40)


##### OPTUNA USING CROSS VALIDATION BUT ALSO EARLY STOPPING WITHIN THE LOOP ######

from optuna.integration import LightGBMPruningCallback

def objective(trial, X, y, cv_folds):
    param_grid = {
        
        
        'learning_rate':trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'n_estimators':trial.suggest_int('n_estimators',600,2000),
        'subsample':trial.suggest_float('subsample', 0.5, 0.9),
        
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_scores = np.empty(cv_folds)
    
    X = X.reset_index().drop(columns='account_number')
    y = y.reset_index().drop(columns='account_number')
    
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx]['target'], y.iloc[test_idx]['target']

        model = lgbm.LGBMClassifier(**param_grid,random_state=random_state,metric='auc')
        model.fit(
            X_train_cv,
            y_train_cv,
            eval_set=[(X_test_cv, y_test_cv)],
            eval_metric="auc",
            early_stopping_rounds=30,
            callbacks=[
                LightGBMPruningCallback(trial, "auc")
            ],  # Add a pruning callback
        )
        proba_preds = model.predict_proba(X_test_cv)[:,1]
        cv_scores[idx] = roc_auc_score(y_test_cv, proba_preds)

    return np.mean(cv_scores)

# Create study that maximises AUC
study = optuna.create_study(study_name='cv_optimisation',direction="maximize")
func = lambda trial: objective(trial, X_train_subset[0:num_samples], y_train[0:num_samples]['target'],cv_folds=5)
study.optimize(func, n_trials=20)


############################################





objective = 

study = optuna.create_study(study_name='optimize_LGBM', direction='maximize')
study.optimize(objective, timeout=27000, n_jobs=1) # currently set to 7.5 hours - maybe add a stopping criterion once AUC only increases by 0.05 after say 50 more iterations
best_params = study.best_params
best_AUC = study.best_value
df_trials = study.trials_dataframe().set_index('number')
# Save determined best params
with open('data/best_indiv_model_params.pickle', 'wb') as file:
    pickle.dump(best_params, file, protocol=pickle.HIGHEST_PROTOCOL) # The higher the protocol used, the more recent the version of Python needed to read the pickle produced - so here uses most the version of python you haveinstalled. Means less backward compatibility. 
df_trials.drop(columns='duration').to_parquet('data/df_trials_indiv_model.pq')


####

import optuna
optuna.logging.set_verbosity(0)
action = 'load'

if action == 'load':
    # Load determined best params
    with open('best_indiv_model_params.pickle', 'rb') as file:
        best_params = pickle.load(file)
    df_trials = pd.read_parquet('df_trials_indiv_model.pq')

import optuna
optuna.logging.set_verbosity(0)

# Define objective function for hyperparameter optimization
def objective(trial):
    
    variable_params = {
        
        'learning_rate':trial.suggest_loguniform('learning_rate', 0.001, 0.2),
        'n_estimators':n_estimators = trial.suggest_int('n_estimators', 100, 350),
        'num_leaves':trial.suggest_int('num_leaves', 20, 60),
        'max_depth':trial.suggest_int('max_depth', 4, 14),
        'colsample_bytree':trial.suggest_uniform('feature_fraction', 0.4, 0.8)
        
    }
    
    clf_opt = lgbm_classifier(X_train_subset, y_train.values.ravel(), X_test_subset, y_test.values.ravel(), **variable_params)
    
    return clf_opt._best_score['valid_1']['auc']

# Optimize hyperparams or load optimized hyperparams
action = 'tune'

# if action == 'load':
#     # Load determined best params
#     with open('best_indiv_model_params.pickle', 'rb') as file:
#         best_params = pickle.load(file)
#     df_trials = pd.read_parquet('df_trials_indiv_model.pq')
        
elif action == 'tune':
    study = optuna.create_study(study_name='optimize_LGBM', direction='maximize')
    study.optimize(objective, timeout=27000, n_jobs=-1) # currently set to 7.5 hours
    best_params = study.best_params # study.best_trial.params ?? 
    best_AUC = study.best_value
    df_trials = study.trials_dataframe().set_index('number')

    # Save determined best params
    today = datetime.now().strftime('%Y_%m_%d')
    with open('best_indiv_model_params_{}.pickle'.format(today), 'wb') as file:
        pickle.dump(best_params, file, protocol=pickle.HIGHEST_PROTOCOL)
    df_trials.drop(columns='duration').to_parquet('df_trials_indiv_model_{}.pq'.format(today))
    
# Visualize optimization path
df_trials.sort_values(by='value', inplace=True)
df_trials.reset_index(drop=True, inplace=True)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(22,15))

ax1.plot(df_trials.index, df_trials.value, 'ro')
ax1.set_ylabel('Validation AUC')
ax1.grid()

ax2.plot(df_trials.index, df_trials.params_feature_fraction, 'ro')
ax2.set_ylabel('Feature frac')
ax2.grid()

ax3.plot(df_trials.index, df_trials.params_min_sum_hessian_in_leaf, 'ro')
ax3.set_ylabel('min. Hessian')
ax3.grid()

ax4.plot(df_trials.index, df_trials.params_learning_rate, 'ro')
ax4.set_ylabel('log(Learning rate)')
ax4.set_yscale('log')
ax4.grid()

ax5.plot(df_trials.index, df_trials.params_num_leaves, 'ro')
ax5.set_ylabel('Num. leaves')
ax5.grid()


### Generalised Optuna Approach : as optuna is framework agnostic 


def optimize(self, trial):
    
    # [1] Search Spaces Definition
    # E.g:
    hyperparamter_a = trial.suggest_int('hyper_a', 0, 10)
    
    # [2] Your PyTorch, Scikit-learn, Keras, etc, model. 
    # E.g:
    model = Model(hyperparamter_a) 
    model.fit(x_train, y_train)
    
    # [3] Function to be optimized (maximized or minimized: 
    # E.g: error, accuracy, mse, etc.
    error = model.get_error()
 
    # Value to be maximized or minimized
    # In this case, error aims to be minimized
    return error

# Study definition
study = optuna.create_study(direction='minimize')

# Starts optimization
study.optimize(optimize, n_trials=100)

##### 3. Bayesian Optimisation ######

def bayes_parameter_opt_lgb(X, y, init_round=20, opt_round=30, n_folds=5, random_seed=6, n_estimators=10000,learning_rate=0.05, output_process=False):
    # prepare data

    train_data = lgb.Dataset(data=X, label=y)
    # parameters

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, 
                 lambda_l1, lambda_l2, min_split_gain, min_child_weight):

        params = {'objective':'regression','num_iterations':1000, 'learning_rate':0.05,
                  'early_stopping_round':100, 'metric':'rmse'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)

        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        
        cv_result = lgb.cv(params, train_data, nfold=3, seed=random_seed,
                           stratified=False, verbose_eval =200, metrics=['rmse'])

        return min(cv_result['rmse-mean'])

    # setting range of the parameters
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.5, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 60)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return
    return lgbBO

opt_params = bayes_parameter_opt_lgb(train_X, train_y, init_round=5, opt_round=10, n_folds=3,
                                     random_seed=6, n_estimators=10000, learning_rate=0.05)
params = opt_params.max['params']                                    
params
