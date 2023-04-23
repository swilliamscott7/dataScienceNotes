

#### VARCLUSTERING (PCA under the hood) ####
# finds groups of features that are as correlated as possible among themselves and as uncorrelated as possible with features in other clusters
# Of the grouped correlation, will pick a feature (or two) that is representative of that cluster and keep it in the model
# A cluster is chosen for splitting.
# The chosen cluster is split into two clusters by finding the first two principal components, performing an orthoblique rotation, and assigning each variable to the rotated component with which it has the higher squared correlation.
# - Variables are iteratively reassigned to clusters to maximize the variance accounted for by the cluster components.
# - For each group we try to extract just one or two of these variables. Removal of correlated features is crucial when features importance methods are used to identify the most important drivers for a given target.
# Features selection from each cluster based on RS_ratio. Small values of this ratio indicate good clustering.
# RS_ratio = 1 - RS Own cluster / 1 - RS Next closest cluster
# Select one or two features from each cluster which are having lowest RS_RATIO.


# !pip install varclushi
from varclushi import VarClusHi

var_clust_model = VarClusHi(X_train,
                     maxeigval2=1, # Means clusters split if 2nd eigenvalue > 1. A larger value means fewer clusters and less variations explained. 1 is good default as represents the average size of eigenvalues
                     maxclus=None) # Can set this to the number of features we are willing to include 
# To get number of clusters + number of variables in each cluster + variance explained (i.e. Eigval1)
var_clust_model.varclus()
# Variable selection from each cluster based on the RS_Ratio - Select the feature from each cluster set that minimises RS_Ratio (i.e. that minimses 1-R2 ratio)
vc_results = var_clust_model.rsquare
# Pick the feature from each cluster
chosen_features = vc_results.loc[vc_results.groupby('Cluster').RS_Ratio.idxmin()]['Variable']
 
# Can then use this set of features to subset X to produce our X_train, y_train sets
not_chosen_idx = [i for i in vc_results.index.tolist() if i not in vc_results.groupby('Cluster').RS_Ratio.idxmin()]
dropped_features = vc_results.iloc[not_chosen_idx]['Variable']



#### mutual information #####
selector = SelectKBest(mutual_info_classif, k=20)
X_reduced = selector.fit_transform(X_train_sample, y_train_lc_sample)
cols = selector.get_support(indices=True)
selected_columns = X_train_sample.iloc[:,cols].columns.tolist()

#### K-Best Chi Squared ######
KBest = SelectKBest(chi2, k=10).fit(X, y) # select top 10 features based on chi-squared N.B. chi squared based on frequencies so input X must be positives only, else use different criteria 
