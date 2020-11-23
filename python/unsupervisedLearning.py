from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
import matplotlib.pyplot as plt

### K-Means ######

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

range_n_clusters = range(10,60,10)
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters = n_clusters)
    preds = clusterer.fit_predict(df)
    centers = clusterer.cluster_centers_
    score = silhouette_score(df, preds, metric = 'euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score) )


# To measure inertia to gauge optimum clusters to use - inspect resultant graph #

k_clusts = range(10, 15)
inertias = []
for k in k_clusts:
    model = KMeans(n_clusters = k) # instantiate a model
    model.fit(log_channel_df)
    inertia = model.inertia_
    inertias.append(inertia)
_ = plt.plot(ks, inertias, '-o')
_ = plt.xlabel('number of clusters, k')
_ = plt.ylabel('inertia')
_ = plt.xticks(ks)
plt.show()


# K Means in action - uses make_pipeline() or Pipeline() framework # N.B. Needs to be in the logical order 

# Pipeline() 
    # explicity give each step a name, so know how to access each part
    # pipe = Pipeline([('vec', CountVectorizer()), ('clf', LogisticRegression()]) # could then access logistic model using clf__C

# make_pipeline()
    # shorter and arguably more readable notation
    # names are auto-generated using a straightforward rule (lowercase name of an estimator
    # e.g. 'LogisticRegression' becomes 'logisticregression__C'

from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import P 
n_clusters_km = 15
scaler = StandardScaler() 
kmean  = KMeans(n_clusters= n_clusters_km)   # or can add .fit() at this point and not use pipeline
pipeline = make_pipeline(scaler, kmean)     # needs to be in the logical order i.e scale first then cluster 
pipeline.fit(df) 
labels = pipeline[1].predict(df)              # gives you cluster labels - not entirely sure if should be [0] or [1] 
centroid_df = pipeline[1].cluster_centers_ # gives K centroid coordinates i.e. mean value per attribute that defines the centroid
km_df = pd.DataFrame(data=centroid_df.T, columns=range(1,16), index=df.columns)

# Hierarchical using sklearn# 
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(pipeline[1].cluster_centers_.T)
ac = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(km.cluster_centers_.T)
clust_out = pd.DataFrame({'cluster': ac.labels_, 'channel': df3.columns}).sort_values(['cluster', 'channel'])
clust_out

########### Using scipy ###################
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram # hierarchical
from scipy.cluster.vq import kmeans, vq # kmeans

# easy, simple example #
x_coordinates = list(np.random.randint(10, size = 10))
y_coordinates = list(np.random.randint(10, size = 10))
df = pd.DataFrame({'x_coord' : x_coordinates, 'y_coord' : y_coordinates})
Z = linkage(df, 'ward') # linkage object created i.e. uses 'ward' here to compute distances between intermediate clusters
df['cluster_labels']= fcluster(Z, 2, criterion = 'maxclust')
sns.scatterplot(x = 'x_coord', y = 'y_coord', hue = 'cluster_labels', data = df) # hue argument associates clusters with different colours
plt.show()

# kmeans #
import random; random.seed((1000,2000))  # random centroid initialisation 
x = list(np.random.randint(100, size = 8)); x = [float(i) for i in x]
y = list(np.random.randint(100, size = 8)); y = [float(i) for i in y]
df = pd.DataFrame({'x_coord': x, 'y_coord': y})
centroids, distortion = kmeans(df, 3) # only accepts float/dbl - not int
df['cluster_labels'], _ = vq(df, centroids) # assigns obs
sns.scatterplot(x = 'x_coord', y='y_coord', hue='cluster_labels', data=df)
plt.show()
print(df.groupby('cluster_labels')['ID'].count()) # number of observations per cluster
print(df.groupby('cluster_labels')['salary'].mean()) # mean salary per cluster

# hierarchical # 

distance_matrix = linkage(df[['x_scaled', 'y_scaled']], method = 'ward', metric = 'euclidean') # create distance matrix - linkage() performs hierarchical clustering
# method = 'complete' ; the distance between clusters is the distance between the furthest points of the clusters    
# method = 'single' ; distance between clusters is the distance between the closest points of the clusters
df['cluster_labels'] = fcluster(distance_matrix, height = 2, criterion = 'maxclust') # assigns obs a cluster label - decide on height value (intermediate level) to label clusters on
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data = df)
plt.show()
colors = {1:'red', 2:'blue'}
df.plot.scatter(x = 'x_scaled', y = 'y_scaled', c = df['cluster_labels'].apply(lambda x: colors[x]))
plt.show()
# dendogram
from scipy.cluster.hierarchy import dendrogram
dn = dendrogram(distance_matrix) # labels = , leaf_rotation=90, leaf_font_size=6 etc. 
plt.show()

# t-SNE #
from sklearn.manifold import TSNE 
model = TSNE(learning_rate=100)     # wrong learning_rate will show all points bunched together - normally 50-1000
model.fit_transform(df)
x = transformed[:,0]       # select the 0th feature
y = transformed[:,1]       # select the 1st feature 
plt.scatter(x, y, c = species)
plt.show() 