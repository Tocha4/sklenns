import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.cluster import MeanShift, KMeans, AgglomerativeClustering
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#%% MODEL
#model = GradientBoostingClassifier(loss="deviance", learning_rate=0.1, n_estimators=100, subsample=1.0, 
#                                   criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1,
#                                   min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 
#                                   min_impurity_split=None, init=None, random_state=None, max_features=None, 
#                                   verbose=1, max_leaf_nodes=None, warm_start=False, presort="auto", 
#                                   validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

model = SGDClassifier(loss="modified_huber", penalty="l2", alpha=0.05, l1_ratio=0.15, fit_intercept=True, 
                      max_iter=None, tol=None, shuffle=True, verbose=1, epsilon=0.1, n_jobs=4, 
                      random_state=42, learning_rate="optimal", eta0=0.0, power_t=0.5, early_stopping=False, 
                      validation_fraction=0.1, n_iter_no_change=50, class_weight=None, warm_start=False, 
                      average=False, n_iter=100)


#%% LOAD TRAIN DATASET
X = pd.read_csv("../../input/microsoft-malware-prediction/train_numeric.csv", chunksize=5*10**4).get_chunk()
X = X.drop(["dpi","dpi_square","MegaPixels", "MachineIdentifier"], axis=1)
X = X.dropna()
#X = X.drop(["MachineIdentifier"], axis=1)
y = X["HasDetections"]
X = X.drop(["HasDetections"], axis=1)

#%% SPLIT THE DATASET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#%% RBFSampler
#rbf_feature = RBFSampler(gamma=0.1, n_components=200, random_state=42)
#X_train = rbf_feature.fit_transform(X_train)
#X_test = rbf_feature.transform(X_test)

#%% SCALING AND TRANSFORMING
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% WITHOUT CLUSTER
estimator = model#LinearSVC(C=0.1, loss="hinge", tol=1e-5)
estimator.fit(X_train, y_train)

res = estimator.score(X_test, y_test)*100
res_train = estimator.score(X_train, y_train)*100
proba = estimator.predict_proba(X_train)
proba_test = estimator.predict_proba(X_test)
print(f"Cluster-1: size = {len(y_train)} | C = 0.1 | Accuracy test: {res:.2f}% | train: {res_train:.2f}%\n")

#%% CLUSTERING
#clustering = KMeans(n_clusters=2, tol=1).fit(X_train)
#train_cluster = clustering.predict(X_train)
#test_cluster = clustering.predict(X_test)
#
#for i in np.unique(train_cluster):
#    print(len(train_cluster[train_cluster==i]))
#
##%% SPLIT IN CLUSTERS
#def split_cluster(X_train,y_train, train_cluster): 
#    res = []
#    for i in range(len(np.unique(train_cluster))):
#        res.append([X_train[train_cluster==i], y_train[train_cluster==i]])
#    return res
#
#train = split_cluster(X_train,y_train, train_cluster)
#test = split_cluster(X_test,y_test, test_cluster)
##%% LinearSVC
#c = 0
#for tr,te in zip(train,test):
#
#    estimator = model#LinearSVC(C=0.1, loss="hinge", tol=1e-5)
#    estimator.fit(tr[0],tr[1])
#    
#    res = estimator.score(te[0],te[1])*100
#    proba_test = estimator.predict_proba(te[0])
#    res_train = estimator.score(tr[0],tr[1])*100
#    print(f"Cluster{c}: size = {len(tr[1])} | C = 0.1 | Accuracy test: {res:.2f}% | train: {res_train:.2f}%")
#    c += 1
    

#%%
bins = plt.hist(proba[:,0], bins=150)
bins = plt.hist(proba_test[:,0], bins=150)


#%% ANALYSIS OF RESULTS

proba_bin = estimator.predict(X_train)
layer_train = proba_bin!=y_train
proba_new = proba[layer_train]
bins = plt.hist(proba_new[:,0], bins=150)

proba_test = estimator.predict(X_test)






















