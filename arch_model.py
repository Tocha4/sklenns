import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import tensorflow as tf

model = MLPClassifier(hidden_layer_sizes=(1000), activation="relu", solver="adam", 
                      alpha=0.001, batch_size="auto", learning_rate="adaptive", 
                      learning_rate_init=0.03, power_t=0.5, max_iter=100, shuffle=True, 
                      random_state=None, tol=0, verbose=True, warm_start=False, 
                      momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                      validation_fraction=0, beta_1=0.9, beta_2=0.999, 
                      epsilon=1e-08, n_iter_no_change=100)

#%% LOAD DATA
train_x = pd.read_csv("../../input/microsoft-malware-prediction/train_numeric.csv", nrows=10**4)
train_x = train_x.drop(["dpi","dpi_square","MegaPixels"], axis=1)
train_x = train_x.dropna()
train_x, test_x, train_y, test_y = train_test_split(train_x.drop(["HasDetections", "MachineIdentifier"], axis=1),train_x["HasDetections"])

#%% SCALER
scaler = RobustScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

#%% CLUSTERING
clustering = KMeans(n_clusters=2, n_jobs=12)
clustering.fit(train_x)

train_cluster = clustering.predict(train_x)
test_cluster = clustering.predict(test_x)

#%% TRAINING

model.fit(train_x, train_y)

# TODO
#X = {name:values for name,values in zip(feature_columns, np.rot90(features[clusters==clr]))}
#clr_labels = labels[clusters==clr]
#
#test_input_fn = tf.estimator.inputs.numpy_input_fn(X, y=None, batch_size=len(clr_labels),
#                                                    num_epochs=1, shuffle=False, queue_capacity=1,
#                                                    num_threads=1)
#
#my_feature_columns = [tf.feature_column.numeric_column(key=key) for key in X.keys()]
#
##Create the graph
#classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
#                                        hidden_units=[50,4],
#                                        n_classes=2,
#                                        weight_column=None,
#                                        label_vocabulary=None,
#                                        optimizer='Adagrad',
#                                        activation_fn=tf.nn.tanh,
#                                        dropout=None,
#                                        input_layer_partitioner=None,
#                                        config=None,
#                                        warm_start_from=None,
#                                        loss_reduction=tf.losses.Reduction.SUM,
#                                        model_dir=f"./cluster_split/tf_model_dir_{clr}"                                        
#                                        )

#%% SCORE
# TODO

res_train = model.score(train_x,train_y)*100
res_test = model.score(test_x,test_y)*100

print(f"Accuracy test = {res_test:.2f}% | train = {res_train:.2f}%")