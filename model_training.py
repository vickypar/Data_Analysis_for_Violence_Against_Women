#=========== SELECT HYPERPARAMETERS =====================#
# In this phase, hyperparameters for each algorithm will be selected based on a validation set

# More specifically, training set will be further split into train and validation set
# the algorithm will be trained using the new training set and it will be tested using the validation set

import matplotlib.pyplot as plt
from sklearn import ensemble, metrics
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


x_train2, x_val, y_train2, y_val = model_selection.train_test_split(x_train, y_train, random_state = 0, stratify = y_train)

#======= Random Forest =========#
acc = []
for i in range (1,40,5):
    model = ensemble.RandomForestClassifier(criterion = 'gini', n_estimators = i, max_depth= 4, random_state = 0)

    model.fit(x_train2,y_train2)
    predict_test2 = model.predict(x_val)
    accAd = metrics.accuracy_score(y_val, predict_test2)
    acc.append(accAd)

plt.figure(0)
plt.title("Random Forest")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.plot(range(1,40,5),acc)
#print(max(acc))
#print(acc.index(max(acc)))

#====== XGBoost ================#
acc = []
for i in range (15,100,5):
    model = XGBClassifier(n_estimators=i,max_depth = 3)
    model.fit(x_train2,y_train2)
    predict_test2 = model.predict(x_val)
    acc.append(metrics.accuracy_score(y_val, predict_test2))

plt.figure(1)
plt.title("XGBoost")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.plot(range(15,100,5),acc)
#print(max(acc))
#print(acc.index(max(acc)))

#========= k-NN =================#
acc = []
for i in range (10,100,1):
    model = KNeighborsClassifier(n_neighbors= i, weights='distance', p=2)
    model.fit(x_train2,y_train2)
    predict_test2 = model.predict(x_val)
    accAd = metrics.accuracy_score(y_val, predict_test2)
    acc.append(accAd)

plt.figure(2)
plt.title("K-Nearest Neighbors")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.plot(range(10,100,1),acc)
#print(max(acc))
#print(acc.index(max(acc)))

#============ Gradient Boosting ======#
acc = []
for i in range (10,70,5):
    model = ensemble.GradientBoostingClassifier(n_estimators = i, random_state = 0)
    model.fit(x_train2,y_train2)
    predict_test2 = model.predict(x_val)
    accAd = metrics.accuracy_score(y_val, predict_test2)
    acc.append(accAd)

plt.figure(3)
plt.title("Gradient Boosting")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.plot(range(10,70,5),acc)
#print(max(acc))
#print(acc.index(max(acc)))

#=========== AdaBoost ================#
acc = []
for i in range (15,50,5):
    dt = tree.DecisionTreeClassifier(max_depth=4, random_state = 0)
    model = ensemble.AdaBoostClassifier(base_estimator = dt, n_estimators = i, algorithm= 'SAMME')
    model.fit(x_train2,y_train2)
    predict_test2 = model.predict(x_val)
    accAd = metrics.accuracy_score(y_val, predict_test2)
    acc.append(accAd)

plt.figure(4)
plt.title("AdaBoost")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.plot(range(15,50,5),acc)
print(max(acc))
print(acc.index(max(acc)))

# =============== DATA MINING ==================== ##
from sklearn import ensemble, metrics, linear_model, tree, neural_network, svm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


# ========== Define the models that will be used ===== #
# Random Forest classifier is trained 
modelRf = ensemble.RandomForestClassifier(criterion = 'gini', n_estimators = 6, max_depth= 4, random_state = 0)
# Logistic Regression Classifier
modelLR = linear_model.LogisticRegression(penalty = 'l2', multi_class='multinomial', random_state=0, solver = 'lbfgs', max_iter=1000)
#Gradient Boosting Classifier
modelGBC = ensemble.GradientBoostingClassifier(n_estimators = 40, random_state = 0)
#AdaBoost
dt = tree.DecisionTreeClassifier(max_depth=4, random_state = 0)
modelAda = ensemble.AdaBoostClassifier(base_estimator = dt, n_estimators = 25, algorithm= 'SAMME', random_state=0)
#Neural Networks
modelNNs = neural_network.MLPClassifier(hidden_layer_sizes= (50,50), activation = "relu", solver = "lbfgs", tol = 0.001, max_iter=10000)
#XGBoost
modelXGB = XGBClassifier(n_estimators = 40, max_depth = 3, random_state = 0)
#K-NN
modelKnn = KNeighborsClassifier(n_neighbors= 50, weights='distance', p=2)

# Define a list of algorithms' names and the algorithms
algo_names = ["Random Forest ", "Logistic Regression ", "Gradient Boosting Classifier ", "AdaBoost Classifier ", "Neural Networks ", "XGBoost", "K-Nearest Neighbors"]
algorithms = [modelRf, modelLR, modelGBC, modelAda, modelNNs, modelXGB, modelKnn]

# Preparation of the evaluation plot
labels = ['Accuracy', 'Precision']
x = np.arange(len(labels))  # the label locations
width = 0.8  # the width of the bars
fig, ax = plt.subplots(figsize=(14,8))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Evaluation Metrics')
ax.set_title('Evaluation of Algorithms')
v = [-3,-2,-1,0,1,2,3]

#Train each algorithm in the training set and evaluate it using the test set
#in terms of accuracy, precision, recall and f1-score.

for i in range(0,len(algorithms)):
    algorithms[i].fit(x_train, y_train)
    y_pred = algorithms[i].predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    pre = metrics.precision_score(y_test, y_pred, average ="weighted", zero_division=1)
    rec = metrics.recall_score(y_test, y_pred, average="macro")
    f1  = metrics.f1_score(y_test, y_pred, average="macro")
    metr = [acc, pre]
    rect = ax.bar(x + v[i]*width/7, metr, width/7, label=algo_names[i])
    ax.bar_label(rect)
    print(algo_names[i])
    print("Accuracy: %2f" % acc)
    print("Precision: %2f" % pre)
    print("Recall: %2f" % rec)
    print("F1 score: %2f" % f1)
    print("=================================================")

# ============ Plot the bar chart ===========================================
plt.xticks(x, labels)
plt.ylim(0,1)
ax.legend()
fig.tight_layout()
plt.show()
