# ================= Predict Unknown Classes  - INFERENCE ======================================= #
## ============================= PREPROCESSING =================================== ##
from sklearn import preprocessing, ensemble, metrics, tree
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

x_train = final_df.loc[:, final_df.columns != "OUTCOME: Outcome"]
y_train = final_df["OUTCOME: Outcome"]
x_predict = x_unknown.loc[:, final_df.columns != "OUTCOME: Outcome"]

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_predict = x_predict.to_numpy()

# ============= Imputation ================== #
# Imputation for completing missing data for numeric "OBS_VALUE"
imp_mean = SimpleImputer(missing_values=np.nan, strategy = 'mean')
x_train_obs = imp_mean.fit_transform(np.reshape(x_train[:,-1],(-1,1)))
x_predict_obs = imp_mean.transform(np.reshape(x_predict[:,-1],(-1,1)))

# Imputation for completing missing data for categorical features
imp_freq = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
x_train = imp_freq.fit_transform(x_train)
x_predict = imp_freq.transform(x_predict)

pd.DataFrame(x_train, columns = ['TIME_PERIOD','GEO_PICT','TOPIC','CONDITION','VIOLENCE_TYPE','PERPETRATOR','ACTUALITY','LIFEPER','OBS_VALUE']).append(pd.DataFrame(x_predict, columns = ['TIME_PERIOD','GEO_PICT','TOPIC','CONDITION','VIOLENCE_TYPE','PERPETRATOR','ACTUALITY','LIFEPER','OBS_VALUE'])).to_csv('final_dataset_after_imputing.csv', index=False)

# ============== One-hot vectors ============= #
# Encode only categorical features as a one-hot numeric array
one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
x_train = one_hot_encoder.fit_transform(x_train[:,0:8]).toarray()
x_predict = one_hot_encoder.transform(x_predict[:,0:8]).toarray()

# Encode target labels with value between 0 and n_classes-1.
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
#print(label_encoder.classes_)
#label_encoder.inverse_transform([0, 0, 1, 2])

## ================================= TRANSFORMATION ============================ ##
# Normalize the data using Standard Scaler
minStandardScaler = preprocessing.StandardScaler()
x_train_obs = minStandardScaler.fit_transform(np.reshape(x_train_obs, (-1,1)))
x_predict_obs = minStandardScaler.transform(np.reshape(x_predict_obs, (-1,1)))

# Add the numeric feature after it has been normalized.
x_train = np.c_[x_train, x_train_obs] 
x_predict = np.c_[x_predict, x_predict_obs] 
#print(x_train.shape)
#print(x_predict.shape)


# Apply Principal Component Analysis
# The number of components is selected to be 23 since there is no loss of information
pca = PCA(n_components=23)
x_train = pca.fit_transform(x_train)
x_predict = pca.transform(x_predict)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))


# =============== DATA MINING ==================== ##
# ========== Define the model that will be used ===== #
#AdaBoost
dt = tree.DecisionTreeClassifier(max_depth=4, random_state = 0)
modelAda = ensemble.AdaBoostClassifier(base_estimator = dt, n_estimators = 25, algorithm= 'SAMME', random_state=0)

#Train the algorithm in the training set and use it to predict the classes of the unknown test
#in terms of accuracy, precision, recall and f1-score.

modelAda.fit(x_train, y_train)
y_predicted = modelAda.predict(x_predict)
y_pred = label_encoder.inverse_transform(y_predicted)

y_pred_train = modelAda.predict(x_train)
print(metrics.accuracy_score(y_train, y_pred_train))

#============= Visualize Results ================#
#np.savetxt("Predictions.txt", y_pred, fmt='%s')
df = pd.DataFrame(y_pred)
df.value_counts().plot(kind='bar')
#df.value_counts().plot(kind='pie')
print(df.value_counts())

df = pd.DataFrame(label_encoder.inverse_transform(y_train)).append(pd.DataFrame(y_pred))
df.value_counts().plot(kind='bar')
print(df.value_counts())

