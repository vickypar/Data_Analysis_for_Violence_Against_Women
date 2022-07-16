## ============================= PREPROCESSING =================================== ##
from sklearn import preprocessing, model_selection
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

X = final_df.loc[:, final_df.columns != "OUTCOME: Outcome"]
y = final_df["OUTCOME: Outcome"]

# Split train and test set
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 0, stratify = y)

# ============= Imputation ================== #
# Imputation for completing missing data for numeric "OBS_VALUE"
imp_mean = SimpleImputer(missing_values=np.nan, strategy = 'mean')
x_train['OBS_VALUE'] = imp_mean.fit_transform(x_train.loc[:,x_train.columns == 'OBS_VALUE'])
x_test['OBS_VALUE'] = imp_mean.transform(x_test.loc[:,x_test.columns == 'OBS_VALUE'])

# Imputation for completing missing data for categorical features
imp_freq = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
x_train = imp_freq.fit_transform(x_train)
x_test = imp_freq.transform(x_test)

#Keep the numeric features in separate columns to add it later (after one-hot encoding)
x_train_obs = x_train[:,-1]
x_test_obs = x_test[:,-1]


# ============== One-hot vectors ============= #
# Encode only categorical features as a one-hot numeric array
one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
x_train = one_hot_encoder.fit_transform(x_train[:,0:8]).toarray()
x_test = one_hot_encoder.transform(x_test[:,0:8]).toarray()


# Encode target labels with value between 0 and n_classes-1.
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
#print(label_encoder.classes_)
y_test = label_encoder.transform(y_test)
#label_encoder.inverse_transform([0, 0, 1, 2])
