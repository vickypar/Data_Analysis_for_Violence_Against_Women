## ================================= TRANSFORMATION ============================ ##
# Normalize the data using Standard Scaler
minStandardScaler = preprocessing.StandardScaler()
x_train_obs = minStandardScaler.fit_transform(np.reshape(x_train_obs, (-1,1)))
x_test_obs = minStandardScaler.transform(np.reshape(x_test_obs, (-1,1)))

# Add the numeric feature after it has been normalized.
x_train = np.c_[x_train, x_train_obs] 
x_test = np.c_[x_test, x_test_obs] 
#print(x_train.shape)
#print(x_test.shape)


# Apply Principal Component Analysis
# The number of components is selected to be 23 since there is no loss of information
pca = PCA(n_components=23)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
